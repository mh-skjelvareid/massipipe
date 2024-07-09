# Imports
import logging
from pathlib import Path
from typing import Union

import numpy as np
import pyproj
import spectral
from numpy.typing import ArrayLike, NDArray
from scipy.signal import savgol_filter

# Get loggger
logger = logging.getLogger(__name__)


def read_envi(
    header_path: Union[Path, str],
    image_path: Union[Path, str, None] = None,
    write_byte_order_if_missing=True,
) -> tuple[NDArray, NDArray, dict]:
    """Load image in ENVI format, including wavelength vector and other metadata

    Parameters
    ----------
    header_path: Path | str
        Path to ENVI file header.
    image_path: Path | str | None, default None
        Path to ENVI data file, useful if data file is not found
        automatically from header file name (see spectral.io.envi.open).
    write_byte_order_if_missing: bool, default True
        Flag to indicate if the string "byte order = 0" should be written
        to the header file in case of MissingEnviHeaderParameter error
        (byte order is required by the "spectral" library, but is missing
        in some Resonon ENVI files)

    Returns
    -------
    image: NDArray
        image, shape (n_lines, n_samples, n_channels)
    wl: NDArray
        Wavelength vector, shape (n_channels,). None if no wavelengths listed.
    metadata:  dict
        Image metadata (ENVI header content).
    """

    # Open image handle
    try:
        im_handle = spectral.io.envi.open(header_path, image_path)
    except spectral.io.envi.MissingEnviHeaderParameter as e:
        logging.debug(f"Header file has missing parameter: {header_path}")
        byte_order_missing_str = (
            'Mandatory parameter "byte order" missing from header file.'
        )
        if str(e) == byte_order_missing_str and write_byte_order_if_missing:
            logging.debug('Writing "byte order = 0" to header file and retrying')
            try:
                with open(header_path, "a") as file:
                    file.write("byte order = 0\n")
            except OSError:
                logger.error(
                    f"Error writing to header file {header_path}", exc_info=True
                )

            try:
                im_handle = spectral.io.envi.open(header_path, image_path)
            except Exception as e:
                logger.error(
                    f"Unsucessful when reading modified header file {header_path}",
                    exc_info=True,
                )
                return

    # Read wavelengths
    if "wavelength" in im_handle.metadata:
        wl = np.array([float(i) for i in im_handle.metadata["wavelength"]])
    else:
        wl = None

    # Read data from disk
    image = np.array(im_handle.load())

    # Returns
    return (image, wl, im_handle.metadata)


def save_envi(
    header_path: Union[Path, str], image: NDArray, metadata: dict, **kwargs
) -> None:
    """Save ENVI file with parameters compatible with Spectronon

    Parameters
    ----------
    header_path : Union[Path, str]
        Path to header file.
        Data file will be saved in the same location and with
        the same name, but without the '.hdr' extension.
    image : NDArray
        Hyperspectral image, shape (n_lines, n_samples, n_bands)
    metadata : dict
        Dict containing (updated) image metadata.
        See read_envi()
    """
    # Save file
    spectral.envi.save_image(
        header_path, image, metadata=metadata, force=True, ext=None, **kwargs
    )


def wavelength_array_to_header_string(wavelengths: ArrayLike) -> str:
    """Convert wavelength array to ENVI header string

    Parameters
    ----------
    wavelengths : ArrayLike
        Array of wavelengths

    Returns
    -------
    str
        Single string with wavelengths in curly braces,
        with 3 decimals.

    Examples
    --------
    >>> wavelength_array_to_header_string([420.32, 500, 581.28849])
    '{420.320, 500.000, 581.288}'

    """
    wl_str = [f"{wl:.3f}" for wl in wavelengths]  # Convert each number to string
    wl_str = "{" + ", ".join(wl_str) + "}"  # Join into single string
    return wl_str


def update_header_wavelengths(
    wavelengths: NDArray, header_path: Union[Path, str]
) -> None:
    """Update ENVI header wavelengths

    Parameters
    ----------
    wavelengths : NDArray
        Array of wavelengths.
    header_path : Union[Path, str]
        Path to ENVI header file.
    """
    header_path = Path(header_path)
    header_dict = spectral.io.envi.read_envi_header(header_path)
    wl_str = wavelength_array_to_header_string(wavelengths)
    header_dict["wavelength"] = wl_str
    spectral.io.envi.write_envi_header(header_path, header_dict)


def bin_image(
    image: NDArray,
    line_bin_size: int = 1,
    sample_bin_size: int = 1,
    channel_bin_size: int = 1,
    average: bool = True,
) -> NDArray:
    """Bin image cube (combine neighboring pixels)

    Parameters
    ----------
    image : NDArray
        Image formatted as 3D NumPy array, with shape
        (n_lines, n_samples, n_bands). If the original array
        is only 2D, extend it t0 3D by inserting a singleton axis.
        For example, for a "single-line image" with shape (900,600),
        use image = np.expand_dims(image,axis=0), resulting in shape
        (1,900,600).
    line_bin_size : int, default 1
        How many neighboring lines to combine (axis 0)
    sample_bin_size : int, default 1
        How many neighboring samples to combine (axis 1)
    channel_bin_size : int, default 1
        How many neighboring spectral channels to combine (axis 2)
    average : bool, default True
        Whether to use averaging across neighboring pixels.
        If false, neighboring pixels are simply summed. Note that
        this shifts the statistical distribution of pixel values.

    Returns
    -------
    NDArray
        Binned version of image. The shape in reduced along each axis
        by a factor corresponding to the bin size.

    References
    ----------
    - Inspired by https://stackoverflow.com/a/36102436
    - See also https://en.wikipedia.org/wiki/Pixel_binning
    """
    assert image.ndim == 3
    n_lines, n_samples, n_channels = image.shape
    assert (n_lines % line_bin_size) == 0
    assert (n_samples % sample_bin_size) == 0
    assert (n_channels % channel_bin_size) == 0

    n_lines_binned = n_lines // line_bin_size
    n_samples_binned = n_samples // sample_bin_size
    n_channels_binned = n_channels // channel_bin_size

    image = image.reshape(
        n_lines_binned,
        line_bin_size,
        n_samples_binned,
        sample_bin_size,
        n_channels_binned,
        channel_bin_size,
    )
    if average:
        image = np.mean(image, axis=(1, 3, 5))
    else:
        image = np.sum(image, axis=(1, 3, 5))

    return image


def savitzky_golay_filter(
    image: NDArray, window_length: int = 13, polyorder: int = 3, axis: int = 2
) -> NDArray:
    """Filter hyperspectral image using Savitzky-Golay filter

    Parameters
    ----------
    image : NDArray
        Image array, shape (n_lines, n_samples, n_bands)
    window_length : int, default 13
        Length of "local" window withing which a polynomial is fitted
        to the data.
    polyorder : int, default 3
        Order of fitted polynomial.
    axis : int, default 2
        Axis along which filtering is applied.

    Returns
    -------
    NDArray
        Filtered version of image.
    """
    return savgol_filter(
        image, window_length=window_length, polyorder=polyorder, axis=axis
    )


def closest_wl_index(wl_array: ArrayLike, target_wl: Union[float, int]) -> int:
    """Get index in sampled wavelength array closest to target wavelength

    Parameters
    ----------
    wl_array : ArrayLike
        Array of wavelengths
    target_wl : Union[float, int]
        Single target wavelength

    Returns
    -------
    int
        Index of element in wl_array which is closest to target_wl.

    Examples
    --------
    >>> closest_wl_index([420, 450, 470], 468)
    2
    """
    return np.argmin(abs(np.array(wl_array) - target_wl))


def rgb_subset_from_hsi(
    hyspec_im: NDArray, hyspec_wl: NDArray, rgb_target_wl: ArrayLike = (650, 550, 450)
) -> tuple[NDArray, NDArray]:
    """Extract 3 bands from hyperspectral image representing red, green, blue

    Parameters
    ----------
    hyspec_im : NDArray
        Hyperspectral image, shape (n_lines, n_samples, n_bands)
    hyspec_wl : NDArray
        Wavelengths for each band of hyperspectral image, in nm.
        Shape (n_bands,)
    rgb_target_wl : ArrayLike, default (650, 550, 450)
        Wavelengths (in nm) representing red, green and blue.

    Returns
    -------
    rgb_im: NDArray
        3-band image representing red, green and blue color (in that order)
    rgb_wl: NDArray
        3-element vector with wavelengths (in nm) corresponding to
        each band of rgb_im. Values correspond to the wavelengths in
        hyspec_wl that are closest to rgb_target_wl.
    """
    wl_ind = [closest_wl_index(hyspec_wl, wl) for wl in rgb_target_wl]
    rgb_im = hyspec_im[:, :, wl_ind]
    rgb_wl = hyspec_wl[wl_ind]
    return rgb_im, rgb_wl


def convert_long_lat_to_utm(
    long: ArrayLike, lat: ArrayLike
) -> tuple[NDArray, NDArray, int]:
    """Convert longitude and latitude coordinates (WGS84) to UTM

    Parameters
    ----------
    long : ArrayLike
        Longitude coordinate(s)
    lat : ArrayLike
        Latitude coordinate(s)

    Returns
    -------
    UTMx:NDArray
        UTM x coordinate(s) ("Easting")
    UTMy: NDArray
        UTM y coordinate(s) ("Northing")
    UTM_epsg : int
        EPSG code (integer) for UTM zone
    """
    utm_crs_list = pyproj.database.query_utm_crs_info(
        datum_name="WGS 84",
        area_of_interest=pyproj.aoi.AreaOfInterest(
            west_lon_degree=min(long),
            south_lat_degree=min(lat),
            east_lon_degree=max(long),
            north_lat_degree=max(lat),
        ),
    )
    utm_crs = pyproj.CRS.from_epsg(utm_crs_list[0].code)
    proj = pyproj.Proj(utm_crs)
    UTMx, UTMy = proj(long, lat)

    return UTMx, UTMy, utm_crs.to_epsg()


def get_nir_ind(
    wl: NDArray,
    nir_band: tuple[float] = (740.0, 805.0),
    nir_ignore_band: tuple[float] = (753.0, 773.0),
) -> NDArray:
    """Get indices of NIR band

    Parameters
    ----------
    nir_band: tuple[float], default (740.0, 805.0)
        Lower and upper edge of near-infrared (NIR) band.
    nir_ignore_band: tuple [float], default (753.0, 773.0)
        Lower and upper edge of band to ignore (not include in indices)
        with nir_band. Default value corresponds to O2 absorption band
        around 760 nm.

    Returns
    -------
    NDArray:
        Array with indices of NIR band wavelengths.

    Notes:
    ------
    - Default values are at relatively short wavelengths (just above visible)
    in order to generate a NIR image with high signal-no-noise level.
    The default nir_ignore_band is used to exclude the "A" Fraunhofer
    line (around 759 nm).

    """
    nir_ind = (wl >= nir_band[0]) & (wl <= nir_band[1])
    ignore_ind = (wl >= nir_ignore_band[0]) & (wl <= nir_ignore_band[1])
    nir_ind = nir_ind & ~ignore_ind
    return nir_ind
