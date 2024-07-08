# Imports
import logging
from pathlib import Path
from typing import Union

import numpy as np
import pyproj
import spectral
from scipy.signal import savgol_filter

# Get loggger
logger = logging.getLogger(__name__)


def read_envi(
    header_path: Union[Path, str],
    image_path: Union[Path, str, None] = None,
    write_byte_order_if_missing=True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load image in ENVI format, including wavelength vector and other metadata

    Usage:
    ------
    (image,wl,rgb_ind,metadata) = read_envi(header_path,...)

    Arguments:
    ----------
    header_path: Path | str
        Path to ENVI file header.

    Keyword arguments:
    ------------------
    image_path: Path | str
        Path to ENVI data file, useful if data file is not found
        automatically from header file name (see spectral.io.envi.open).
    write_byte_order_if_missing: bool
        Flag to indicate if the string "byte order = 0" should be written
        to the header file in case of MissingEnviHeaderParameter error
        (byte order is required by the "spectral" library, but is missing
        in some Resonon ENVI files)

    Returns:
    --------
    image: np.ndarray
        image, shape (n_lines, n_samples, n_channels)
    wl: np.ndarray
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
    header_path: Union[Path, str], image: np.ndarray, metadata: dict, **kwargs
) -> None:
    """Save ENVI file with parameters compatible with Spectronon

    # Usage:
    save_envi(header_path,image,metadata)

    # Arguments:
    ------------
    header_path: Path | str
        Path to header file.
        Data file will be saved in the same location and with
        the same name, but without the '.hdr' extension.
    image: np.ndarray
        Numpy array with hyperspectral image
    metadata:
        Dict containing (updated) image metadata.
        See load_envi_image()

    Optional arguments:
    -------------------
    dtype:
        Data type for ENVI file. Follows numpy naming convention.
        Typically 'uint16' or 'single' (32-bit float)
        If None, dtype = image.dtype
    """

    # Save file
    spectral.envi.save_image(
        header_path, image, metadata=metadata, force=True, ext=None, **kwargs
    )


def wavelength_array_to_header_string(wavelengths: np.ndarray):
    """Convert numeric wavelength array to single ENVI-formatted string"""
    wl_str = [f"{wl:.3f}" for wl in wavelengths]  # Convert each number to string
    wl_str = "{" + ", ".join(wl_str) + "}"  # Join into single string
    return wl_str


def update_header_wavelengths(wavelengths: np.ndarray, header_path: Union[Path, str]):
    """Update ENVI header wavelengths"""
    header_path = Path(header_path)
    header_dict = spectral.io.envi.read_envi_header(header_path)
    wl_str = wavelength_array_to_header_string(wavelengths)
    header_dict["wavelength"] = wl_str
    spectral.io.envi.write_envi_header(header_path, header_dict)


def bin_image(
    image: np.ndarray,
    line_bin_size: int = 1,
    sample_bin_size: int = 1,
    channel_bin_size: int = 1,
    average: bool = True,
) -> np.ndarray:
    """Bin image cube (combine neighboring pixels)

    Arguments
    ---------
    image: np.ndarray
        Image formatted as 3D NumPy array, with shape
        (n_lines, n_samples, n_channels). If the original array
        is only 2D, extend it t0 3D by inserting a singleton axis.
        For example, for a "single-line image" with shape (900,600),
        use image = np.expand_dims(image,axis=0), resulting in shape
        (1,900,600).

    Keyword arguments:
    ------------------
    line_bin_size, sample_bin_size, channel_bin_size: int
        Bin size, i.e. number of neighboring pixels to merge,
        for line, sample and channel dimensions, respectively.
    average: bool
        Whether to use averaging across neighboring pixels.
        If false, neighboring pixels are simply summed. Note that
        this shifts the statistical distribution of pixel values.

    References:
    -----------
    Inspired by https://stackoverflow.com/a/36102436
    See also https://en.wikipedia.org/wiki/Pixel_binning

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
    image, window_length: int = 13, polyorder: int = 3, axis: int = 2
) -> np.ndarray:
    """Filter hyperspectral image using Savitzky-Golay filter with default arguments"""
    return savgol_filter(
        image, window_length=window_length, polyorder=polyorder, axis=axis
    )


def closest_wl_index(wl_array: np.ndarray, target_wl: Union[float, int]) -> int:
    """Get index in sampled wavelength array closest to target wavelength"""
    return np.argmin(abs(wl_array - target_wl))


def rgb_subset_from_hsi(
    hyspec_im: np.ndarray, hyspec_wl, rgb_target_wl=(650, 550, 450)
) -> tuple[np.ndarray, np.ndarray]:
    """Extract 3 bands from hyperspectral image representing red, green, blue

    Arguments:
    ----------
    hyspec_im: np.ndarray
        Hyperspectral image, shape (n_lines, n_samples, n_bands)
    hyspec_wl: np.ndarray
        Wavelengths for each band of hyperspectral image, in nm.
        Shape (n_bands,)

    Returns:
    --------
    rgb_im: np.ndarray
        3-band image representing red, green and blue color (in that order)
    rgb_wl: np.ndarray
        3-element vector with wavelengths (in nm) corresponding to
        each band of rgb_im.

    """
    wl_ind = [closest_wl_index(hyspec_wl, wl) for wl in rgb_target_wl]
    rgb_im = hyspec_im[:, :, wl_ind]
    rgb_wl = hyspec_wl[wl_ind]
    return rgb_im, rgb_wl


def convert_long_lat_to_utm(
    long: Union[float, np.ndarray], lat: Union[float, np.ndarray]
) -> tuple[Union[float, np.ndarray], Union[float, np.ndarray], int]:
    """Convert longitude and latitude coordinates (WGS84) to UTM

    # Input parameters:
    long:
        Longitude coordinate(s), scalar or array
    lat:
        Latitude coordinate(s), scalar or array

    Returns:
    UTMx:
        UTM x coordinate ("Easting"), scalar or array
    UTMy:
        UTM y coordinate ("Northing"), scalar or array
    UTM_epsg :
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
