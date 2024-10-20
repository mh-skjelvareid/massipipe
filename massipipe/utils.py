# Imports
import json
import logging
from pathlib import Path
from typing import Union

import numpy as np
import pyproj
import pyproj.aoi
import pyproj.database
import rasterio
import spectral
from numpy.typing import ArrayLike, NDArray
from pyproj import CRS, Proj
from rasterio.plot import reshape_as_raster
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
        byte_order_missing_str = 'Mandatory parameter "byte order" missing from header file.'
        if str(e) == byte_order_missing_str and write_byte_order_if_missing:
            logging.debug('Writing "byte order = 0" to header file and retrying')
            try:
                with open(header_path, "a") as file:
                    file.write("byte order = 0\n")
            except OSError:
                logger.error(f"Error writing to header file {header_path}", exc_info=True)
                raise

            try:
                im_handle = spectral.io.envi.open(header_path, image_path)
            except Exception as e:
                logger.error(
                    f"Unsucessful when reading modified header file {header_path}",
                    exc_info=True,
                )
                raise

    # Read wavelengths if listed in metadata, set as empty array if not
    # NOTE: Calibration files ("gain"/"offset") don't include wavelength information
    if "wavelength" in im_handle.metadata:
        wl = np.array([float(i) for i in im_handle.metadata["wavelength"]])
    else:
        wl = np.array([])

    # Read data from disk
    image = np.array(im_handle.load())  # type: ignore

    # Returns
    return (image, wl, im_handle.metadata)


def read_envi_header(header_path: Union[Path, str]) -> tuple[dict, NDArray]:
    """Read ENVI header information and convert wavelengths to numeric array

    Parameters
    ----------
    header_path : Union[Path, str]
        Path to ENVI header

    Returns
    -------
    metadata: dict[str,str]
        All header information formatted as dict
    wl: NDArray
        If "wavelength" is present in header file, wavelengths are returned as numeric
        array. If not, an empty array is returned.
    """
    metadata = spectral.io.envi.read_envi_header(header_path)
    if "wavelength" in metadata:
        wl = np.array([float(i) for i in metadata["wavelength"]])
    else:
        wl = np.array([])
    return metadata, wl


def save_envi(header_path: Union[Path, str], image: NDArray, metadata: dict, **kwargs) -> None:
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
    spectral.envi.save_image(header_path, image, metadata=metadata, force=True, ext=None, **kwargs)


def array_to_header_string(num_array: ArrayLike, decimals: int = 3) -> str:
    """Convert numeric array to ENVI header string

    Parameters
    ----------
    num_array : ArrayLike
        Array or iterable (list, tuple, ...) of wavelengths
    decimals: int, default 3
        Number of decimals used in text output

    Returns
    -------
    header_str
        String with comma separated numeric values in curly braces.

    Examples
    --------
    >>> array_to_header_string([420.32, 500, 581.28849])
    '{420.320, 500.000, 581.288}'

    """
    num_array = np.atleast_1d(np.array(num_array))  # Ensure array format
    str_array = [f"{num:.{decimals}f}" for num in num_array]  # Convert each number to string
    header_str = "{" + ", ".join(str_array) + "}"  # Join into single string
    return header_str


def update_header_wavelengths(wavelengths: NDArray, header_path: Union[Path, str]) -> None:
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
    wl_str = array_to_header_string(wavelengths)
    header_dict["wavelength"] = wl_str
    spectral.io.envi.write_envi_header(header_path, header_dict)


def add_header_irradiance(irradiance: NDArray, header_path: Union[Path, str]) -> None:
    """Add irradiance information to ENVI header ("solar irradiance")

    The irradiance is converted to unit W/(m2*um) (i.e., input irradiance is divided by
    1000), since this is standard for ENVI files. The values are written to the header
    field "solar irradiance".

    Parameters
    ----------
    irradiance : NDArray
        Downwelling irradiance, shape (n_bands,), in units W/(m2*nm).
        The number of bands and the corresponding wavelengths for these bands
        should match that of the hyperspectral image (see "wavelength" in header file).
    header_path : Union[Path, str]
        Path to ENVI header file (typically radiance image that can be converted to
        reflectance using irradiance information).
    """
    header_path = Path(header_path)
    header_dict = spectral.io.envi.read_envi_header(header_path)
    irrad_str = array_to_header_string(irradiance / 1000)
    header_dict["solar irradiance"] = irrad_str
    spectral.io.envi.write_envi_header(header_path, header_dict)


def get_image_shape(image_path: Union[Path, str]) -> tuple[int, int, int]:
    """Get shape of image cube (lines, samples, bands)"""
    header = spectral.envi.read_envi_header(image_path)
    shape = (int(header["lines"]), int(header["samples"]), int(header["bands"]))
    return shape


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
    return savgol_filter(image, window_length=window_length, polyorder=polyorder, axis=axis)


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
    return int(np.argmin(abs(np.array(wl_array) - target_wl)))


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
    wl_ind = [closest_wl_index(hyspec_wl, wl) for wl in rgb_target_wl]  # type: ignore
    rgb_im = hyspec_im[:, :, wl_ind]
    rgb_wl = hyspec_wl[wl_ind]
    return rgb_im, rgb_wl


def percentile_stretch_image(
    image: NDArray,
    percentiles: tuple[float, float] = (2, 98),
    saturation_value: int = 2**12 - 1,
) -> NDArray:
    """Scale array values within percentile limits to range 0-255

    Parameters
    ----------
    image : NDArray
        Image, shape (n_lines, n_samples, n_bands)

    Returns
    -------
    image_stretched: NDArray, dtype = uint8
        Image with same shape as input.
        Image intensity values are stretched so that the lower
        percentile corresponds to 0 and the higher percentile corresponds
        to 255 (maximum value for unsigned 8-bit integer, typical for PNG/JPG).
        Pixels for which one or more bands are saturated are set to zero.
    """
    assert image.ndim == 3
    saturated = np.any(image >= saturation_value, axis=2)
    image_stretched = np.zeros_like(image, dtype=np.float64)

    for band_index in range(image.shape[2]):
        image_band = image[:, :, band_index]
        p_low, p_high = np.percentile(image_band[~saturated], percentiles)
        image_band[image_band < p_low] = p_low
        image_band[image_band > p_high] = p_high
        p_range = p_high - p_low
        image_stretched[:, :, band_index] = (image_band - p_low) * (255 / p_range)

    image_stretched[saturated] = 0
    return image_stretched.astype(np.uint8)


def convert_long_lat_to_utm(
    long: ArrayLike, lat: ArrayLike
) -> tuple[NDArray, NDArray, Union[int, None]]:
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
            west_lon_degree=np.min(long),
            south_lat_degree=np.min(lat),
            east_lon_degree=np.max(long),
            north_lat_degree=np.max(lat),
        ),
    )
    utm_crs = CRS.from_epsg(utm_crs_list[0].code)
    proj = Proj(utm_crs)
    UTMx, UTMy = proj(long, lat)

    return np.array(UTMx), np.array(UTMy), utm_crs.to_epsg()


def get_vis_ind(wl: NDArray, vis_band: tuple[float, float] = (400.0, 730.0)) -> NDArray:
    """Get indices of VIS band

    Parameters
    ----------
    wl : NDArray
        Array of wavelengths (monotonically increasing), in nm
    vis_band : tuple[float, float], default (400.0, 730.0)
        Lower and upper limit of visible light range, in nm

    Returns
    -------
    vis_ind: NDArray, boolean
        Boolean array, True where wl is within vis_band
    """
    return (wl >= vis_band[0]) & (wl <= vis_band[1])


def get_nir_ind(
    wl: NDArray,
    nir_band: tuple[float, float] = (740.0, 805.0),
    nir_ignore_band: tuple[float, float] = (753.0, 773.0),
) -> NDArray:
    """Get indices of NIR band

    Parameters
    ----------
    wl : NDArray
        Array of wavelengths (monotonically increasing)
    nir_band: tuple[float, float], default (740.0, 805.0)
        Lower and upper edge of near-infrared (NIR) band.
    nir_ignore_band: tuple [float, float], default (753.0, 773.0)
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


def save_png(rgb_image: NDArray, png_path: Union[Path, str]):
    """Save RGB image as PNG using rasterio / GDAL

    Parameters
    ----------
    rgb_image : NDArray, dtype uint8
        Image, shape (n_lines, n_samples, n_bands=3)
        Image bands must be ordered as RGB (band index 0 = red,
        band index 1 = green, band index 2 = blue),
        The image must already be scaled to uint8 values 0-255.
    png_path : Union[Path, str]
        Path to output PNG file.
    """
    assert (rgb_image.ndim == 3) and (rgb_image.shape[2] == 3)
    with rasterio.Env():
        with rasterio.open(
            png_path,
            "w",
            driver="PNG",
            height=rgb_image.shape[0],
            width=rgb_image.shape[1],
            count=3,
            dtype="uint8",
        ) as dst:
            dst.write(reshape_as_raster(rgb_image))


def read_json(json_path: Union[Path, str]) -> dict:
    """Read data saved in JSON file

    Parameters
    ----------
    json_path : Union[Path,str]
        Path to JSON file

    Returns
    -------
    data: dict
        Data from JSON file
    """
    with open(json_path, "r") as file:
        imu_data = json.load(file)
    return imu_data


def random_sample_image(image: NDArray, sample_frac=0.5, ignore_zeros: bool = True):
    """_summary_

    Parameters
    ----------
    image : NDArray
        Hyperspectral image, shape (n_rows, n_lines, n_bands)
    sample_frac : float, optional
        Number of samples expressed as a fraction of the total
        number of pixels in the image. Range: [0.0 - 1.0]
    ignore_zeros : bool, optional
        If True, ignore pixels which are zero across all channels

    Returns
    -------
    X: NDArray
        2D array of sampled spectra, shape (n_samples, n_bands)
    """

    # Create mask
    if ignore_zeros:
        mask = ~np.all(image == 0, axis=2)
    else:
        mask = np.ones(image.shape[0:2])

    # Calculate number of samples
    n_samp = np.int64(sample_frac * np.count_nonzero(mask))

    # Create random number generator
    rng = np.random.default_rng()
    samp = rng.choice(image[mask], size=n_samp, axis=0, replace=False)

    return samp


def unix2gps(unix_time):
    """Convert UNIX time to GPS time (both in seconds)
    UNIX-GPS time conversion code adapted from
    https://www.andrews.edu/~tzs/timeconv/timealgorithm.html by Håvard S. Løvås
    """
    gps_time = unix_time - 315964800
    nleaps = _count_leaps(gps_time, "unix2gps")
    gps_time += nleaps + (1 if unix_time % 1 != 0 else 0)
    return gps_time


def gps2unix(gps_time):
    """Convert GPS time to UNIX time (both in seconds)
    GPS-UNIX time conversion code adapted from
    https://www.andrews.edu/~tzs/timeconv/timealgorithm.html by Håvard S. Løvås
    """
    unix_time = gps_time + 315964800
    nleaps = _count_leaps(gps_time, "gps2unix")
    unix_time -= nleaps
    if gps_time in _get_leaps():
        unix_time += 0.5
    return unix_time


def _get_leaps():
    """List of leap seconds in GPS time"""
    return [
        46828800,
        78364801,
        109900802,
        173059203,
        252028804,
        315187205,
        346723206,
        393984007,
        425520008,
        457056009,
        504489610,
        551750411,
        599184012,
        820108813,
        914803214,
        1025136015,
        1119744016,
        1167264017,
    ]


def _count_leaps(gps_time, dir_flag):
    """Count number of leap seconds passed for given GPS time"""
    leaps = _get_leaps()
    nleaps = 0
    for leap in leaps:
        if dir_flag == "unix2gps":
            if gps_time >= leap - nleaps:
                nleaps += 1
        elif dir_flag == "gps2unix":
            if gps_time >= leap:
                nleaps += 1
        else:
            raise ValueError("Invalid Flag!")
    return nleaps


def get_nested_dict_value(nested_dict, *keys):
    """Get (nested) dictionary value if it exists

    Parameters
    ----------
    *keys
        One or multiple keys needed to access value in nested dict.

    Returns
    -------
    value
        Value if value is defined in dictionary, None if not.
        Note that "null" values in YAML also correspond to None (not defined).
    """
    current = nested_dict
    for key in keys:
        if key in current:
            current = current[key]
        else:
            return None
    return current


def irrad_uflicklike_to_si_nm(irrad: NDArray) -> NDArray:
    """Convert irradiance from uW/(cm2*um) to W/(m2*nm)"""
    return irrad / 100_000


def irrad_si_nm_to_uflicklike(irrad: NDArray) -> NDArray:
    """Convert irradiance from uW/(cm2*um) to W/(m2*nm)"""
    return irrad * 100_000


def irrad_si_nm_to_si_um(irrad: NDArray) -> NDArray:
    """Convert irradiance from W/(m2*nm) to W/(m2*um)"""
    return irrad * 1000


def irrad_si_um_to_uflicklike(irrad: NDArray) -> NDArray:
    """Convert irradiance from W/(m2*um) to uW/(cm2*um)"""
    return irrad * 100
