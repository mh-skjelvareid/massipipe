# Imports
import json
import logging
import subprocess
import zipfile
from pathlib import Path
from typing import Iterable, Union

import numpy as np
import rasterio
from numpy.polynomial import Polynomial
from numpy.typing import ArrayLike, NDArray
from rasterio.crs import CRS
from rasterio.plot import reshape_as_raster
from rasterio.profiles import DefaultGTiffProfile
from rasterio.transform import Affine
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

import massipipe.utils as mpu

# Get logger
logger = logging.getLogger(__name__)


class RadianceCalibrationDataset:
    """A radiance calibration dataset for Resonon hyperspectral cameras.

    Attributes
    ----------
    calibration_file: Path
        Path to Imager Calibration Pack (*.icp) file
    calibration_dir: Path
        Folder into which all the data in the calibration_file
        is put (unzipped).
    gain_file_path: Path
        Path to "gain file", i.e. ENVI file with "gain spectra",
        per-wavelength values for converting from raw (digital numbers)
        data to radiance data (in physical units).
    dark_frame_paths: list[Path]
        List of paths to "dark frames", i.e. files with a single line / spectrum
        taken with no light incident on the sensor, taken with different gain
        and shutter values.

    Methods
    -------
    get_rad_conv_frame():
        Returns radiance conversion frame, shape (n_samples, n_channels)
    get_closest_dark_frame(gain,shutter)
        Returns dark frame which best matches given gain and shutter values

    """

    def __init__(
        self,
        calibration_file: Union[Path, str],
        calibration_dir_name: str = "radiance_calibration_frames",
    ):
        """Un-zip calibration file and create radiance calibration dataset

        Parameters
        ----------
        calibration_file: Path | str
            Path to *.icp Resonon "Imager Calibration Pack" file
        calibration_dir_name: str
            Name of subdirectory into which calibration frames are unzipped

        Raises
        -------
        zipfile.BadZipfile:
            If given *.icp file is not a valid zip file

        """
        # Register image calibration "pack" (*.icp) and check that it exists
        self.calibration_file = Path(calibration_file)
        assert self.calibration_file.exists()

        # Unzip into same directory
        self.calibration_dir = self.calibration_file.parent / calibration_dir_name
        self.calibration_dir.mkdir(exist_ok=True)
        self._unzip_calibration_file()

        # Register (single) gain curve file and multiple dark frame files
        self.gain_file_path = self.calibration_dir / "gain.bip.hdr"
        assert self.gain_file_path.exists()
        self.dark_frame_paths = list(
            self.calibration_dir.glob("offset*gain*shutter.bip.hdr")
        )

        # Get dark frame gain and shutter info from filenames
        self._get_dark_frames_gain_shutter()

        # Sort gain/shutter values and corresponding filenames
        self._sort_dark_frame_gains_shutters_paths()

    def _unzip_calibration_file(self, unzip_into_nonempty_dir: bool = False) -> None:
        """Unzip *.icp file (which is a zip file)"""
        if not unzip_into_nonempty_dir and any(list(self.calibration_dir.iterdir())):
            logger.info(f"Non-empty calibration directory {self.calibration_dir}")
            logger.info("Assuming calibration file already unzipped.")
            return
        try:
            with zipfile.ZipFile(self.calibration_file, mode="r") as zip_file:
                for filename in zip_file.namelist():
                    zip_file.extract(filename, self.calibration_dir)
        except zipfile.BadZipFile:
            logger.error(
                f"File {self.calibration_file} is not a valid ZIP file.", exc_info=True
            )
        except Exception:
            logger.error(
                "Unexpected error when extracting calibration file "
                f"{self.calibration_file}",
                exc_info=True,
            )

    def _get_dark_frames_gain_shutter(self) -> None:
        """Extract and save gain and shutter values for each dark frame"""
        # Example dark frame pattern:
        # offset_600bands_4095ceiling_5gain_900samples_75shutter.bip.hdr
        dark_frame_gains = []
        dark_frame_shutters = []
        for dark_frame_path in self.dark_frame_paths:
            # Strip file extensions, split on underscores, keep gain and shutter info
            _, _, _, gain_str, _, shutter_str = dark_frame_path.name.split(".")[
                0
            ].split("_")
            dark_frame_gains.append(int(gain_str[:-4]))
            dark_frame_shutters.append(int(shutter_str[:-7]))
        # Save as NumPy arrays
        self._dark_frame_gains = np.array(dark_frame_gains, dtype=float)
        self._dark_frame_shutters = np.array(dark_frame_shutters, dtype=float)

    def _sort_dark_frame_gains_shutters_paths(self) -> None:
        """Sort gain/shutter values and corresponding file names"""
        gain_shutter_path_sorted = sorted(
            zip(
                self._dark_frame_gains, self._dark_frame_shutters, self.dark_frame_paths
            )
        )
        self._dark_frame_gains = np.array(
            [gain for gain, _, _ in gain_shutter_path_sorted]
        )
        self._dark_frame_shutters = np.array(
            [shutter for _, shutter, _ in gain_shutter_path_sorted]
        )
        self.dark_frame_paths = [path for _, _, path in gain_shutter_path_sorted]

    def _get_closest_dark_frame_path(
        self, gain: Union[int, float], shutter: Union[int, float]
    ) -> tuple[Path, float, float]:
        """Search for dark frame with best matching gain and shutter values"""
        # First search for files with closest matching gain
        candidate_gains = np.unique(self._dark_frame_gains)
        closest_gain = candidate_gains[np.argmin(abs(candidate_gains - gain))]

        # Then search (in subset) for single file with closest matching shutter
        candidate_shutters = np.unique(
            self._dark_frame_shutters[self._dark_frame_gains == closest_gain]
        )
        closest_shutter = self._dark_frame_shutters[
            np.argmin(abs(candidate_shutters - shutter))
        ]

        # Return best match
        best_match_mask = (self._dark_frame_gains == closest_gain) & (
            self._dark_frame_shutters == closest_shutter
        )
        best_match_ind = np.nonzero(best_match_mask)[0]
        assert len(best_match_ind) == 1  # There should only be a single best match
        return self.dark_frame_paths[best_match_ind[0]], closest_gain, closest_shutter

    def get_closest_dark_frame(
        self, gain: Union[int, float], shutter: Union[int, float]
    ) -> tuple[NDArray, NDArray, dict, float, float]:
        """Get dark frame which most closely matches given gain and shutter values

        Parameters
        ----------
        gain : Union[int, float]
            Gain value used for search, typically gain value of image that should
            be converted from raw to radiance. Values follow Resonon convention
            used in header files++ (logarithmic values, 20 log10).
        shutter : Union[int, float]
            Shutter value used for search, typically shutter value of image that should
            be converted from raw to radiance. Values follow Resonon convention
            used in header files++ (unit: milliseconds).

        Returns
        -------
        frame: NDArray
            Single dark frame, shape (1,n_samples,n_channels)
        wl: NDArray
            Vector of wavelengths for each spectral channel (nanometers)
        metadata: dict
            Metadata from ENVI header, formatted as dictionary
        closest_gain: float
            Gain value for closest match among available dark frames
        closest_shutter: float
            Shutter value for closest match among available dark frames
        """
        closest_file, closest_gain, closest_shutter = self._get_closest_dark_frame_path(
            gain=gain, shutter=shutter
        )
        frame, wl, metadata = mpu.read_envi(closest_file)
        return frame, wl, metadata, closest_gain, closest_shutter

    def get_rad_conv_frame(self) -> tuple[NDArray, NDArray, dict]:
        """Read and return radiance conversion curve ("gain" file)

        Returns
        -------
        frame: NDArray
            Radiance conversion frame, shape (1,n_samples,n_channels)
        wl: NDArray
            Vector of wavelengths for each spectral channel (nanometers)
        metadata: dict
            Metadata from ENVI header, formatted as dictionary

        """
        return mpu.read_envi(self.gain_file_path)


class QuickLookProcessor:
    def __init__(
        self,
        rgb_wl: tuple[float, float, float] = (640.0, 550.0, 460.0),
        percentiles: tuple[float, float] = (2, 98),
        saturation_value: int = 2**12 - 1,
    ):
        """Initialize quicklook processor"""
        self.rgb_wl = rgb_wl
        self.percentiles = percentiles
        self.saturation_value = saturation_value

    def percentile_stretch_image(self, image: NDArray) -> NDArray:
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
        saturated = np.any(image >= self.saturation_value, axis=2)
        image_stretched = np.zeros_like(image, dtype=np.float64)

        for band_index in range(image.shape[2]):
            image_band = image[:, :, band_index]
            p_low, p_high = np.percentile(image_band[~saturated], self.percentiles)
            image_band[image_band < p_low] = p_low
            image_band[image_band > p_high] = p_high
            p_range = p_high - p_low
            image_stretched[:, :, band_index] = (image_band - p_low) * (255 / p_range)

        image_stretched[saturated] = 0
        return image_stretched.astype(np.uint8)

    def create_quicklook_image_file(
        self, raw_path: Union[Path, str], quicklook_path: Union[Path, str]
    ):
        """Create per-band percentile stretched RGB image file from raw hyspec image

        Parameters
        ----------
        raw_path : Union[Path, str]
            Path to raw hyperspectral image (header file)
        quicklook_path : Union[Path, str]
            Path to output PNG image file
        """
        image, wl, _ = mpu.read_envi(raw_path)
        rgb_image, _ = mpu.rgb_subset_from_hsi(image, wl, rgb_target_wl=self.rgb_wl)
        rgb_image = self.percentile_stretch_image(rgb_image)
        mpu.save_png(rgb_image, quicklook_path)


class RadianceConverter:
    """A class for converting raw hyperspectral images from Pika L cameras to radiance.

    Attributes
    ----------
    radiance_calibration_file: Path | str
        Path to Imager Calibration Pack (*.icp) file.
    rc_dataset: RadianceCalibrationDataset
        A radiance calibration dataset object representing the calibration
        data supplied in the *.icp file.
    rad_conv_frame: NDArray
        Frame representing conversion factors from raw data to radiance for
        every pixel and wavelength.
    rad_conv_metadata: dict
        ENVI metadata for rad_conv_frame

    Methods
    -------
    convert_raw_image_to_radiance(raw_image, raw_image_metadata):
        Convert single image (3D array) from raw to radiance
    convert_raw_file_to_radiance(raw_image, raw_image_metadata):
        Read raw file, convert, and save as radiance image

    Notes
    -----
    Most image sensors register some amount of "dark current", i.e. a signal
    which is present even though no photons enter the sensor. The dark current
    should be subtracted for the measurement to be as accurate as possible.
    The amount of dark current also depends on camera gain (signal amplification)
    and camera shutter (time available for the sensor to collect photons).
    The *.icp calibration file (ICP = "Imager Calibration Pack") is a ZIP archine
    which includes dark current measurements taken at different gain and shutter
    settings (raw images). To remove dark current from a new image, this set of
    dark current frames is searched, and the one which best matches the gain and
    shutter values of the new image is used as a reference. The dark frame
    needs to be scaled to account for differences in binning, gain and shutter
    between it and the new image. The dark frame is then subtracted from the image.

    To convert raw images from digital numbers to radiance, a physical quantity,
    the raw images are multiplied with a "radiance conversion frame". This frame
    represents "microflicks per digital number" for every pixel and every
    spectral channel, where the spectral radiance unit "flick" is defined as
    "watts per steradian per square centimeter of surface per micrometer of span
    in wavelength" (see https://en.wikipedia.org/wiki/Flick_(physics) ).
    The radiance conversion frame also needs to be scaled to account for differences
    in binning, gain and shutter.

    When both the dark current frame (DCF) and the radiance conversion frame (RCF)
    have been scaled to match the raw input image (IM), the radiance output image
    (OI) is given by OI = (IM-DCF)*RCF

    Note that the conversion assumes that the camera response is completely linear
    after dark current is removed. This may not be completely accurate, but is
    assumed to be within an acceptable margin of error.

    """

    def __init__(self, radiance_calibration_file: Union[Path, str]):
        """Create radiance converter object

        Parameters
        ----------
        radiance_calibration_file: Path | str
            Path to Imager Calibration Pack (*.icp) file.
            The *.icp file is a zip archive, and the file will be unzipped into
            a subfolder in the same folder containing the *.icp file.

        """

        self.radiance_calibration_file = Path(radiance_calibration_file)
        self.rc_dataset = RadianceCalibrationDataset(
            calibration_file=radiance_calibration_file
        )
        self.rad_conv_frame = np.array([])
        self.rad_conv_metadata = dict()
        self._get_rad_conv_frame()

    def _get_rad_conv_frame(self) -> None:
        """Read radiance conversion frame from file and save as attribute"""
        rad_conv_frame, _, rad_conv_metadata = self.rc_dataset.get_rad_conv_frame()
        assert rad_conv_metadata["sample binning"] == "1"
        assert rad_conv_metadata["spectral binning"] == "1"
        assert rad_conv_metadata["samples"] == "900"
        assert rad_conv_metadata["bands"] == "600"
        self.rad_conv_frame = rad_conv_frame
        self.rad_conv_metadata = rad_conv_metadata

    def _get_best_matching_dark_frame(
        self, raw_image_metadata: dict
    ) -> tuple[NDArray, dict]:
        """Get dark fram from calibration data that best matches input data"""
        dark_frame, _, dark_frame_metadata, _, _ = (
            self.rc_dataset.get_closest_dark_frame(
                gain=float(raw_image_metadata["gain"]),
                shutter=float(raw_image_metadata["shutter"]),
            )
        )
        return (dark_frame, dark_frame_metadata)

    def _scale_dark_frame(
        self,
        dark_frame: NDArray,
        dark_frame_metadata: dict,
        raw_image_metadata: dict,
    ) -> NDArray:
        """Scale dark frame to match binning for input image"""
        assert dark_frame_metadata["sample binning"] == "1"
        assert dark_frame_metadata["spectral binning"] == "1"
        binning_factor = float(raw_image_metadata["sample binning"]) * float(
            raw_image_metadata["spectral binning"]
        )
        dark_frame = mpu.bin_image(
            dark_frame,
            sample_bin_size=int(raw_image_metadata["sample binning"]),
            channel_bin_size=int(raw_image_metadata["spectral binning"]),
        )
        dark_frame = dark_frame * binning_factor
        # NOTE: Dark frame not scaled based on differences in gain and shutter because
        # the best matching dark frame has (approx.) the same values already.

        return dark_frame

    def _scale_rad_conv_frame(self, raw_image_metadata: dict) -> NDArray:
        """Scale radiance conversion frame to match input binning, gain and shutter"""
        # Scaling due to binning
        binning_factor = 1.0 / (
            float(raw_image_metadata["sample binning"])
            * float(raw_image_metadata["spectral binning"])
        )

        # Scaling due to gain differences
        rad_conv_gain = 10 ** (float(self.rad_conv_metadata["gain"]) / 20.0)
        input_gain = 10 ** (float(raw_image_metadata["gain"]) / 20.0)
        gain_factor = rad_conv_gain / input_gain

        # Scaling due to shutter differences
        rad_conv_shutter = float(self.rad_conv_metadata["shutter"])
        input_shutter = float(raw_image_metadata["shutter"])
        shutter_factor = rad_conv_shutter / input_shutter

        # Bin (average) radiance conversion frame to have same dimensions as input
        rad_conv_frame = mpu.bin_image(
            self.rad_conv_frame,
            sample_bin_size=int(raw_image_metadata["sample binning"]),
            channel_bin_size=int(raw_image_metadata["spectral binning"]),
            average=True,
        )

        # Combine factors and scale frame
        scaling_factor = binning_factor * gain_factor * shutter_factor
        rad_conv_frame = rad_conv_frame * scaling_factor

        return rad_conv_frame

    def convert_raw_image_to_radiance(
        self,
        raw_image: NDArray,
        raw_image_metadata: dict,
        set_saturated_pixels_to_zero: bool = True,
        saturation_value: int = 2**12 - 1,
    ) -> NDArray:
        """Convert raw image (3d array) to radiance image

        Parameters
        ----------
        raw_image: NDArray
            Raw hyperspectral image, shape (n_lines, n_samples, n_channels)
            The image is assumed to have been created by a Resonon Pika L
            camera, which has 900 spatial pixels x 600 spectral channels
            before any binning has been applied. Typically, spectral binning
            with a bin size of 2 is applied during image aqusition, resulting
            in images with shape (n_lines, 900, 300). It is assumed that no
            spectral or spatial (sample) cropping has been applied. Where binning
            has been applied, it is assumed that
                - n_samples*sample_bin_size = 900
                - n_channels*channel_bin_size = 600
        raw_image_metadata: dict
            ENVI metadata formatted as dict.
            See spectral.io.envi.open()

        Returns
        --------
        radiance_image: NDArray (int16, microflicks)
            Radiance image with same shape as raw image, with spectral radiance
            in units of microflicks = 10e-5 W/(m2*nm). Microflicks are used
            to be consistent with Resonon formatting, and because microflick
            values typically are in a range suitable for (memory-efficient)
            encoding as 16-bit unsigned integer.

        Raises
        -------
        ValueError:
            In case the raw image does not have the expected dimensions.

        References:
        -----------
        - ["flick" unit](https://en.wikipedia.org/wiki/Flick_(physics))
        """
        # Check input dimensions
        if (
            int(raw_image_metadata["samples"])
            * int(raw_image_metadata["sample binning"])
            != 900
        ):
            raise ValueError(
                "Sample count and binning does not correspond to "
                "900 samples in the original image."
            )
        if (
            int(raw_image_metadata["bands"])
            * int(raw_image_metadata["spectral binning"])
            != 600
        ):
            raise ValueError(
                "Spectral band count and binning does not correspond to "
                "600 spectral bands in the original image."
            )

        # Get dark frame and radiance conversion frames scaled to input image
        dark_frame, dark_frame_metadata = self._get_best_matching_dark_frame(
            raw_image_metadata
        )
        dark_frame = self._scale_dark_frame(
            dark_frame, dark_frame_metadata, raw_image_metadata
        )
        rad_conv_frame = self._scale_rad_conv_frame(raw_image_metadata)

        # Flip frames if necessary
        if ("flip radiometric calibration" in raw_image_metadata) and (
            raw_image_metadata["flip radiometric calibration"] == "True"
        ):
            dark_frame = np.flip(dark_frame, axis=1)
            rad_conv_frame = np.flip(rad_conv_frame, axis=1)

        # Subtract dark current and convert to radiance (microflicks)
        # Note: dark_frame and rad_conv_frame are implicitly "expanded" by broadcasting
        radiance_image = (raw_image - dark_frame) * rad_conv_frame

        # Set negative (non-physical) values to zero
        radiance_image[radiance_image < 0] = 0

        # Set saturated pixels to zero (optional)
        if set_saturated_pixels_to_zero:
            radiance_image[np.any(raw_image >= saturation_value, axis=2)] = 0

        # Convert to 16-bit integer format (more efficient for storage)
        return radiance_image.astype(np.uint16)

    def convert_raw_file_to_radiance(
        self,
        raw_header_path: Union[Path, str],
        radiance_header_path: Union[Path, str],
        interleave: str = "bip",
    ) -> None:
        """Read raw image file, convert to radiance, and save to file

        Parameters
        ----------
        raw_header_path: Path | str
            Path to raw hyperspectral image acquired with Resonon Pika L camera.
        radiance_header_path: Path | str
            Path to save converted radiance image to.
            The name of the header file should match the 'interleave' argument
            (default: bip), e.g. 'radiance_image.bip.hdr'
        interleave: str, {'bip','bil','bsq'}, default 'bip'
            String inticating how binary image file is organized.
            See spectral.io.envi.save_image()

        Notes
        -----
        The radiance image is saved with the same metadata as the raw image.
        """
        raw_image, _, raw_image_metadata = mpu.read_envi(raw_header_path)
        radiance_image = self.convert_raw_image_to_radiance(
            raw_image, raw_image_metadata
        )
        mpu.save_envi(
            radiance_header_path,
            radiance_image,
            raw_image_metadata,
            interleave=interleave,
        )


class IrradianceConverter:
    def __init__(
        self,
        irrad_cal_file: Union[Path, str],
        irrad_cal_dir_name: str = "downwelling_calibration_spectra",
        wl_min: Union[int, float, None] = 370,
        wl_max: Union[int, float, None] = 1000,
    ):
        """Initialize irradiance converter

        Parameters
        ----------
        irrad_cal_file : Union[Path, str]
            Path to downwelling irradiance calibration file.
        irrad_cal_dir_name : str, default "downwelling_calibration_spectra"
            Name of folder which calibration files will be unzipped into.
        wl_min : Union[int, float, None], default 370
            Shortest valid wavelength (nm) in irradiance spectrum.
        wl_max : Union[int, float, None], default 1000
            Longest valid wavelength (nm) in irradiance spectrum.
        """
        # Save calibration file path
        self.irrad_cal_file = Path(irrad_cal_file)
        assert self.irrad_cal_file.exists()

        # Unzip calibration frames into subdirectory
        self.irrad_cal_dir = self.irrad_cal_file.parent / irrad_cal_dir_name
        self.irrad_cal_dir.mkdir(exist_ok=True)
        self._unzip_irrad_cal_file()

        # Load calibration data
        self._load_cal_dark_and_sensitivity_spectra()

        # Set valid wavelength range
        self.wl_min = self._irrad_wl[0] if wl_min is None else wl_min
        self.wl_max = self._irrad_wl[-1] if wl_max is None else wl_max
        self._valid_wl_ind = (self._irrad_wl >= wl_min) & (self._irrad_wl <= wl_max)

    def _unzip_irrad_cal_file(self, unzip_into_nonempty_dir: bool = False) -> None:
        """Unzip *.dcp file (which is a zip file)"""
        if not unzip_into_nonempty_dir and any(list(self.irrad_cal_dir.iterdir())):
            logger.info(
                f"Non-empty downwelling calibration directory {self.irrad_cal_dir}"
            )
            logger.info(
                "Skipping unzipping of downwelling calibration file, "
                "assuming unzipping already done."
            )
            return
        try:
            with zipfile.ZipFile(self.irrad_cal_file, mode="r") as zip_file:
                for filename in zip_file.namelist():
                    zip_file.extract(filename, self.irrad_cal_dir)
        except zipfile.BadZipFile:
            logger.error(
                f"File {self.irrad_cal_file} is not a valid ZIP file.", exc_info=True
            )
        except Exception as e:
            logger.error(
                "Error while extracting downwelling calibration file "
                f"{self.irrad_cal_file}",
                exc_info=True,
            )

    def _load_cal_dark_and_sensitivity_spectra(self):
        """Load dark current and irradiance sensitivity spectra from cal. files"""
        # Define paths
        cal_dark_path = self.irrad_cal_dir / "offset.spec.hdr"
        irrad_sens_path = self.irrad_cal_dir / "gain.spec.hdr"
        assert cal_dark_path.exists()
        assert irrad_sens_path.exists()

        # Read from files
        cal_dark_spec, cal_dark_wl, cal_dark_metadata = mpu.read_envi(cal_dark_path)
        irrad_sens_spec, irrad_sens_wl, irrad_sens_metadata = mpu.read_envi(
            irrad_sens_path
        )

        # Save attributes
        assert np.array_equal(cal_dark_wl, irrad_sens_wl)
        self._irrad_wl = cal_dark_wl
        self._cal_dark_spec = np.squeeze(cal_dark_spec)  # Remove singleton axes
        self._cal_dark_metadata = cal_dark_metadata
        self._irrad_sens_spec = np.squeeze(irrad_sens_spec)  # Remove singleton axes
        self._irrad_sens_metadata = irrad_sens_metadata
        self._cal_dark_shutter = float(cal_dark_metadata["shutter"])
        self._irrad_sens_shutter = float(irrad_sens_metadata["shutter"])

    def convert_raw_spectrum_to_irradiance(
        self,
        raw_spec: NDArray,
        raw_metadata: dict,
        set_irradiance_outside_wl_limits_to_zero: bool = True,
        keep_original_dimensions: bool = True,
    ) -> NDArray:
        """_summary_

        Parameters
        ----------
        raw_spec : NDArray
            Raw spectrum
        raw_metadata : dict
            Dictionary with ENVI metadata for raw spectrum.
        set_irradiance_outside_wl_limits_to_zero : bool, default True
            Whether to set spectrum values outside wavelength limits to zero.
            If False, values outside valid range are treated the same as
            values inside valid range.
        keep_original_dimensions : bool, default True
            Whether to format output spectrum with same (singleton)
            dimensions as input spectrum. Can be useful for broadcasting.

        Returns
        -------
        irradiance_spectrum: NDArray, shape (n_wl,)
            Spectrom converted to spectral irradiance, unit W/(m2*nm)

        Notes
        -----
        The irradiance conversion spectrum is _inversely_ scaled with
        the input spectrum shutter. E.g., if the input spectrum has a higher shutter
        value than the calibration file (i.e. higher values per amount of photons),
        the conversion spectrum values are decreased to account for this.
        Dark current is assumed to be independent of shutter value.
        """

        original_input_dimensions = raw_spec.shape
        raw_spec = np.squeeze(raw_spec)
        assert raw_spec.ndim == 1

        # Scale conversion spectrum according to difference in shutter values
        raw_shutter = float(raw_metadata["shutter"])
        scaled_conv_spec = (
            self._irrad_sens_shutter / raw_shutter
        ) * self._irrad_sens_spec

        # Subtract dark current, multiply with radiance conversion spectrum
        # NOTE: Resonon irradiance unit is uW/(pi*cm2*um) = 10e-5 W/(pi*m2*nm)
        cal_irrad_spec = (raw_spec - self._cal_dark_spec) * scaled_conv_spec

        # Convert to standard spectral irradiance unit W/(m2*nm)
        cal_irrad_spec = cal_irrad_spec * (np.pi / 100_000)

        # Set spectrum outside wavelength limits to zero
        if set_irradiance_outside_wl_limits_to_zero:
            cal_irrad_spec[~self._valid_wl_ind] = 0

        if keep_original_dimensions:
            cal_irrad_spec = np.reshape(cal_irrad_spec, original_input_dimensions)

        return cal_irrad_spec

    def convert_raw_file_to_irradiance(
        self, raw_spec_path: Union[Path, str], irrad_spec_path: Union[Path, str]
    ):
        """Read raw spectrum, convert to irradiance, and save

        Parameters
        ----------
        raw_spec_path : Union[Path, str]
            Path to ENVI header file for raw spectrum
        irrad_spec_path : Union[Path, str]
            Path to ENVI header file for saving irradiance spectrum
        """
        raw_spec, _, spec_metadata = mpu.read_envi(raw_spec_path)
        irrad_spec = self.convert_raw_spectrum_to_irradiance(raw_spec, spec_metadata)
        spec_metadata["unit"] = "W/(m2*nm)"
        mpu.save_envi(irrad_spec_path, irrad_spec, spec_metadata)


class WavelengthCalibrator:
    def __init__(self):
        """Initialize wavelength calibrator"""
        self._fh_line_indices = None
        self._fh_wavelengths = None
        self._wl_poly_coeff = None
        self.reference_spectrum_path = None
        self.wl_cal = None
        self.max_wl_diff = None

        self.fraunhofer_wls = {
            "L": 382.04,
            "G": 430.78,
            "F": 486.13,
            "b1": 518.36,
            "D": 589.30,
            "C": 656.28,
            "B": 686.72,
            "A": 760.30,  # Not well defined (O2 band), approximate
            "Z": 822.70,
        }

    @staticmethod
    def _detect_absorption_lines(
        spec: NDArray,
        wl: NDArray,
        distance: int = 20,
        width: int = 5,
        rel_prominence: float = 0.1,
    ):
        """Detect absorption lines using local peak detection"""
        wl_550_ind = mpu.closest_wl_index(wl, 550)
        prominence = spec[wl_550_ind] * rel_prominence
        peak_indices, peak_properties = find_peaks(
            -spec, distance=distance, width=width, prominence=prominence
        )
        return peak_indices, peak_properties

    @staticmethod
    def _fit_wavelength_polynomial(
        sample_indices: NDArray, wavelengths: NDArray, n_samples: int
    ) -> tuple[NDArray, NDArray]:
        """Fit 2nd degree polynomial to set of (sample, wavelength) pairs

        Parameters
        ----------
        sample_indices: NDArray
            Indices of samples in a sampled spectrum, typically for spectral
            peaks / absorption lines with known wavelengths.
        wavelengths: NDArray
            Wavelengths (in physical units) corresponding to sample_indices.
        n_samples: int
            Total number of samples in sampled spectrum

        Returns
        --------
        wl_cal:
            Array of (calibrated) wavelengths, shape (n_samples,)
        wl_poly_coeff:
            Coefficients for 2nd degree polynomial, ordered from degree zero upward
            (index corresponds to polynomial degree)

        """
        polynomial_fitted = Polynomial.fit(
            sample_indices, wavelengths, deg=2, domain=[]
        )
        wl_cal = polynomial_fitted(np.arange(n_samples))
        wl_poly_coeff = polynomial_fitted.coef
        return wl_cal, wl_poly_coeff

    def _filter_fraunhofer_lines(
        self,
        line_indices: NDArray,
        orig_wl: NDArray,
        win_width_nm: Union[int, float] = 20,
    ):
        """Calibrate wavelength values from known Fraunhofer absorption lines

        Parameters
        ----------
        line_indices: NDArray
            Indices of samples in spectrum where a (potential) absorption
            line has been detected. Indices must be within the range
            (0, len(orig_wl))
        orig_wl: NDArray
            Original wavelength vector. Values are assumed to be "close enough"
            to be used to create search windows for each Fraunhofer line,
            typically within a few nm. Shape (n_samples,)
        win_width_nm: Union[int, float], default 20
            The width of the search windows in units of nanometers.

        Returns
        filtered_line_indices: NDArray
            Indices for absorption lines found close to Fraunhofer line
        fraunhofer_wavelengths: NDArray
            Corresponding Fraunhofer line wavelengths for filtered absorption lines

        """
        filtered_line_indices = []
        fraunhofer_wavelengths = []
        for fh_line_wl in self.fraunhofer_wls.values():
            # Find index of closest sample to Fraunhofer wavelength
            fh_wl_ind = mpu.closest_wl_index(orig_wl, fh_line_wl)

            # Calculate half window width in samples at current wavelength
            wl_resolution = orig_wl[fh_wl_ind + 1] - orig_wl[fh_wl_ind]
            win_half_width = round((0.5 * win_width_nm) / wl_resolution)

            # Calculate edges of search window
            win_low = fh_wl_ind - win_half_width
            win_high = fh_wl_ind + win_half_width

            # Find lines within search window, accept if single peak found
            peaks_in_window = line_indices[
                (line_indices >= win_low) & (line_indices <= win_high)
            ]
            if len(peaks_in_window) == 1:
                filtered_line_indices.append(peaks_in_window[0])
                fraunhofer_wavelengths.append(fh_line_wl)

        return np.array(filtered_line_indices), np.array(fraunhofer_wavelengths)

    def fit(self, spec: NDArray, wl_orig: NDArray):
        """_summary_

        Parameters
        ----------
        spec: NDArray
            Sampled radiance/irradiance spectrum, shape (n_samples,)
        wl_orig: NDArray
            Wavelengths corresponding to spectral samples, shape (n_samples)
            Wavelengths values are assumed to be close (within a few nm)
            to their true values.

        Raises
        ------
        ValueError
            Raised if less than 3 absorption lines found in data.
        """
        spec = np.squeeze(spec)
        assert spec.ndim == 1

        line_indices, _ = self._detect_absorption_lines(spec, wl_orig)
        fh_line_indices, fh_wavelengths = self._filter_fraunhofer_lines(
            line_indices, wl_orig
        )
        if len(fh_line_indices) < 3:
            raise ValueError(
                "Too low data quality: Less than 3 absorption lines found."
            )
        wl_cal, wl_poly_coeff = self._fit_wavelength_polynomial(
            fh_line_indices, fh_wavelengths, len(wl_orig)
        )

        self._fh_line_indices = fh_line_indices
        self._fh_wavelengths = fh_wavelengths
        self._wl_poly_coeff = wl_poly_coeff
        self.wl_cal = wl_cal
        self.max_wl_diff = np.max(abs(wl_cal - wl_orig))

    def fit_batch(self, spectrum_header_paths: Iterable[Union[Path, str]]):
        """Calibrate wavelength based on spectrum with highest SNR (among many)

        Parameters
        ----------
        spectrum_header_paths: Iterable[Path | str]
            Paths to multiple spectra. The spectrum with the highest maximum
            value will be used for wavelength calibration.
            Spectra are assumed to be ENVI files.

        """
        spectrum_header_paths = list(spectrum_header_paths)
        spectra = []
        for spectrum_path in spectrum_header_paths:
            spectrum_path = Path(spectrum_path)
            try:
                spec, wl, _ = mpu.read_envi(spectrum_path)
            except OSError:
                logger.warning(f"Error opening spectrum {spectrum_path}", exc_info=True)
                logger.warning("Skipping spectrum.")
            spectra.append(np.squeeze(spec))

        spectra = np.array(spectra)
        best_spec_ind = np.argmax(np.max(spectra, axis=1))
        cal_spec = spectra[best_spec_ind]
        self.reference_spectrum_path = str(spectrum_header_paths[best_spec_ind])

        self.fit(cal_spec, wl)

    def update_header_wavelengths(self, header_path: Union[Path, str]):
        """Update header file with calibrated wavelengths

        Parameters
        ----------
        header_path: Union[Path, str]
            Iterable with paths multiple spectra.

        """
        if self.wl_cal is None:
            raise AttributeError("Attribute wl_cal is not set - fit (calibrate) first.")
        mpu.update_header_wavelengths(self.wl_cal, header_path)


class ImuDataParser:
    @staticmethod
    def read_lcf_file(lcf_file_path: Union[str, Path], time_rel_to_file_start=True):
        """Read location files (.lcf) generated by Resonon Airborne Hyperspectral imager

        Parameters
        ----------
        lcf_file_path: Union[str, Path]
            Path to lcf file. Usually a "sidecar" file to an hyperspectral image
            with same "base" filename.
        time_rel_to_file_start: bool, default True
            Boolean indicating if first timestamp should be subtracted from each
            timestamp, making time relative to start of file.

        Returns
        -------
        lcf_data:
            Dictionary with keys describing the type of data, and data
            formatted as numpy arrays. All arrays have equal length.
            The 7 types of data:
            - 'time': System time in seconds, relative to some (unknown)
            starting point. Similar to "Unix time" (seconds since January 1. 1970),
            but values indicate starting point around 1980. The values are
            usually offset to make the first timestamp equal to zero.
            See flag time_rel_to_file_start.
            - 'roll': Roll angle in radians, positive for "right wing up"
            - 'pitch': Pitch angle in radians, positive nose up
            - 'yaw': (heading) in radians, zero at due North, PI/2 at due East
            - 'longitude': Longitude in decimal degrees, negative for west longitude
            - 'latitude': Latitude in decimal degrees, negative for southern hemisphere
            - 'altitude': Altitude in meters relative to the WGS-84 ellipsiod.

        Notes:
        - The LCF file format was shared by Casey Smith at Resonon on February 16. 2021.
        - The LCF specification was inherited from Space Computer Corp.
        """
        lcf_raw = np.loadtxt(lcf_file_path)
        column_headers = [
            "time",
            "roll",
            "pitch",
            "yaw",
            "longitude",
            "latitude",
            "altitude",
        ]
        lcf_data = {header: lcf_raw[:, i] for i, header in enumerate(column_headers)}

        if time_rel_to_file_start:
            lcf_data["time"] -= lcf_data["time"][0]

        return lcf_data

    @staticmethod
    def read_times_file(times_file_path: Union[Path, str], time_rel_to_file_start=True):
        """Read image line timestamps (.times) file generated by Resonon camera

        Parameters
        ----------
        times_file_path: Union[Path,str]
            Path to times file. Usually a "sidecar" file to an hyperspectral image
            with same "base" filename.
        time_rel_to_file_start: bool, default True
            Boolean indicating if times should be offset so that first
            timestamp is zero. If not, the original timestamp value is returned.

        Returns
        -------
        times:
            Numpy array containing timestamps for every line of the corresponding
            hyperspectral image. The timestamps are in units of seconds, and are
            relative to when the system started (values are usually within the
            0-10000 second range). If time_rel_to_file_start=True, the times
            are offset so that the first timestamp is zero.
            The first timestamp of the times file and the first timestamp of the
            corresponding lcf file (GPS/IMU data) are assumed to the
            recorded at exactly the same time. If both sets of timestamps are
            offset so that time is measured relative to the start of the file,
            the times can be used to calculate interpolated GPS/IMU values
            for each image line.

        """
        image_times = np.loadtxt(times_file_path)
        if time_rel_to_file_start:
            image_times = image_times - image_times[0]
        return image_times

    @staticmethod
    def interpolate_lcf_to_times(
        lcf_data: dict, image_times: NDArray, convert_to_list=True
    ) -> dict:
        """Interpolate LCF data to image line times

        Parameters
        ----------
        lcf_data : dict
            Dictionary with data read from *.lcf file
        image_times : NDArray
            Array of time stamps from *.times file
        convert_to_list : bool, default True
            Whether to convert interpolated LCF data to list
            (useful for saving data as JSON).
            If False, interpolated values in lcf_data_interp are
            formatted as NDArray.

        Returns
        -------
        lcf_data_interp: dict[list | NDArray]
            Version of lcf_data with all measured values
            interpolated to image timestamps.
        """
        lcf_data_interp = {}
        lcf_times = lcf_data["time"]
        for lcf_key, lcf_value in lcf_data.items():
            lcf_data_interp[lcf_key] = np.interp(image_times, lcf_times, lcf_value)
            if convert_to_list:
                lcf_data_interp[lcf_key] = lcf_data_interp[lcf_key].tolist()
        return lcf_data_interp

    def read_and_save_imu_data(
        self,
        lcf_path: Union[Path, str],
        times_path: Union[Path, str],
        json_path: Union[Path, str],
    ):
        """Parse *.lcf and *.times files and save as JSON

        Parameters
        ----------
        lcf_path : Union[Path, str]
            Path to *.lcf (IMU data) file
        times_path : Union[Path, str]
            Path to *.times (image line timestamps) file
        json_path : Union[Path, str]
            Path to output JSON file
        """
        lcf_data = self.read_lcf_file(lcf_path)
        times_data = self.read_times_file(times_path)
        lcf_data_interp = self.interpolate_lcf_to_times(lcf_data, times_data)

        with open(json_path, "w", encoding="utf-8") as write_file:
            json.dump(lcf_data_interp, write_file, ensure_ascii=False, indent=4)

    @staticmethod
    def read_imu_json_file(imu_json_path: Union[Path, str]) -> dict:
        """Read IMU data saved in JSON file

        Parameters
        ----------
        imu_json_path : Union[Path,str]
            Path to JSON file with IMU data

        Returns
        -------
        imu_data: dict
            IMU data
        """
        with open(imu_json_path, "r") as imu_file:
            imu_data = json.load(imu_file)
        return imu_data


class ReflectanceConverter:
    """A class for converting images from Resonon Pika L cameras to reflectance"""

    def __init__(
        self,
        wl_min: Union[int, float] = 400,
        wl_max: Union[int, float] = 930,
        irrad_spec_paths: Union[Iterable[Union[Path, str]], None] = None,
    ):
        """Initialize reflectance converter

        Parameters
        ----------
        wl_min : Union[int, float], default 400
            Minimum wavelength (nm) to include in reflectance image.
        wl_max : Union[int, float], default 930
            Maximum wavelength (nm) to include in reflectance image.
        irrad_spec_paths : Union[Iterable[Union[Path, str]], None], default None
            List of paths to irradiance spectra which can be used as reference
            spectra when convering radiance to irradiance.

        Notes
        ------
        The signal-to-noise ratio of both radiance images and irradiance
        spectra is generally lower at the low and high ends. When
        radiance is divided by noisy irradiance values close to zero, the
        noise can "blow up". Limiting the wavelength range can ensure
        that the reflectance images have more well-behaved values.
        """
        self.wl_min = float(wl_min)
        self.wl_max = float(wl_max)
        if irrad_spec_paths is not None:
            irrad_spec_mean, irrad_wl, irrad_spectra = self._get_mean_irrad_spec(
                irrad_spec_paths
            )
        else:
            irrad_spec_mean = np.array([])
            irrad_wl = np.array([])
            irrad_spectra = np.array([])
        self.ref_irrad_spec_mean = irrad_spec_mean
        self.ref_irrad_spec_wl = irrad_wl
        self.ref_irrad_spectra = irrad_spectra

    @staticmethod
    def _get_mean_irrad_spec(irrad_spec_paths):
        """Read irradiance spectra from file and calculate mean"""
        irrad_spectra = []
        for irrad_spec_path in irrad_spec_paths:
            if irrad_spec_path.exists():
                irrad_spec, irrad_wl, _ = mpu.read_envi(irrad_spec_path)
                irrad_spectra.append(irrad_spec.squeeze())
        irrad_spectra = np.array(irrad_spectra)
        irrad_spec_mean = np.mean(irrad_spectra, axis=0)
        return irrad_spec_mean, irrad_wl, irrad_spectra

    @staticmethod
    def conv_spec_with_gaussian(
        spec: NDArray, wl: NDArray, gauss_fwhm: float
    ) -> NDArray:
        """Convolve spectrum with Gaussian kernel to smooth / blur spectral details

        Parameters
        ----------
        spec: NDArray
            Input spectrum, shape (n_bands,)
        wl: NDArray, nanometers
            Wavelengths corresponding to each spectrum value, shape (n_bands,)
        gauss_fwhm: float
            "Full width half maximum" (FWHM) of Gaussian kernel used for
            smoothin the spectrum. FWHM is the width of the kernel in nanometers
            at the level where the kernel values are half of the maximum value.

        Returns
        -------
        spec_filtered: NDArray
            Filtered / smoothed version of spec, with same dimensions

        Notes
        -----
        When the kernel extends outside the data while filtering, edges are handled
        by repeating the nearest sampled value (edge value).

        """
        sigma_wl = gauss_fwhm * 0.588705  # sigma = FWHM / 2*sqrt(2*ln(2))
        dwl = np.mean(np.diff(wl))  # Downwelling wavelength sampling dist.
        sigma_pix = sigma_wl / dwl  # Sigma in units of spectral samples
        spec_filtered = gaussian_filter1d(input=spec, sigma=sigma_pix, mode="nearest")
        return spec_filtered

    @staticmethod
    def _interpolate_irrad_to_image_wl(
        irrad_spec: NDArray,
        irrad_wl: NDArray,
        image_wl: NDArray,
    ) -> NDArray:
        """Interpolate downwelling spectrum to image wavelengths"""
        return np.interp(x=image_wl, xp=irrad_wl, fp=irrad_spec)

    def convert_radiance_image_to_reflectance(
        self,
        rad_image: NDArray,
        rad_wl: NDArray,
        irrad_spec: NDArray,
        irrad_wl: NDArray,
        convolve_irradiance_with_gaussian: bool = True,
        gauss_fwhm: float = 3.5,  # TODO: Find "optimal" default value for Pika-L
        smooth_with_savitsky_golay=False,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Convert radiance image to reflectance using downwelling spectrum

        Parameters
        ----------
        rad_image: NDArray
            Spectral radiance image in units of microflicks = 10e-5 W/(sr*m2*nm)
            Shape (n_lines, n_samples, n_bands)
        rad_wl: NDArray
            Wavelengths (in nanometers) corresponding to each band in rad_image
        irrad_spec: NDArray
            Spectral irradiance in units of W/(m2*nm)
        irrad_wl: NDArray
            Wavelengths (in nanometers) corresponding to each band in irrad_spec
        convolve_irradiance_with_gaussian: bool, default True
            Indicate if irradiance spectrum should be smoothed with Gaussian kernel.
            This may be useful if irradiance is measured with a higher spectral
            resolution than radiance, and thus has sharper "spikes".
        gauss_fwhm: float, default 3.5
            Full-width-half-maximum for Gaussian kernel, in nanometers.
            Only used if convolve_irradiance_with_gaussian==True
        smooth_with_savitsky_golay: bool, default False
            Whether to smooth the reflectance spectra using a Savitzky-Golay filter

        Returns
        -------
        refl_image: NDArray
            Reflectance image, unitless. Shape is same as rad_image for the
            first 2 axes, but may be less than that of rad_image along
            3rd axis due to limiting of wavelength range.
        refl_wl: NDArray
            Wavelengths for reflectance image
        irrad_spec:
            Irradiance spectrum used for reflectance conversion.
            The spectrum has been interpolated to match the wavelengths
            of refl_image.

        """

        # Check that spectrum is 1D, then expand to 3D for broadcasting
        irrad_spec = np.squeeze(irrad_spec)
        assert irrad_spec.ndim == 1

        # Limit output wavelength range
        valid_image_wl_ind = (rad_wl >= self.wl_min) & (rad_wl <= self.wl_max)
        rad_wl = rad_wl[valid_image_wl_ind]
        rad_image = rad_image[:, :, valid_image_wl_ind]

        # Make irradiance spectrum compatible with image
        irrad_spec = irrad_spec * 100_000  # Convert from W/(m2*nm) to uW/(cm2*um)
        if convolve_irradiance_with_gaussian:
            irrad_spec = self.conv_spec_with_gaussian(irrad_spec, irrad_wl, gauss_fwhm)
        irrad_spec = self._interpolate_irrad_to_image_wl(irrad_spec, irrad_wl, rad_wl)
        irrad_spec = np.expand_dims(irrad_spec, axis=(0, 1))

        # Convert to reflectance, assuming Lambertian (perfectly diffuse) surface
        refl_image = np.pi * (
            rad_image.astype(np.float32) / irrad_spec.astype(np.float32)
        )
        refl_wl = rad_wl

        # Spectral smoothing (optional)
        if smooth_with_savitsky_golay:
            refl_image = mpu.savitzky_golay_filter(refl_image)

        return refl_image, refl_wl, irrad_spec

    def convert_radiance_file_to_reflectance(
        self,
        radiance_image_header: Union[Path, str],
        irradiance_header: Union[Path, str],
        reflectance_image_header: Union[Path, str],
        use_mean_ref_irrad_spec: bool = False,
        **kwargs,
    ):
        """Read radiance image from file, convert to reflectance and save

        Parameters
        ----------
        radiance_image_header: Union[Path, str]
            Path to header file for radiance image.
        irradiance_header: Union[Path, str]
            Path to ENVI file containing irradiance measurement
            corresponding to radiance image file.
            Not used if use_mean_ref_irrad_spec is True - in this
            case, it can be set to None.
        reflectance_image_header: Union[Path, str]
            Path to header file for (output) reflectance image.
            Binary file will be saved with same name, except .hdr extension.
        use_mean_ref_irrad_spec: bool, default False
            Whether to use mean of irradiance reference spectra (see __init__)
            rather than an irradiance spectrum recorded together with the
            radiance image. This may be useful in cases where the recorded
            irradiance spectra are missing, or have low quality, e.g. at low
            sun angles where movement of the UAV strongly affects the measurement.

        """
        rad_image, rad_wl, rad_meta = mpu.read_envi(radiance_image_header)
        if use_mean_ref_irrad_spec:
            if self.ref_irrad_spec_mean is None:
                raise ValueError("Missing reference irradiance spectra.")
            irrad_spec = self.ref_irrad_spec_mean
            irrad_wl = self.ref_irrad_spec_wl
        else:
            irrad_spec, irrad_wl, _ = mpu.read_envi(irradiance_header)
        refl_im, refl_wl, _ = self.convert_radiance_image_to_reflectance(
            rad_image, rad_wl, irrad_spec, irrad_wl
        )
        wl_str = mpu.wavelength_array_to_header_string(refl_wl)
        refl_meta = rad_meta
        refl_meta["wavelength"] = wl_str
        mpu.save_envi(reflectance_image_header, refl_im, refl_meta)


class FlatSpecGlintCorrector:
    def __init__(self, smooth_with_savitsky_golay: bool = True):
        """Initialize glint corrector

        Parameters
        ----------
        smooth_with_savitsky_golay : bool, default True
            Whether to smooth glint corrected images using a
            Savitsky-Golay filter.
        """
        self.smooth_with_savitsky_golay = smooth_with_savitsky_golay

    def remove_glint_flat_spec(
        self, refl_image: NDArray, refl_wl: NDArray, **kwargs
    ) -> NDArray:
        """Remove sun and sky glint from image assuming flat glint spectrum

        Parameters
        ----------
        refl_image: NDArray
            Reflectance image, shape (n_lines, n_samples, n_bands)
        refl_wl: NDArray
            Wavelengths (in nm) for each band in refl_image

        Returns
        --------
        refl_image_glint_corr: NDArray
            Glint corrected reflectance image, same shape as refl_image.
            The mean NIR value is subtracted from each spectrum in the input
            image. Thus, only the spectral baseline / offset is changed -
            the original spectral shape is maintained.

        Notes
        -----
        - The glint correction is based on the assumption that there is
        (approximately) no water-leaving radiance in the NIR spectral region.
        This is often the case, since NIR light is very effectively
        absorbed by water.
        - The glint correction also assumes that the sun and sky glint
        reflected in the water surface has a flat spectrum, so that the
        glint in the visible region can be estimated from the glint in the
        NIR region. This is usually _not_ exactly true, but the assumption
        can be "close enough" to still produce useful results.
        """
        nir_ind = mpu.get_nir_ind(refl_wl, **kwargs)
        nir_im = np.mean(refl_image[:, :, nir_ind], axis=2, keepdims=True)
        refl_image_glint_corr = refl_image - nir_im

        if self.smooth_with_savitsky_golay:
            refl_image_glint_corr = mpu.savitzky_golay_filter(
                refl_image_glint_corr, **kwargs
            )

        return refl_image_glint_corr

    def glint_correct_image_file(
        self,
        image_path: Union[Path, str],
        glint_corr_image_path: Union[Path, str],
        **kwargs,
    ):
        """Read reflectance file, apply glint correction, and save result

        Parameters
        ----------
        image_path : Union[Path, str]
            Path to hyperspectral image (ENVI header file)
        glint_corr_image_path : Union[Path, str]
            Path for saving output image (ENVI header file)

        """
        image, wl, metadata = mpu.read_envi(image_path)
        glint_corr_image = self.remove_glint_flat_spec(image, wl, **kwargs)
        mpu.save_envi(glint_corr_image_path, glint_corr_image, metadata)


class HedleyGlintCorrector:
    def __init__(self, smooth_with_savitsky_golay: bool = True):
        """Initialize glint corrector

        Parameters
        ----------
        smooth_with_savitsky_golay : bool, default True
            Whether to smooth glint corrected images using a
            Savitsky-Golay filter.
        """
        self.smooth_with_savitsky_golay = smooth_with_savitsky_golay
        self.b = None
        self.min_nir = None

    def fit_to_reference_images(
        self, reference_image_paths: list[Union[Path, str]]
    ) -> None:
        train_spec = []
        for ref_im_path in reference_image_paths:
            ref_im, wl, im_meta = mpu.read_envi(Path(ref_im_path))
            sampled_spectra = mpu.random_sample_image(ref_im)
            train_spec.append(sampled_spectra)
        train_spec = np.concat(train_spec)
        self.fit(train_spec, wl)

    def fit(self, train_spec: NDArray, wl: NDArray) -> None:
        """_summary_

        Parameters
        ----------
        train_spec : NDArray
            _description_
        wl : NDArray
            _description_
        """
        vis_ind = mpu.get_vis_ind(wl)
        nir_ind = mpu.get_nir_ind(wl)

        x = np.mean(train_spec[:, nir_ind], axis=1, keepdims=True)
        Y = train_spec[:, vis_ind]
        self.b = self.linear_regression_multiple_dependent_variables(x, Y)
        self.min_nir = np.percentile(x, q=2, axis=None)  # Using 2nd percentile as min.

    @staticmethod
    def linear_regression_multiple_dependent_variables(
        x: NDArray, Y: NDArray
    ) -> NDArray:
        """Compute linear regression slopes for multiple dependent variables

        Parameters
        ----------
        x : NDArray
            Single sampled variable, shape (n_samples,)
        Y : NDArray
            Set of sampled variables, shape (n_samples, n_x_variables)

        Returns
        -------
        b: NDArray
            Slopes for linear regression with y as independent variable
            as each column of Y as dependent variable.
            The name "b" follows the convention for linear functions, f(x) = a + b*x

        Notes
        -----
        - Implementation inspired by https://stackoverflow.com/questions/
        48105922/numpy-covariance-between-each-column-of-a-matrix-and-a-vector

        """
        assert Y.shape[0] == x.shape[0]

        # Compute variance of y and covariance between each column of X and y
        # NOTE: No need to scale each with N-1 as it will be cancelled when computing b
        x_zero_mean = x - x.mean()
        x_var = x_zero_mean.T @ x_zero_mean
        x_Y_cov = x_zero_mean.T @ (Y - Y.mean(axis=0))

        # Compute slopes of linear regression with y as independent variable
        b = x_Y_cov / x_var

        return b

    def remove_glint(self, image: NDArray, wl: NDArray, **kwargs) -> NDArray:
        """Remove sun and sky glint from image using fit linear model

        Parameters
        ----------
        image: NDArray
            Hyperspectral image, shape (n_lines, n_samples, n_bands)
        wl: NDArray
            Wavelengths (in nm) for each band in image

        Returns
        --------
        refl_image_glint_corr: NDArray
            Glint corrected reflectance image, same shape as refl_image.
            The mean NIR value is subtracted from each spectrum in the input
            image. Thus, only the spectral baseline / offset is changed -
            the original spectral shape is maintained.

        Notes
        -----
        - The glint correction is based on the assumption that there is
        (approximately) no water-leaving radiance in the NIR spectral region.
        This is often the case, since NIR light is very effectively
        absorbed by water.
        """
        # nir_ind = mpu.get_nir_ind(refl_wl, **kwargs)
        # nir_im = np.mean(refl_image[:, :, nir_ind], axis=2, keepdims=True)
        # refl_image_glint_corr = refl_image - nir_im

        # if self.smooth_with_savitsky_golay:
        #     refl_image_glint_corr = mpu.savitzky_golay_filter(
        #         refl_image_glint_corr, **kwargs
        #     )

        # return refl_image_glint_corr
        pass

    def glint_correct_image_file(
        self,
        image_path: Union[Path, str],
        glint_corr_image_path: Union[Path, str],
        **kwargs,
    ):
        """Read reflectance file, apply glint correction, and save result

        Parameters
        ----------
        image_path : Union[Path, str]
            Path to hyperspectral image (ENVI header file)
        glint_corr_image_path : Union[Path, str]
            Path for saving output image (ENVI header file)

        """
        image, wl, metadata = mpu.read_envi(image_path)
        glint_corr_image = self.remove_glint(image, wl, **kwargs)
        mpu.save_envi(glint_corr_image_path, glint_corr_image, metadata)


class ImageFlightMetadata:
    """

    Attributes
    ----------
    u_alongtrack:
        Unit vector (easting, northing) pointing along flight direction
    u_crosstrack:
        Unit vector (easting, northing) pointing left relative to
        flight direction. The direction is chosen to match that of
        the image coordinate system: Origin in upper left corner,
        down (increasing line number) corresponds to positive along-track
        direction, right (increasing sample number) corresponds to
        positive cross-track direction.


    """

    def __init__(
        self,
        imu_data: dict,
        image_shape: tuple[int, ...],
        camera_opening_angle: float = 36.5,
        pitch_offset: float = 0.0,
        roll_offset: float = 0.0,
        assume_square_pixels: bool = True,
        altitude_offset: float = 0.0,
        **kwargs,
    ):
        """Initialize image flight metadata object

        Parameters
        ----------
        imu_data : dict
            Dictionary with imu_data, as formatted by ImuDataParser
        image_shape : tuple[int]
            Shape of image, typically (n_lines,n_samples,n_bands)
        camera_opening_angle : float, default 36.5
            Full opening angle of camera, in degrees.
            Corresponds to angle between rays hitting leftmost and
            rightmost pixels of image.
        pitch_offset : float, default 0.0
            How much forward the camera is pointing relative to nadir
        roll_offset : float, default 0.0
            How much to the right ("right wing up") the camera is pointing
            relative to nadir.
        assume_square_pixels : bool, default True
            Whether to assume that the original image was acquired with
            flight parameters (flight speed, frame rate, altitude)
            that would produce square pixels. If true, the altitude of the
            camera is estimated from the shape of the image and the (along-track)
            swath length. This can be useful in cases where absolute altitude
            measurement of the camera IMU is not very accurate.
        altitude_offset : float, default 0.0
            Offset added to the estimated altitude. If the UAV was higher
            in reality than that estimated by the ImageFlightMetadata
            object, add a positive altitude_offset.
        """

        # Set input attributes
        self.imu_data = imu_data
        self.image_shape = image_shape[0:2]
        self.camera_opening_angle = camera_opening_angle * (np.pi / 180)
        self.pitch_offset = pitch_offset * (np.pi / 180)
        self.roll_offset = roll_offset * (np.pi / 180)
        self.altitude_offset = altitude_offset

        # Get UTM coordinates and CRS code
        utm_x, utm_y, utm_epsg = mpu.convert_long_lat_to_utm(
            imu_data["longitude"], imu_data["latitude"]
        )
        self.utm_x = utm_x
        self.utm_y = utm_y
        self.camera_origin = np.array([utm_x[0], utm_y[0]])
        self.utm_epsg = utm_epsg

        # Time-related attributes
        t_total, dt = self._calc_time_attributes()
        self.t_total = t_total
        self.dt = dt

        # Along-track properties
        v_at, u_at, gsd_at, sl = self._calc_alongtrack_properties()
        self.v_alongtrack = v_at
        self.u_alongtrack = u_at
        self.gsd_alongtrack = gsd_at
        self.swath_length = sl

        # Altitude
        self.mean_altitude = self._calc_mean_altitude(assume_square_pixels)

        # Cross-track properties
        u_ct, sw, gsd_ct = self._calc_crosstrack_properties()
        self.u_crosstrack = u_ct
        self.swath_width = sw
        self.gsd_crosstrack = gsd_ct

        # Image origin (image transform offset)
        self.image_origin = self._calc_image_origin()

    def _calc_time_attributes(self):
        """Calculate time duration and sampling interval of IMU data"""
        t = np.array(self.imu_data["time"])
        dt = np.mean(np.diff(t))
        t_total = len(t) * dt
        return t_total, dt

    def _calc_alongtrack_properties(self):
        """Calculate along-track velocity, gsd, and swath length"""
        vx_alongtrack = (self.utm_x[-1] - self.utm_x[0]) / self.t_total
        vy_alongtrack = (self.utm_y[-1] - self.utm_y[0]) / self.t_total
        v_alongtrack = np.array((vx_alongtrack, vy_alongtrack))
        v_alongtrack_abs = np.linalg.norm(v_alongtrack)
        u_alongtrack = v_alongtrack / v_alongtrack_abs

        swath_length = self.t_total * v_alongtrack_abs
        gsd_alongtrack = self.dt * v_alongtrack_abs

        return v_alongtrack, u_alongtrack, gsd_alongtrack, swath_length

    def _calc_mean_altitude(self, assume_square_pixels):
        """Calculate mean altitude of uav during imaging

        Parameters
        ----------
        assume_square_pixels: bool
            If true, the across-track sampling distance is assumed to
            be equal to the alongtrack sampling distance. The altitude
            is calculated based on this and the number of cross-track samples.
            If false, the mean of the altitude values from the imu data
            is used. In both cases, the altitude offset is added.
        """
        if assume_square_pixels:
            swath_width = self.gsd_alongtrack * self.image_shape[1]
            altitude = swath_width / (2 * np.tan(self.camera_opening_angle / 2))
        else:
            altitude = np.mean(self.imu_data["altitude"])
        return altitude + self.altitude_offset

    def _calc_crosstrack_properties(self):
        """Calculate cross-track unit vector, swath width and sampling distance"""
        u_crosstrack = np.array(
            [-self.u_alongtrack[1], self.u_alongtrack[0]]
        )  # Rotate 90 CCW
        swath_width = 2 * self.mean_altitude * np.tan(self.camera_opening_angle / 2)
        gsd_crosstrack = swath_width / self.image_shape[1]
        return u_crosstrack, swath_width, gsd_crosstrack

    def _calc_image_origin(self):
        """Calculate location of image pixel (0,0) in georeferenced coordinates"""
        alongtrack_offset = (
            self.mean_altitude * np.tan(self.pitch_offset) * self.u_alongtrack
        )
        crosstrack_offset = (
            self.mean_altitude * np.tan(self.roll_offset) * self.u_crosstrack
        )
        # NOTE: Signs of cross-track elements in equation below are "flipped"
        # because UTM coordinate system is right-handed and image coordinate
        # system is left-handed. If the camera_origin is in the middle of the
        # top line of the image, u_crosstrack points away from the image
        # origin (line 0, sample 0).
        image_origin = (
            self.camera_origin
            - 0.5 * self.swath_width * self.u_crosstrack  # Edge of swath
            + crosstrack_offset
            - alongtrack_offset
        )
        return image_origin

    def get_image_transform(self, ordering: str = "alphabetical") -> tuple[float, ...]:
        """Get 6-element affine transform for image

        Parameters
        ----------
        ordering : str, {"alphabetical","worldfile"}, default "alphabetical"
            If 'alphabetical', return A,B,C,D,E,F
            If 'worldfile', return A,D,B,E,C,F
            See https://en.wikipedia.org/wiki/World_file

        Returns
        -------
        transform: tuple[float]
            6-element affine transform

        Raises
        ------
        ValueError
            If invalid ordering parameter is used
        """
        A, D = self.gsd_crosstrack * self.u_crosstrack
        B, E = self.gsd_alongtrack * self.u_alongtrack
        C, F = self.image_origin

        if ordering == "alphabetical":
            return A, B, C, D, E, F
        elif ordering == "worldfile":
            return A, D, B, E, C, F
        else:
            error_msg = f"Invalid ordering argument {ordering}"
            logger.error(error_msg)
            raise ValueError(error_msg)


class SimpleGeoreferencer:
    def georeference_hyspec_save_geotiff(
        self,
        image_path: Union[Path, str],
        imudata_path: Union[Path, str],
        geotiff_path: Union[Path, str],
        rgb_only: bool = True,
        nodata_value: int = -9999,
        **kwargs,
    ):
        """Georeference hyperspectral image and save as GeoTIFF

        Parameters
        ----------
        image_path:
            Path to hyperspectral image header.
        imudata_path:
            Path to JSON file containing IMU data.
        geotiff_path:
            Path to (output) GeoTIFF file.
        rgb_only: bool
            Whether to only output an RGB version of the hyperspectral image.
            If false, the entire hyperspectral image is used. Note that
            this typically creates very large files that some programs
            (e.g. QGIS) can struggle to read.
        nodata_value:
            Value to insert in place of invalid pixels.
            Pixels which contain "all zeros" are considered invalid.
        """
        image, wl, _ = mpu.read_envi(image_path)
        if rgb_only:
            image, wl = mpu.rgb_subset_from_hsi(image, wl)
        self._insert_image_nodata_value(image, nodata_value)
        geotiff_profile = self.create_geotiff_profile(
            image, imudata_path, nodata_value=nodata_value, **kwargs
        )

        self.write_geotiff(geotiff_path, image, wl, geotiff_profile)

    # @staticmethod
    # def _move_bands_axis_first(image):
    #     """Move spectral bands axis from position 2 to 0"""
    #     return np.moveaxis(image, 2, 0)

    @staticmethod
    def _insert_image_nodata_value(image, nodata_value):
        """Insert nodata values in image (in-place)

        Parameters
        ----------
        image:
            3D image array ordered as (lines, samples, bands)
            Pixels where every band value is equal to zero
            are interpreted as invalid (no data).
        nodata_value:
            Value to insert in place of invalid data.
        """
        nodata_mask = np.all(image == 0, axis=2)
        image[nodata_mask] = nodata_value

    @staticmethod
    def create_geotiff_profile(
        image: NDArray,
        imudata_path: Union[Path, str],
        nodata_value: int = -9999,
        **kwargs,
    ) -> dict:
        """Create profile for writing image as geotiff using rasterio

        Parameters
        ----------
        image:
            3D image array ordered, shape (n_lines,n_samples,n_bands).
        imudata_path:
            Path to JSON file containing IMU data for image
        nodata_value: int, default -9999
            Nodata value to insert for invalid pixels

        """
        imu_data = ImuDataParser.read_imu_json_file(imudata_path)
        image_flight_meta = ImageFlightMetadata(imu_data, image.shape[:], **kwargs)
        transform = Affine(*image_flight_meta.get_image_transform())
        crs_epsg = image_flight_meta.utm_epsg

        profile = DefaultGTiffProfile()
        profile.update(
            height=image.shape[0],
            width=image.shape[1],
            count=image.shape[2],
            dtype=str(image.dtype),
            crs=CRS.from_epsg(crs_epsg),
            transform=transform,
            nodata=nodata_value,
        )

        return profile  # type: ignore

    def write_geotiff(
        self,
        geotiff_path: Union[Path, str],
        image: NDArray,
        wavelengths: NDArray,
        geotiff_profile: dict,
    ):
        """Write image as GeoTIFF

        Parameters
        ----------
        geotiff_path : Union[Path, str]
            Path to (output) GeoTIFF file
        image : NDArray
            Image to write, shape (n_lines, n_samples, n_bands)
        wavelengths : NDArray
            Wavelengths (in nm) corresponding to each image band.
            The wavelengths are used to set the descption of each band
            in the GeoTIFF file.
        geotiff_profile : dict
            Dict with GeoTIFF parameters ("profile")

        Notes
        -----
        Rasterio / GDAL requires the image to be ordered "bands first",
        e.g. shape (bands, lines, samples). However, the default used by e.g.
        the 'spectral' library is (lines, samples, bands), and this convention
        should be used consistenly to avoid bugs. This function moves the band
        axis directly before writing.
        """
        image = reshape_as_raster(image)  # Band ordering required by GeoTIFF
        band_names = [f"{wl:.3f}" for wl in wavelengths]
        with rasterio.Env():
            with rasterio.open(geotiff_path, "w", **geotiff_profile) as dataset:
                if band_names is not None:
                    for i in range(dataset.count):
                        dataset.set_band_description(i + 1, band_names[i])
                dataset.write(image)

    @staticmethod
    def update_image_file_transform(
        geotiff_path: Union[Path, str], imu_data_path: Union[Path, str], **kwargs
    ):
        """Update the affine transform of an image

        Parameters
        ----------
        geotiff_path:
            Path to existing GeoTIFF file.
        imu_data_path:
            Path to JSON file with IMU data.
        **kwargs:
            Keyword arguments are passed along to create an ImageFlightMetadata object.
            Options include e.g. 'altitude_offset'. This can be useful in case
            the shape of the existing GeoTIFF indicates that some adjustments
            should be made to the image transform (which can be re-generated using
            an ImageFlightMetadata object).

        References:
        -----------
        - https://rasterio.readthedocs.io/en/latest/api/rasterio.rio.edit_info.html
        """
        imu_data = ImuDataParser.read_imu_json_file(imu_data_path)
        with rasterio.open(geotiff_path, "r") as dataset:
            im_width = dataset.width
            im_height = dataset.height
        image_flight_meta = ImageFlightMetadata(
            imu_data, image_shape=(im_height, im_width), **kwargs
        )
        new_transform = image_flight_meta.get_image_transform()
        rio_cmd = [
            "rio",
            "edit-info",
            "--transform",
            str(list(new_transform)),
            str(geotiff_path),
        ]
        subprocess.run(rio_cmd)
