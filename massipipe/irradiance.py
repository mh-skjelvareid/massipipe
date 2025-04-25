"""
Irradiance conversion and wavelength calibration for hyperspectral images.

This module contains classes and methods for converting raw spectra to irradiance
and for calibrating wavelengths using known Fraunhofer absorption lines.

Classes:
--------
- IrradianceConverter: Convert raw spectra to irradiance.
- WavelengthCalibrator: Calibrate wavelengths using Fraunhofer absorption lines.

"""

# Imports
import logging
import zipfile
from pathlib import Path
from typing import Iterable, Union

import numpy as np
from numpy.polynomial import Polynomial
from numpy.typing import NDArray
from scipy.signal import find_peaks

import massipipe.utils as mpu

# Get logger
logger = logging.getLogger(__name__)


class IrradianceConverter:
    """
    A class to convert raw spectra to irradiance using calibration data.

    Methods
    -------
    convert_raw_spectrum_to_irradiance(raw_spec, raw_metadata,
        set_irradiance_outside_wl_limits_to_zero=True, keep_original_dimensions=True)
        Convert raw spectrum to irradiance.
    convert_raw_file_to_irradiance(raw_spec_path, irrad_spec_path)
        Read raw spectrum, convert to irradiance, and save.
    """

    def __init__(
        self,
        irrad_cal_file: Union[Path, str],
        irrad_cal_dir_name: str = "downwelling_calibration_spectra",
        wl_min: Union[float, None] = None,
        wl_max: Union[float, None] = None,
    ):
        """Initialize irradiance converter.

        Parameters
        ----------
        irrad_cal_file : Union[Path, str]
            Path to downwelling irradiance calibration file.
        irrad_cal_dir_name : str, default "downwelling_calibration_spectra"
            Name of folder which calibration files will be unzipped into.
        wl_min : Union[float, None], optional
            Shortest valid wavelength (nm) in irradiance spectrum.
        wl_max : Union[float, None], optional
            Longest valid wavelength (nm) in irradiance spectrum.
        """
        # Save calibration file path
        self.irrad_cal_file = Path(irrad_cal_file)
        if not self.irrad_cal_file.exists():
            raise FileNotFoundError(f"Calibration file {self.irrad_cal_file} does not exist.")

        # Unzip calibration frames into subdirectory
        self.irrad_cal_dir = self.irrad_cal_file.parent / irrad_cal_dir_name
        self.irrad_cal_dir.mkdir(exist_ok=True)
        self._unzip_irrad_cal_file()

        # Load calibration data
        self._load_cal_dark_and_sensitivity_spectra()

        # Set valid wavelength range
        self.wl_min = self._irrad_wl[0] if wl_min is None else wl_min
        self.wl_max = self._irrad_wl[-1] if wl_max is None else wl_max
        self._valid_wl_ind = (self._irrad_wl >= self.wl_min) & (self._irrad_wl <= self.wl_max)

    def _unzip_irrad_cal_file(self, unzip_into_nonempty_dir: bool = False) -> None:
        """Unzip *.dcp file (which is a zip file).

        Parameters
        ----------
        unzip_into_nonempty_dir : bool, optional
            Whether to unzip into a non-empty directory.
        """
        if not unzip_into_nonempty_dir and any(list(self.irrad_cal_dir.iterdir())):
            logger.info(f"Non-empty downwelling calibration directory {self.irrad_cal_dir}")
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
            logger.error(f"File {self.irrad_cal_file} is not a valid ZIP file.", exc_info=True)
            raise
        except Exception as e:
            logger.error(
                f"Error while extracting downwelling calibration file {self.irrad_cal_file}",
                exc_info=True,
            )
            raise

    def _load_cal_dark_and_sensitivity_spectra(self) -> None:
        """Load dark current and irradiance sensitivity spectra from cal. files."""
        # Define paths
        cal_dark_path = self.irrad_cal_dir / "offset.spec.hdr"
        irrad_sens_path = self.irrad_cal_dir / "gain.spec.hdr"
        if not cal_dark_path.exists() or not irrad_sens_path.exists():
            raise FileNotFoundError("Calibration files not found in the specified directory.")

        # Read from files
        cal_dark_spec, cal_dark_wl, cal_dark_metadata = mpu.read_envi(cal_dark_path)
        irrad_sens_spec, irrad_sens_wl, irrad_sens_metadata = mpu.read_envi(irrad_sens_path)

        # Save attributes
        if not np.array_equal(cal_dark_wl, irrad_sens_wl):
            raise ValueError("Wavelengths in dark current and gain spectra do not match.")
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
        """Convert raw spectrum to irradiance.

        Parameters
        ----------
        raw_spec : NDArray
            Raw spectrum.
        raw_metadata : dict
            Dictionary with ENVI metadata for raw spectrum.
        set_irradiance_outside_wl_limits_to_zero : bool, optional
            Whether to set spectrum values outside wavelength limits to zero.
            If False, values outside valid range are treated the same as
            values inside valid range.
        keep_original_dimensions : bool, optional
            Whether to format output spectrum with same (singleton)
            dimensions as input spectrum. Can be useful for broadcasting.

        Returns
        -------
        NDArray
            Spectrum converted to spectral irradiance, unit W/(m2*nm).

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
        if raw_spec.ndim != 1:
            raise ValueError("Input raw_spec must be a 1D array.")

        # Scale conversion spectrum according to difference in shutter values
        raw_shutter = float(raw_metadata["shutter"])
        scaled_conv_spec = (self._irrad_sens_shutter / raw_shutter) * self._irrad_sens_spec

        # Subtract dark current, multiply with radiance conversion spectrum
        # NOTE: Resonon irradiance unit is uW/(pi*cm2*um) = 10e-5 W/(pi*m2*nm)
        # Scaling with pi is at least consistent with Resonon code.
        cal_irrad_spec = (raw_spec - self._cal_dark_spec) * scaled_conv_spec

        # Convert to standard spectral irradiance unit W/(m2*nm)
        cal_irrad_spec = mpu.irrad_uflicklike_to_si_nm(cal_irrad_spec * np.pi)

        # Set spectrum outside wavelength limits to zero
        if set_irradiance_outside_wl_limits_to_zero:
            cal_irrad_spec[~self._valid_wl_ind] = 0

        if keep_original_dimensions:
            cal_irrad_spec = np.reshape(cal_irrad_spec, original_input_dimensions)

        return cal_irrad_spec

    def convert_raw_file_to_irradiance(
        self, raw_spec_path: Union[Path, str], irrad_spec_path: Union[Path, str]
    ) -> None:
        """Read raw spectrum, convert to irradiance, and save.

        Parameters
        ----------
        raw_spec_path : Union[Path, str]
            Path to ENVI header file for raw spectrum.
        irrad_spec_path : Union[Path, str]
            Path to ENVI header file for saving irradiance spectrum.
        """
        try:
            raw_spec, _, spec_metadata = mpu.read_envi(raw_spec_path)
            irrad_spec = self.convert_raw_spectrum_to_irradiance(raw_spec, spec_metadata)
            spec_metadata["unit"] = "W/(m2*nm)"
            mpu.save_envi(irrad_spec_path, irrad_spec, spec_metadata)
        except Exception as e:
            logger.error(f"Error converting raw file to irradiance: {e}", exc_info=True)
            raise


class WavelengthCalibrator:
    """
    A class to calibrate wavelengths using Fraunhofer absorption lines.

    Methods
    -------
    fit(spec, wl_orig)
        Fit wavelength calibration polynomial.
    fit_batch(spectrum_header_paths)
        Calibrate wavelength based on spectrum with highest SNR (among many).
    update_header_wavelengths(header_path)
        Update header file with calibrated wavelengths.

    Notes
    -----
    The method used here is very similar to that described in the paper cited below.
    However, note that only a 2nd degree polynomial is used here.

    Mori et al. (2024):
    "Fraunhofer line-based wavelength-calibration method without calibration targets for
    planetary lander instruments", Planetary and Space Science , Vol. 240, 2024-01, p. 105835
    https://doi.org/10.1016/j.pss.2023.105835.
    """

    def __init__(self):
        """Initialize wavelength calibrator."""
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
        reference_wl: int = 550,
        distance: int = 20,
        width: int = 5,
        rel_prominence: float = 0.1,
    ) -> tuple[NDArray, dict[str, NDArray]]:
        """Detect absorption lines using local peak detection.

        Parameters
        ----------
        spec : NDArray
            Spectrum to analyze.
        wl : NDArray
            Wavelengths corresponding to the spectrum.
        reference_wl : int, optional
            Reference wavelength for prominence calculation.
        distance : int, optional
            Minimum distance between peaks.
        width : int, optional
            Minimum width of peaks.
        rel_prominence : float, optional
            Relative prominence of peaks.

        Returns
        -------
        tuple[NDArray, dict[str, NDArray]]
            Indices of detected peaks and properties of the peaks.
        """
        ref_wl_ind = mpu.closest_wl_index(wl, reference_wl)
        prominence = spec[ref_wl_ind] * rel_prominence
        peak_indices, peak_properties = find_peaks(
            -spec, distance=distance, width=width, prominence=prominence
        )
        return peak_indices, peak_properties

    @staticmethod
    def _fit_wavelength_polynomial(
        sample_indices: NDArray, wavelengths: NDArray, n_samples: int
    ) -> tuple[NDArray, NDArray]:
        """Fit 2nd degree polynomial to set of (sample, wavelength) pairs.

        Parameters
        ----------
        sample_indices : NDArray
            Indices of samples in a sampled spectrum, typically for spectral
            peaks / absorption lines with known wavelengths.
        wavelengths : NDArray
            Wavelengths (in physical units) corresponding to sample_indices.
        n_samples : int
            Total number of samples in sampled spectrum.

        Returns
        -------
        tuple[NDArray, NDArray]
            Calibrated wavelengths and polynomial coefficients.
        """
        polynomial_fitted = Polynomial.fit(sample_indices, wavelengths, deg=2, domain=[])
        wl_cal = polynomial_fitted(np.arange(n_samples))
        wl_poly_coeff = polynomial_fitted.coef
        return wl_cal, wl_poly_coeff

    def _filter_fraunhofer_lines(
        self,
        line_indices: NDArray,
        orig_wl: NDArray,
        win_width_nm: Union[int, float] = 20,
    ) -> tuple[NDArray, NDArray]:
        """Calibrate wavelength values from known Fraunhofer absorption lines.

        Parameters
        ----------
        line_indices : NDArray
            Indices of samples in spectrum where a (potential) absorption
            line has been detected. Indices must be within the range
            (0, len(orig_wl)).
        orig_wl : NDArray
            Original wavelength vector. Values are assumed to be "close enough"
            to be used to create search windows for each Fraunhofer line,
            typically within a few nm.
        win_width_nm : Union[int, float], optional
            The width of the search windows in units of nanometers.

        Returns
        -------
        tuple[NDArray, NDArray]
            Indices for absorption lines found close to Fraunhofer lines and
            corresponding Fraunhofer line wavelengths.
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
            peaks_in_window = line_indices[(line_indices >= win_low) & (line_indices <= win_high)]
            if len(peaks_in_window) == 1:
                filtered_line_indices.append(peaks_in_window[0])
                fraunhofer_wavelengths.append(fh_line_wl)

        return np.array(filtered_line_indices), np.array(fraunhofer_wavelengths)

    def fit(self, spec: NDArray, wl_orig: NDArray) -> None:
        """Fit wavelength calibration polynomial.

        Parameters
        ----------
        spec : NDArray
            Sampled radiance/irradiance spectrum, shape (n_samples,).
        wl_orig : NDArray
            Wavelengths corresponding to spectral samples, shape (n_samples).
            Wavelengths values are assumed to be close (within a few nm)
            to their true values.

        Raises
        ------
        ValueError
            Raised if less than 3 absorption lines found in data.
        """
        spec = np.squeeze(spec)
        if spec.ndim != 1:
            raise ValueError("Input spectrum must be a 1D array.")

        line_indices, _ = self._detect_absorption_lines(spec, wl_orig)
        fh_line_indices, fh_wavelengths = self._filter_fraunhofer_lines(line_indices, wl_orig)
        if len(fh_line_indices) < 3:
            raise ValueError("Too low data quality: Less than 3 absorption lines found.")
        wl_cal, wl_poly_coeff = self._fit_wavelength_polynomial(
            fh_line_indices, fh_wavelengths, len(wl_orig)
        )

        self._fh_line_indices = fh_line_indices
        self._fh_wavelengths = fh_wavelengths
        self._wl_poly_coeff = wl_poly_coeff
        self.wl_cal = wl_cal
        self.max_wl_diff = np.max(abs(wl_cal - wl_orig))

    def fit_batch(self, spectrum_header_paths: Iterable[Union[Path, str]]) -> None:
        """Calibrate wavelength based on spectrum with highest SNR (among many).

        Parameters
        ----------
        spectrum_header_paths : Iterable[Union[Path, str]]
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
                continue
            spectra.append(np.squeeze(spec))

        if not spectra:
            raise ValueError("No valid spectra found for calibration.")

        spectra = np.array(spectra)
        best_spec_ind = np.argmax(np.max(spectra, axis=1))
        cal_spec = spectra[best_spec_ind]
        self.reference_spectrum_path = str(spectrum_header_paths[best_spec_ind])

        self.fit(cal_spec, wl)

    def update_header_wavelengths(self, header_path: Union[Path, str]) -> None:
        """Update header file with calibrated wavelengths.

        Parameters
        ----------
        header_path : Union[Path, str]
            Path to ENVI header file.
        """
        if self.wl_cal is None:
            raise AttributeError("Attribute wl_cal is not set - fit (calibrate) first.")
        mpu.update_header_wavelengths(self.wl_cal, header_path)
