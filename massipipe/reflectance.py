# Imports
import logging
import zipfile
from pathlib import Path
from typing import Iterable, Union

import numpy as np
from numpy.polynomial import Polynomial
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

import massipipe.utils as mpu

# Get logger
logger = logging.getLogger(__name__)


class ReflectanceConverter:
    """A class for converting images from Resonon Pika L cameras to reflectance"""

    def __init__(
        self,
        wl_min: Union[int, float, None] = 400,
        wl_max: Union[int, float, None] = 930,
        conv_irrad_with_gauss: Union[bool, None] = True,
        fwhm_irrad_smooth: Union[float, None] = 3.5,
        smooth_spectra: Union[bool, None] = False,
        add_map_info: Union[bool, None] = True,
        refl_from_mean_irrad: Union[bool, None] = False,
        irrad_spec_paths: Union[Iterable[Union[Path, str]], None] = None,
    ):
        """Initialize reflectance converter

        Parameters
        ----------
        wl_min : Union[int, float], default 400
            Minimum wavelength (nm) to include in reflectance image.
        wl_max : Union[int, float], default 930
            Maximum wavelength (nm) to include in reflectance image.
        conv_irrad_with_gauss: bool, default True
            Indicate if irradiance spectrum should be smoothed with Gaussian kernel.
            This may be useful if irradiance is measured with a higher spectral
            resolution than radiance, and thus has sharper "spikes".
        fwhm_irrad_smooth: float, default 3.5
            Full-width-half-maximum for Gaussian smoothing kernel, in nanometers.
            Only used if conv_irrad_with_gauss==True
        smooth_spectra: bool, default False
            Whether to smooth the reflectance spectra using a Savitzky-Golay filter
        refl_from_mean_irrad: Union[bool, True], default False
            If True, the mean irradiance for the whole dataset is used for
            calculating reflectance, rather than the irradiance for a single image.
            This can be useful if individual irradiance are compromised and
            using the mean is more "robust". Paths to the irradiance files
            (irrad_spec_paths) must be specified.
        irrad_spec_paths : Union[Iterable[Union[Path, str]], None], default None
            List of paths to irradiance spectra for the dataset.
            If specified (not None), a mean irradiance value is caluculated based
            on the spectra, and this irradiance value is used for every reflectance
            conversion, rather than the

        Notes
        ------
        The signal-to-noise ratio of both radiance images and irradiance
        spectra is generally lower at the low and high ends. When
        radiance is divided by noisy irradiance values close to zero, the
        noise can "blow up". Limiting the wavelength range can ensure
        that the reflectance images have more well-behaved values.
        """
        # Defaults
        self.wl_min = float(wl_min) if wl_min else 400.0
        self.wl_max = float(wl_max) if wl_max else 930.0

        self.conv_irrad_with_gauss = (
            True if conv_irrad_with_gauss or (conv_irrad_with_gauss is None) else False
        )
        self.fwhm_irrad_smooth = fwhm_irrad_smooth if fwhm_irrad_smooth else 3.5
        self.smooth_spectra = bool(smooth_spectra)
        self.add_map_info = True if add_map_info or (add_map_info is None) else False

        if refl_from_mean_irrad:
            self.refl_from_mean_irrad = True
            irrad_spec_mean, irrad_wl, _ = self._get_mean_irrad_spec(irrad_spec_paths)
        else:
            self.refl_from_mean_irrad = False
            irrad_spec_mean, irrad_wl = np.array([]), np.array([])
        self.ref_irrad_spec_mean = irrad_spec_mean
        self.ref_irrad_spec_wl = irrad_wl

    @staticmethod
    def _get_mean_irrad_spec(irrad_spec_paths):
        """Read irradiance spectra from file and calculate mean"""
        irrad_spectra = []
        for irrad_spec_path in irrad_spec_paths:
            if irrad_spec_path.exists():
                irrad_spec, irrad_wl, _ = mpu.read_envi(irrad_spec_path)
                irrad_spectra.append(irrad_spec.squeeze())
        if not irrad_spectra:
            raise FileNotFoundError(f"No valid irradiance spectra found.")
        irrad_spectra = np.array(irrad_spectra)
        irrad_spec_mean = np.mean(irrad_spectra, axis=0)
        return irrad_spec_mean, irrad_wl, irrad_spectra

    def conv_spec_with_gaussian(self, spec: NDArray, wl: NDArray) -> NDArray:
        """Convolve spectrum with Gaussian kernel to smooth / blur spectral details

        Parameters
        ----------
        spec: NDArray
            Input spectrum, shape (n_bands,)
        wl: NDArray, nanometers
            Wavelengths corresponding to each spectrum value, shape (n_bands,)

        Returns
        -------
        spec_filtered: NDArray
            Filtered / smoothed version of spec, with same dimensions

        Notes
        -----
        When the kernel extends outside the data while filtering, edges are handled
        by repeating the nearest sampled value (edge value).

        """
        sigma_wl = self.fwhm_irrad_smooth * 0.588705  # sigma = FWHM / 2*sqrt(2*ln(2))
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
        if self.conv_irrad_with_gauss:
            irrad_spec = self.conv_spec_with_gaussian(irrad_spec, irrad_wl)
        irrad_spec = self._interpolate_irrad_to_image_wl(irrad_spec, irrad_wl, rad_wl)
        irrad_spec = np.expand_dims(irrad_spec, axis=(0, 1))

        # Convert to reflectance, assuming Lambertian (perfectly diffuse) surface
        refl_image = np.pi * (rad_image.astype(np.float32) / irrad_spec.astype(np.float32))
        refl_wl = rad_wl

        # Spectral smoothing (optional)
        if self.smooth_spectra:
            refl_image = mpu.savitzky_golay_filter(refl_image)

        return refl_image, refl_wl, irrad_spec

    def convert_radiance_file_to_reflectance(
        self,
        radiance_image_header: Union[Path, str],
        irradiance_header: Union[Path, str],
        reflectance_image_header: Union[Path, str],
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
        if self.refl_from_mean_irrad:
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
