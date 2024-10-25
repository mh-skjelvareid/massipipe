# Imports
import logging
from pathlib import Path
from typing import Sequence, Union

import numpy as np
from numpy.typing import NDArray

import massipipe.utils as mpu

# Get logger
logger = logging.getLogger(__name__)


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

    def glint_correct_image(self, refl_image: NDArray, refl_wl: NDArray) -> NDArray:
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
        nir_ind = mpu.get_nir_ind(refl_wl)
        nir_im = np.mean(refl_image[:, :, nir_ind], axis=2, keepdims=True)
        refl_image_glint_corr = refl_image - nir_im

        if self.smooth_with_savitsky_golay:
            refl_image_glint_corr = mpu.savitzky_golay_filter(refl_image_glint_corr)

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
        glint_corr_image = self.glint_correct_image(image, wl, **kwargs)
        mpu.save_envi(glint_corr_image_path, glint_corr_image, metadata)


class HedleyGlintCorrector:
    def __init__(
        self,
        smooth_spectra: bool = True,
        subtract_dark_spec: bool = True,
        set_negative_values_to_zero=False,
        max_invalid_frac: float = 0.05,
    ):
        """Initialize glint corrector

        Parameters
        ----------
        smooth_with_savitsky_golay : bool, default True
            Whether to smooth glint corrected images using a Savitzky-Golay filter.
        subtract_dark_spec: bool
            Whether to subtract estimated minimum value in training data (for each
            wavelength) from glint corrected image. This has the effect of removing
            "background" spectrum caused by e.g. reflection of the sky and water column
            scattering.
        set_negative_values_to_zero: bool
            Glint is corrected by subtracting estimated glint from the original image.
            The subtraction process may result in some spectral bands getting negative
            values. If the fraction of pixels that is negative is larger than
            max_invalid_fraction, all pixel values are set to zero, indicating an
            invalid pixel.
        max_invalid_frac: float
            The fraction of spectral bands that is allowed to be invalid (i.e. zero)
            before the whole pixel is declared invalid and all bands are set to zero.
            Allowing some invalid bands may keep useful information, but a high number
            of invalid bands results in severe spectral distortion and indicates poor
            data quality.
        """
        self.smooth_spectra = smooth_spectra
        self.subtract_dark_spec = subtract_dark_spec
        self.set_negative_values_to_zero = set_negative_values_to_zero
        self.max_invalid_frac = max_invalid_frac
        self.b = None
        self.wl = None
        self.min_nir = None
        self.vis_ind = None
        self.nir_nid = None
        self.dark_spec = None

    def fit_to_reference_images(
        self,
        reference_image_paths: Sequence[Union[Path, str]],
        reference_image_ranges: Union[None, Sequence[Union[None, Sequence[int]]]] = None,
        sample_frac: float = 0.5,
    ) -> None:
        """Fit glint model based on spectra from reference images

        Parameters
        ----------
        reference_image_paths : list[Union[Path, str]]
            List of paths to reference hyperspectral images (header files)

        reference_image_ranges: Union[None, list[Union[None, list[int]]]]
            List of ranges for pixels to use as reference.
            If reference_image_ranges == None, all images are used in full.
            Pixels should preferrably be from a homogenous, deep water area
            with some visible sun/sky glint. Using ranges enables using part of
            an image fulfilling these requirements.
            Setting the range for a single image to None indicates that the
            whole image should be used.
            Ranges are specified as
            [line_start, line_end, sample_start, sample_end]
            A subset of the 3D image cube is extracted as
            image[line_start:line_end,sample_start:sample_end,:]
        sample_frac: float
            Fraction of total number of image pixels that is used for training.
            Value in range [0.0, 1.0]. Pixels are randomly sampled.
        """
        if reference_image_ranges is None:
            reference_image_ranges = [None for _ in range(len(reference_image_paths))]

        train_spec = []
        for ref_im_path, ref_im_range in zip(reference_image_paths, reference_image_ranges):
            ref_im, wl, _ = mpu.read_envi(Path(ref_im_path))
            if ref_im_range is not None:
                assert len(ref_im_range) == 4
                line_start, line_end, sample_start, sample_end = ref_im_range
                ref_im = ref_im[line_start:line_end, sample_start:sample_end, :]
            sampled_spectra = mpu.random_sample_image(ref_im, sample_frac=sample_frac)
            train_spec.append(sampled_spectra)
        train_spec = np.concat(train_spec)
        self.fit(train_spec, wl)

    def fit(self, train_spec: NDArray, wl: NDArray) -> None:
        """Fit glint model to training spectra

        Parameters
        ----------
        train_spec : NDArray
            Training spectra, shape (n_samples, n_bands)
        wl : NDArray
            Wavelength vector (nanometers)
        """
        self.wl = wl
        self.vis_ind = mpu.get_vis_ind(wl)
        self.nir_ind = mpu.get_nir_ind(wl)

        x = np.mean(train_spec[:, self.nir_ind], axis=1, keepdims=True)
        Y = train_spec[:, self.vis_ind]
        self.b = self.linear_regression_multiple_dependent_variables(x, Y)
        self.min_nir = np.percentile(x, q=1, axis=None)  # Using 1st percentile as min.

        self.dark_spec = np.percentile(train_spec, q=1, axis=0)

    @staticmethod
    def linear_regression_multiple_dependent_variables(x: NDArray, Y: NDArray) -> NDArray:
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

    def glint_correct_image(
        self,
        image: NDArray,
    ) -> NDArray:
        """Remove sun and sky glint from image using fit linear model

        Parameters
        ----------
        image: NDArray
            Hyperspectral image, shape (n_lines, n_samples, n_bands)

        Returns
        --------
        image_gc: NDArray
            Glint corrected image, only containing visible light spectra.

        Notes
        -----
        - The glint correction is based on the assumption that there is
        (approximately) no water-leaving radiance in the NIR spectral region.
        This is often the case, since NIR light is very effectively
        absorbed by water.
        """

        # Shape into 2D array, save original shape for later
        input_shape = image.shape
        image = image.reshape((-1, image.shape[-1]))  # 2D, shape (n_samples, n_bands)

        # Detect all-zero pixels (invalid)
        invalid_mask = ~np.all(image == 0, axis=1)

        # Extract VIS and NIR bands
        vis = image[:, self.vis_ind]
        nir = np.mean(image[:, self.nir_ind], axis=1, keepdims=True)

        # Offset NIR, taking into account "ambient" (minimum) NIR
        nir = nir - self.min_nir
        nir[nir < 0] = 0  # Positivity constraint

        # Estimate glint and subtract from visible spectrum
        glint = nir @ self.b
        vis = vis - glint

        if self.subtract_dark_spec:
            if self.dark_spec is not None:
                vis = vis - self.dark_spec[self.vis_ind]
            else:
                raise ValueError("Dark spectrum not calculated - run fit() first")

        # Set negative values to zero
        if self.set_negative_values_to_zero:
            vis[vis < 0] = 0

        # Set invalid pixels (too many zeros) to all-zeros
        if self.set_negative_values_to_zero:
            zeros_fraction = np.count_nonzero(vis == 0, axis=1) / vis.shape[1]
            invalid_mask = invalid_mask & (zeros_fraction > self.max_invalid_frac)
            vis[invalid_mask] = 0

        # Reshape to fit original dimensions
        output_shape = input_shape[:-1] + (vis.shape[-1],)
        vis = np.reshape(vis, output_shape)

        # Smooth spectra (optional)
        if self.smooth_spectra:
            vis = mpu.savitzky_golay_filter(vis)

        return vis

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
        # Glint correction
        image, wl, metadata = mpu.read_envi(image_path)
        glint_corr_image = self.glint_correct_image(image)

        # Limit wavelengths to only visible
        wl = wl[self.vis_ind]
        metadata["wavelength"] = mpu.array_to_header_string(wl)
        if "solar irradiance" in metadata:
            irrad_spec = mpu.header_string_to_array(metadata["solar irradiance"])
            irrad_spec = irrad_spec[self.vis_ind]
            metadata["solar irradiance"] = mpu.array_to_header_string(irrad_spec)

        # Save
        mpu.save_envi(glint_corr_image_path, glint_corr_image, metadata)
