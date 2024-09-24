# Imports
import logging
import subprocess
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Union

import rasterio
import rasterio.merge
import yaml

from massipipe.georeferencing import GeoTransformer, ImuDataParser, SimpleGeoreferencer
from massipipe.glint import FlatSpecGlintCorrector, HedleyGlintCorrector
from massipipe.irradiance import IrradianceConverter, WavelengthCalibrator
from massipipe.mosaic import add_geotiff_overviews, mosaic_geotiffs
from massipipe.quicklook import QuickLookProcessor
from massipipe.radiance import RadianceConverter
from massipipe.reflectance import ReflectanceConverter

# import massipipe.processors as mpp

# Get logger
logger = logging.getLogger(__name__)


def parse_config(yaml_path):
    """Parse YAML config file, accepting only basic YAML tags"""
    with open(yaml_path, "r") as stream:
        config = yaml.safe_load(stream)
    return config


@dataclass
class ProcessedFilePaths:
    """Data class for simple management of lists of processed files"""

    quicklook: list[Path] = field(default_factory=list)
    radiance: list[Path] = field(default_factory=list)
    radiance_gc: list[Path] = field(default_factory=list)
    radiance_gc_rgb: list[Path] = field(default_factory=list)
    irradiance: list[Path] = field(default_factory=list)
    imudata: list[Path] = field(default_factory=list)
    geotransform: list[Path] = field(default_factory=list)
    reflectance: list[Path] = field(default_factory=list)
    reflectance_gc: list[Path] = field(default_factory=list)
    reflectance_gc_rgb: list[Path] = field(default_factory=list)


class PipelineProcessor:
    def __init__(
        self,
        dataset_dir: Union[Path, str],
        config_file_name: str = "config.seabee.yaml",
    ):
        """Create a pipeline for processing all data in a dataset

        Parameters
        ----------
        dataset_dir : Union[Path, str]
            Path to folder containing dataset. The name of the folder
            will be used as the "base name" for all processed files.
            The folder should contain at least two subfolders:
            - 0_raw: Contains all raw images as saved by Resonon airborne system.
            - calibration: Contains Resonon calibration files for
            camera (*.icp) and downwelling irradiance sensor (*.dcp)

        Raises
        ------
        FileNotFoundError
            No folder called "0_raw" found
        FileNotFoundError
            No folder called "calibration" found
        """
        self.dataset_dir = Path(dataset_dir)

        # Read config
        config_file_path = self.dataset_dir / config_file_name
        self.config_file_path = config_file_path
        try:
            self.config = parse_config(self.dataset_dir / config_file_name)
        except IOError as e:
            logger.error(f"Error parsing config file {config_file_path}")
            raise e

        # Define dataset folder structure
        self.dataset_base_name = self.dataset_dir.name
        self.raw_dir = self.dataset_dir / "0_raw"
        self.quicklook_dir = self.dataset_dir / "0b_quicklook"
        self.radiance_dir = self.dataset_dir / "1a_radiance"
        self.radiance_gc_dir = self.dataset_dir / "1b_radiance_gc"
        self.radiance_gc_rgb_dir = self.radiance_gc_dir / "rgb"
        self.reflectance_dir = self.dataset_dir / "2a_reflectance"
        self.reflectance_gc_dir = self.dataset_dir / "2b_reflectance_gc"
        self.reflectance_gc_rgb_dir = self.reflectance_gc_dir / "rgb"
        self.imudata_dir = self.dataset_dir / "imudata"
        self.geotransform_dir = self.dataset_dir / "geotransform"
        self.mosaic_dir = self.dataset_dir / "mosaics"
        self.calibration_dir = self.dataset_dir / "calibration"
        self.logs_dir = self.dataset_dir / "logs"

        if not self.raw_dir.exists():
            raise FileNotFoundError(f'Folder "0_raw" not found in {self.dataset_dir}')
        if not self.calibration_dir.exists():
            raise FileNotFoundError(
                f'Folder "calibration" not found in {self.dataset_dir}'
            )

        # Get calibration file paths
        self.radiance_calibration_file = self._get_radiance_calibration_path()
        self.irradiance_calibration_file = self._get_irradiance_calibration_path()

        # Search for raw files, sort and validate
        self.raw_image_paths = list(self.raw_dir.rglob("*.bil.hdr"))
        self.raw_image_paths = sorted(self.raw_image_paths, key=self._get_image_number)
        times_paths, lcf_paths = self._validate_raw_files()
        self.times_paths = times_paths
        self.lcf_paths = lcf_paths

        # Search for raw irradiance spectrum files (not always present)
        self.raw_spec_paths = self._get_raw_spectrum_paths()

        # Create "base" file names numbered from 0
        self.base_file_names = self._create_base_file_names()

        # Create lists of processed file paths
        proc_file_paths = self._create_processed_file_paths()
        self.ql_im_paths = proc_file_paths.quicklook
        self.rad_im_paths = proc_file_paths.radiance
        self.rad_gc_im_paths = proc_file_paths.radiance_gc
        self.rad_gc_rgb_im_paths = proc_file_paths.radiance_gc_rgb
        self.irrad_spec_paths = proc_file_paths.irradiance
        self.imu_data_paths = proc_file_paths.imudata
        self.geotransform_paths = proc_file_paths.geotransform
        self.refl_im_paths = proc_file_paths.reflectance
        self.refl_gc_im_paths = proc_file_paths.reflectance_gc
        self.refl_gc_rgb_paths = proc_file_paths.reflectance_gc_rgb

        # Create mosaic file path
        self.mosaic_path = self.mosaic_dir / (self.dataset_base_name + "_rgb.tiff")

        # Configure logging
        self._configure_file_logging()

    def _configure_file_logging(self):
        """Configure logging for pipeline"""

        # Create log file path
        self.logs_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_file_name = f"{timestamp}_{self.dataset_base_name}.log"
        log_path = self.logs_dir / log_file_name

        # Add file handler to module-level logger
        file_handler = logging.FileHandler(log_path)
        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.info("File logging initialized.")

    def _validate_raw_files(self):
        """Check that all expected raw files exist

        Returns:
        --------
        times_paths, lcf_paths: list[Path]
            Lists of paths to *.times and *.lcf files for every valid raw file
        """
        times_paths = []
        lcf_paths = []
        for raw_image_path in list(self.raw_image_paths):  # Use list() to copy
            file_base_name = raw_image_path.name.split(".")[0]
            binary_im_path = raw_image_path.parent / (file_base_name + ".bil")
            times_path = raw_image_path.parent / (file_base_name + ".bil.times")
            lcf_path = raw_image_path.parent / (file_base_name + ".lcf")
            if (
                not (binary_im_path.exists())
                or not (times_path.exists())
                or not (lcf_path.exists())
            ):
                warnings.warn(
                    f"Set of raw files for image {raw_image_path} is incomplete."
                )
                self.raw_image_paths.remove(raw_image_path)
            else:
                times_paths.append(times_path)
                lcf_paths.append(lcf_path)
        return times_paths, lcf_paths

    @staticmethod
    def _get_image_number(raw_image_path: Union[Path, str]) -> int:
        """Get image number from raw image

        Parameters
        ----------
        raw_image_path : Union[Path, str]
            Path to raw image (ENVI header file)

        Returns
        -------
        int
            Image number

        Examples
        --------
        >>> PipelineProcessor._get_image_number("ExampleLocation_Pika_L_5.bil.hdr")
        5

        Notes
        -----
        Raw files are numbered sequentially, but the numbers are not
        zero-padded. This can lead to incorrect sorting of the images, e.g.
        ['im_1','im_2','im_10'] (simplified names for example) are sorted
        ['im_1','im_10','im_2']. By extracting the numbers from filenames
        of raw files and sorting explicitly on these (as integers),
        correct ordering can be achieved.
        """
        raw_image_path = Path(raw_image_path)
        image_file_stem = raw_image_path.name.split(".")[0]
        image_number = image_file_stem.split("_")[-1]
        return int(image_number)

    def _create_base_file_names(self):
        """Create numbered base names for processed files"""
        base_file_names = [
            f"{self.dataset_base_name}_{i:03d}"
            for i in range(len(self.raw_image_paths))
        ]
        return base_file_names

    def _create_processed_file_paths(self):
        """Define default subfolders for processed files"""
        proc_file_paths = ProcessedFilePaths()

        for base_file_name in self.base_file_names:
            # Quicklook
            ql_path = self.quicklook_dir / (base_file_name + "_quicklook.png")
            proc_file_paths.quicklook.append(ql_path)

            # Radiance
            rad_path = self.radiance_dir / (base_file_name + "_radiance.bip.hdr")
            proc_file_paths.radiance.append(rad_path)

            # Radiance, glint corrected
            rad_gc_path = self.radiance_gc_dir / (
                base_file_name + "_radiance_gc.bip.hdr"
            )
            proc_file_paths.radiance_gc.append(rad_gc_path)

            # Radiance, glint corrected, RGB version
            rad_gc_rgb_path = self.radiance_gc_rgb_dir / (
                base_file_name + "_radiance_gc_rgb.tiff"
            )
            proc_file_paths.radiance_gc_rgb.append(rad_gc_rgb_path)

            # Irradiance
            irs_path = self.radiance_dir / (base_file_name + "_irradiance.spec.hdr")
            proc_file_paths.irradiance.append(irs_path)

            # IMU data
            imu_path = self.imudata_dir / (base_file_name + "_imudata.json")
            proc_file_paths.imudata.append(imu_path)

            # Geotransform
            gt_path = self.geotransform_dir / (base_file_name + "_geotransform.json")
            proc_file_paths.geotransform.append(gt_path)

            # Reflectance
            refl_path = self.reflectance_dir / (base_file_name + "_reflectance.bip.hdr")
            proc_file_paths.reflectance.append(refl_path)

            # Reflectance, glint corrected
            rgc_path = self.reflectance_gc_dir / (
                base_file_name + "_reflectance_gc.bip.hdr"
            )
            proc_file_paths.reflectance_gc.append(rgc_path)

            # Reflectance, glint corrected, RGB version
            rgc_rgb_path = self.reflectance_gc_rgb_dir / (
                base_file_name + "_reflectance_gc_rgb.tiff"
            )
            proc_file_paths.reflectance_gc_rgb.append(rgc_rgb_path)

        return proc_file_paths

    def _get_raw_spectrum_paths(self):
        """Search for raw files matching Resonon default naming"""
        spec_paths = []
        for raw_image_path in self.raw_image_paths:
            spec_base_name = raw_image_path.name.split("_Pika_L")[0]
            image_number = self._get_image_number(raw_image_path)
            spec_binary = (
                raw_image_path.parent
                / f"{spec_base_name}_downwelling_{image_number}_pre.spec"
            )
            spec_header = raw_image_path.parent / (spec_binary.name + ".hdr")
            if spec_binary.exists() and spec_header.exists():
                spec_paths.append(spec_header)
            else:
                spec_paths.append(None)
        return spec_paths

    def _get_radiance_calibration_path(self):
        """Search for radiance calibration file (*.icp)"""
        icp_files = list(self.calibration_dir.glob("*.icp"))
        if len(icp_files) == 1:
            return icp_files[0]
        elif len(icp_files) == 0:
            raise FileNotFoundError(
                f"No radiance calibration file (*.icp) found in {self.calibration_dir}"
            )
        else:
            raise ValueError(
                "More than one radiance calibration file (*.icp) found in "
                + f"{self.calibration_dir}"
            )

    def _get_irradiance_calibration_path(self):
        """Search for irradiance calibration file (*.dcp)"""
        dcp_files = list(self.calibration_dir.glob("*.dcp"))
        if len(dcp_files) == 1:
            return dcp_files[0]
        elif len(dcp_files) == 0:
            raise FileNotFoundError(
                "No irradiance calibration file (*.dcp) found in "
                + f"{self.calibration_dir}"
            )
        else:
            raise ValueError(
                "More than one irradiance calibration file (*.dcp) found in "
                + f"{self.calibration_dir}"
            )

    def create_quicklook_images(self, overwrite=False, **kwargs):
        """Create quicklook versions of raw images"""
        logger.info("---- QUICKLOOK IMAGE GENERATION ----")
        self.quicklook_dir.mkdir(exist_ok=True)
        quicklook_processor = QuickLookProcessor()
        for raw_image_path, quicklook_image_path in zip(
            self.raw_image_paths, self.ql_im_paths
        ):
            if quicklook_image_path.exists() and not overwrite:
                logger.info(f"Image {quicklook_image_path.name} exists - skipping.")
                continue
            logger.info(f"Creating quicklook version of {raw_image_path.name}")
            try:
                quicklook_processor.create_quicklook_image_file(
                    raw_image_path, quicklook_image_path
                )
            except Exception as e:
                logger.warning(
                    f"Error occured while processing {raw_image_path}", exc_info=True
                )
                logger.warning("Skipping file")

    def convert_raw_images_to_radiance(self, overwrite=False, **kwargs):
        """Convert raw hyperspectral images (DN) to radiance (microflicks)"""
        logger.info("---- RADIANCE CONVERSION ----")
        self.radiance_dir.mkdir(exist_ok=True)
        radiance_converter = RadianceConverter(self.radiance_calibration_file)
        for raw_image_path, radiance_image_path in zip(
            self.raw_image_paths, self.rad_im_paths
        ):
            if radiance_image_path.exists() and not overwrite:
                logger.info(f"Image {radiance_image_path.name} exists - skipping.")
                continue
            logger.info(f"Converting {raw_image_path.name} to radiance")
            try:
                radiance_converter.convert_raw_file_to_radiance(
                    raw_image_path, radiance_image_path
                )
            except Exception as e:
                logger.warning(
                    f"Error occured while processing {raw_image_path}", exc_info=True
                )
                logger.warning("Skipping file")

    def convert_raw_spectra_to_irradiance(self, **kwargs):
        """Convert raw spectra (DN) to irradiance (W/(m2*nm))"""
        logger.info("---- IRRADIANCE CONVERSION ----")
        self.radiance_dir.mkdir(exist_ok=True)
        irradiance_converter = IrradianceConverter(self.irradiance_calibration_file)
        for raw_spec_path, irrad_spec_path in zip(
            self.raw_spec_paths, self.irrad_spec_paths
        ):
            if raw_spec_path is not None:
                logger.info(
                    f"Converting {raw_spec_path.name} to downwelling irradiance"
                )
                try:
                    irradiance_converter.convert_raw_file_to_irradiance(
                        raw_spec_path, irrad_spec_path
                    )
                except Exception:
                    logger.error(
                        f"Error occured while processing {raw_spec_path}", exc_info=True
                    )
                    logger.error("Skipping file")

    def calibrate_irradiance_wavelengths(self, **kwargs):
        """Calibrate irradiance wavelengths using Fraunhofer absorption lines"""
        logger.info("---- IRRADIANCE WAVELENGTH CALIBRATION ----")
        if not (self.radiance_dir.exists()):
            raise FileNotFoundError(
                "Radiance folder with irradiance spectra does not exist"
            )
        wavelength_calibrator = WavelengthCalibrator()
        irradiance_spec_paths = list(self.radiance_dir.glob("*.spec.hdr"))
        if irradiance_spec_paths:
            wavelength_calibrator.fit_batch(irradiance_spec_paths)
            for irradiance_spec_path in irradiance_spec_paths:
                logger.info(f"Calibrating wavelengths for {irradiance_spec_path.name}")
                try:
                    wavelength_calibrator.update_header_wavelengths(
                        irradiance_spec_path
                    )
                except Exception:
                    logger.error(
                        f"Error occured while processing {irradiance_spec_path}",
                        exc_info=True,
                    )
                    logger.error("Skipping file")

    def parse_and_save_imu_data(self, **kwargs):
        """Parse *.lcf and *.times files with IMU data and save as JSON"""
        logger.info("---- IMU DATA PROCESSING ----")
        self.imudata_dir.mkdir(exist_ok=True)
        imu_data_parser = ImuDataParser()
        for lcf_path, times_path, imu_data_path in zip(
            self.lcf_paths, self.times_paths, self.imu_data_paths
        ):
            logger.info(f"Processing IMU data from {lcf_path.name}")
            try:
                imu_data_parser.read_and_save_imu_data(
                    lcf_path, times_path, imu_data_path
                )
            except Exception:
                logger.error(
                    f"Error occured while processing {lcf_path}", exc_info=True
                )
                logger.error("Skipping file")

    def create_and_save_geotransform(self):
        logger.info("---- GEOTRANSFORM CALCULATION ----")
        self.geotransform_dir.mkdir(exist_ok=True)
        for raw_im_path, imu_data_path, geotrans_path in zip(
            self.raw_image_paths, self.imu_data_paths, self.geotransform_paths
        ):
            logger.info(f"Creating and saving geotransform for {raw_im_path.name}")
            try:
                if imu_data_path.exists() and raw_im_path.exists():
                    geotransformer = GeoTransformer(imu_data_path, raw_im_path)
                    geotransformer.save_image_geotransform(geotrans_path)
            except Exception:
                logger.error(
                    f"Error occured while processing {raw_im_path}", exc_info=True
                )
                logger.error("Skipping file")

    def glint_correct_radiance_images(self, overwrite=False):
        """Remove water surface reflections of sun and sky light"""
        logger.info("---- RADIANCE GLINT CORRECTION ----")
        self.reflectance_dir.mkdir(exist_ok=True)

        # Read glint correction reference information from config
        try:
            ref_im_nums = (self.config["massipipe_options"]
                          ["glint_correction"]["reference_image_numbers"])  # fmt: skip
            ref_im_ranges = (self.config["massipipe_options"]
                            ["glint_correction"]["reference_image_ranges"])  # fmt: skip
        except KeyError as ke:
            logger.error(
                "Missing glint correction reference image numbers / ranges "
                + f"in YAML config file {self.config_file_path}."
            )
            raise
        assert len(ref_im_nums) == len(ref_im_ranges)
        ref_im_paths = [self.rad_im_paths[im_num] for im_num in ref_im_nums]

        # Fit glint corrector
        glint_corrector = HedleyGlintCorrector()
        glint_corrector.fit_to_reference_images(ref_im_paths, ref_im_ranges)

    def convert_radiance_images_to_reflectance(self, overwrite=False, **kwargs):
        """Convert radiance images (microflicks) to reflectance (unitless)"""
        logger.info("---- REFLECTANCE CONVERSION ----")
        self.reflectance_dir.mkdir(exist_ok=True)
        reflectance_converter = ReflectanceConverter(
            irrad_spec_paths=self.irrad_spec_paths
        )

        if all([not rp.exists() for rp in self.rad_im_paths]):
            raise FileNotFoundError(f"No radiance images found in {self.radiance_dir}")
        if all([not irp.exists() for irp in self.irrad_spec_paths]):
            raise FileNotFoundError(
                f"No irradiance spectra found in {self.radiance_dir}"
            )

        for rad_path, irrad_path, refl_path in zip(
            self.rad_im_paths, self.irrad_spec_paths, self.refl_im_paths
        ):
            if refl_path.exists() and not overwrite:
                logger.info(f"Image {refl_path.name} exists - skipping.")
                continue
            if rad_path.exists() and irrad_path.exists():
                logger.info(f"Converting {rad_path.name} to reflectance.")
                try:
                    reflectance_converter.convert_radiance_file_to_reflectance(
                        rad_path, irrad_path, refl_path, **kwargs
                    )
                except Exception as e:
                    logger.error(
                        f"Error occured while processing {rad_path}", exc_info=True
                    )
                    logger.error("Skipping file")

    def glint_correct_reflectance_images(self, overwrite=False, **kwargs):
        """Correct for sun and sky glint in reflectance images"""
        logger.info("---- REFLECTANCE GLINT CORRECTION ----")
        self.reflectance_gc_dir.mkdir(exist_ok=True)
        glint_corrector = FlatSpecGlintCorrector()

        if all([not rp.exists() for rp in self.refl_im_paths]):
            raise FileNotFoundError(
                f"No reflectance images found in {self.reflectance_dir}"
            )

        for refl_path, refl_gc_path in zip(self.refl_im_paths, self.refl_gc_im_paths):
            if refl_gc_path.exists() and not overwrite:
                logger.info(f"Image {refl_gc_path.name} exists - skipping.")
                continue
            if refl_path.exists():
                logger.info(f"Applying glint correction to {refl_path.name}.")
                try:
                    glint_corrector.glint_correct_image_file(refl_path, refl_gc_path)
                except Exception as e:
                    logger.error(
                        f"Error occured while glint correcting {refl_path}",
                        exc_info=True,
                    )
                    logger.error("Skipping file")

    def georeference_glint_corrected_reflectance(self, **kwargs):
        """Create georeferenced GeoTIFF versions of glint corrected reflectance"""
        logger.info("---- GEOREFERENCING GLINT CORRECTED REFLECTANCE ----")
        self.reflectance_gc_rgb_dir.mkdir(exist_ok=True)
        georeferencer = SimpleGeoreferencer()

        if all([not rp.exists() for rp in self.refl_gc_im_paths]):
            warnings.warn(f"No reflectance images found in {self.reflectance_gc_dir}")

        for refl_gc_path, geotrans_path, geotiff_path in zip(
            self.refl_gc_im_paths, self.geotransform_paths, self.refl_gc_rgb_paths
        ):
            if refl_gc_path.exists() and geotrans_path.exists():
                logger.info(
                    f"Georeferencing and exporting RGB version of {refl_gc_path.name}."
                )
                try:
                    georeferencer.georeference_hyspec_save_geotiff(
                        refl_gc_path,
                        geotrans_path,
                        geotiff_path,
                        rgb_only=True,
                        **kwargs,
                    )
                except Exception:
                    logger.error(
                        "Error occured while georeferencing RGB version of "
                        f"{refl_gc_path}",
                        exc_info=True,
                    )
                    logger.error("Skipping file")

    def mosaic_geotiffs(self, overview_factors=(2, 4, 6, 8, 16)):
        """Merge non-rotated geotiffs into mosaic with overviews (rasterio)"""
        logger.info(f"Mosaicing GeoTIFFs in {self.reflectance_gc_rgb_dir}")
        self.mosaic_dir.mkdir(exist_ok=True)

        # Open images
        images_to_merge = []
        for image_path in self.refl_gc_rgb_paths:
            try:
                if image_path.exists():
                    images_to_merge.append(rasterio.open(image_path, "r"))
            except IOError as e:
                logger.warning(f"Error while reading {image_path} - skipping. ")

        # Merge
        try:
            rasterio.merge.merge(images_to_merge, dst_path=self.mosaic_path)
        except Exception:
            logger.error("Error while mosaicing geotiffs", exc_info=True)
            raise
        finally:
            for opened_image in images_to_merge:
                opened_image.close()

        # Add overviews
        logger.info(f"Adding overviews to mosaic {self.mosaic_path.name}")
        try:
            with rasterio.open(self.mosaic_path, "r+") as mosaic_dataset:
                mosaic_dataset.build_overviews(overview_factors)
        except Exception:
            logger.error("Error while adding overviews to mosaic", exc_info=True)
            raise

    def mosaic_geotiffs_gdal_cli(self):
        """Merge rotated geotiffs into single mosaic with overviews (GDAL CLI)"""
        logger.info(f"Mosaicing GeoTIFFs in {self.reflectance_gc_rgb_dir}")
        self.mosaic_dir.mkdir(exist_ok=True)

        # Explanation of gdalwarp options used:
        # -overwrite: Overwrite existing files without error / warning
        # -q: Suppress GDAL output (quiet)
        # -r near: Resampling method: Nearest neighbor
        # -of GTiff: Output format: GeoTiff

        # Run as subprocess without invoking shell. Note input file unpacking.
        gdalwarp_args = [
            "gdalwarp",
            "-overwrite",
            "-q",
            "-r",
            "near",
            "-of",
            "GTiff",
            *[str(p) for p in self.refl_gc_rgb_paths if p.exists()],
            str(self.mosaic_path),
        ]
        subprocess.run(gdalwarp_args)

        # Add image pyramids to file
        logger.info(f"Adding image pyramids to mosaic {self.mosaic_path.name}")
        # Explanation of gdaladdo options used:
        # -r average: Use averaging when resampling to lower spatial resolution
        # -q: Suppress output (be quiet)
        gdaladdo_args = ["gdaladdo", "-q", "-r", "average", str(self.mosaic_path)]
        subprocess.run(gdaladdo_args)

    # def update_geotiff_transforms(self, **kwargs):
    #     """Batch update GeoTIFF transforms

    #     Image affine transforms are re-calculated based on IMU data and
    #     (optional) keyword arguments.

    #     Keyword arguments:
    #     ------------------
    #     **kwargs:
    #         keyword arguments accepted by ImageFlightSegment, e.g.
    #         "altitude_offset".

    #     """
    #     logger.info("---- UPDATING GEOTIFF AFFINE TRANSFORMS ----")
    #     georeferencer = SimpleGeoreferencer()

    #     if all([not gtp.exists() for gtp in self.refl_gc_rgb_paths]):
    #         warnings.warn(f"No GeoTIFF images found in {self.reflectance_gc_rgb_dir}")

    #     for imu_data_path, geotiff_path in zip(
    #         self.imu_data_paths, self.refl_gc_rgb_paths
    #     ):
    #         if imu_data_path.exists() and geotiff_path.exists():
    #             logger.info(f"Updating transform for {geotiff_path.name}.")
    #             try:
    #                 georeferencer.update_image_file_transform(
    #                     geotiff_path,
    #                     imu_data_path,
    #                     **kwargs,
    #                 )
    #             except Exception:
    #                 logger.error(
    #                     f"Error occured while updating transform for {geotiff_path}",
    #                     exc_info=True,
    #                 )
    #                 logger.error("Skipping file")

    def run(
        self,
        create_quicklook_images=True,
        convert_raw_images_to_radiance=True,
        convert_raw_spectra_to_irradiance=True,
        calibrate_irradiance_wavelengths=True,
        parse_imu_data=True,
        create_geotransform_json=True,
        convert_radiance_to_reflectance=True,
        glint_correct_reflectance=True,
        geotiff_from_glint_corrected_reflectance=True,
        mosaic_geotiffs=True,
        **kwargs,
    ):
        """_summary_

        Parameters
        ----------
        create_quicklook_images : bool, default True
            Whether to create "quicklook" images based on raw images
        convert_raw_images_to_radiance : bool, default True
            Whether to convert raw images to radiance
        convert_raw_spectra_to_irradiance : bool, default True
            Whether to convert raw spectra to irradiance.
            Raw spectra ("downwelling") must exist in the 0_raw folder
        calibrate_irradiance_wavelengths : bool, default True
            Whether to calibrate irrdaiance wavelengths using
            known Fraunhofer absorption lines.
        parse_imu_data : bool, default True
            Wheter to read IMU data from *.lcf files,
            interpolate it to match image line timestamps in
            *.times files, and save the results in JSON file.
        convert_radiance_to_reflectance : bool, default True
            Whether to convert radiance iamges to reflectance images.
            Radiance images must exist in folder 1_radiance.
        glint_correct_reflectance : bool, default True
            Whether to apply glint correction to reflectance images.
            Reflectance images must exist in folder 2_reflectance
        geotiff_from_glint_corrected_reflectance : bool, default True
            Whether to create georeferenced GeoTIFF images from
            glint corrected reflectance images.
        mosaic_geotiffs : bool, default True
            Whether to combine all GeoTIFFs in a single "mosaic" GeoTIFF
        """
        if create_quicklook_images:
            try:
                self.create_quicklook_images(**kwargs)
            except Exception:
                logger.error("Error while creating quicklook images", exc_info=True)
        if convert_raw_images_to_radiance:
            try:
                self.convert_raw_images_to_radiance(**kwargs)
            except Exception:
                logger.error(
                    "Error while converting raw images to radiance", exc_info=True
                )
        if convert_raw_spectra_to_irradiance:
            try:
                self.convert_raw_spectra_to_irradiance(**kwargs)
            except Exception:
                logger.error(
                    "Error while converting raw spectra to irradiance", exc_info=True
                )
        if calibrate_irradiance_wavelengths:
            try:
                self.calibrate_irradiance_wavelengths(**kwargs)
            except Exception:
                logger.error(
                    "Error while calibrating irradiance wavelengths", exc_info=True
                )

        if parse_imu_data:
            try:
                self.parse_and_save_imu_data(**kwargs)
            except Exception:
                logger.error("Error while parsing and saving IMU data", exc_info=True)

        if create_geotransform_json:
            try:
                self.create_and_save_geotransform()
            except Exception:
                logger.error("Error while parsing and saving IMU data", exc_info=True)

        if convert_radiance_to_reflectance:
            try:
                self.convert_radiance_images_to_reflectance(**kwargs)
            except FileNotFoundError:
                logger.warning(
                    "Missing input radiance / irradiance files, "
                    "skipping reflectance conversion."
                )
            except Exception:
                logger.error(
                    "Error while converting from radiance to reflectance", exc_info=True
                )

        if glint_correct_reflectance:
            try:
                self.glint_correct_reflectance_images(**kwargs)
            except FileNotFoundError:
                logger.warning(
                    "Missing input reflectance files, skipping glint correction."
                )
            except Exception:
                logger.error(
                    "Error while glint correcting reflectance images", exc_info=True
                )

        if geotiff_from_glint_corrected_reflectance:
            try:
                self.georeference_glint_corrected_reflectance(**kwargs)
            except Exception:
                logger.error(
                    "Error while georeferencing glint corrected images ", exc_info=True
                )

        if mosaic_geotiffs:
            try:
                self.mosaic_geotiffs()
            except Exception:
                logger.error("Error while mosaicing geotiffs ", exc_info=True)


if __name__ == "__main__":
    print(PipelineProcessor._get_image_number("ExampleLocation_Pika_L_5.bil.hdr"))
    # dataset_dir = Path(
    #     "/media/mha114/Massimal2/seabee-minio/larvik/olbergholmen/aerial/hsi/"
    #     + "20230830/massimal_larvik_olbergholmen_202308301001-south-test_hsi"
    # )
    # pl = PipelineProcessor(dataset_dir)
    # pl.run()

    # pl.glint_correct_reflectance_images()
    # pl.georeference_glint_corrected_reflectance(
    #     altitude_offset=-2.2, pitch_offset=3.4, roll_offset=-0.0
    # )
    # )
