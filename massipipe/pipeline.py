# Imports
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Union

from pydantic import ValidationError

from massipipe.config import Config, export_template_yaml, read_config, write_config
from massipipe.georeferencing import GeoTransformer, ImuDataParser, SimpleGeoreferencer
from massipipe.glint import FlatSpecGlintCorrector, HedleyGlintCorrector
from massipipe.irradiance import IrradianceConverter, WavelengthCalibrator
from massipipe.mosaic import add_geotiff_overviews, mosaic_geotiffs
from massipipe.quicklook import QuickLookProcessor
from massipipe.radiance import RadianceConverter
from massipipe.reflectance import ReflectanceConverter

# Get logger
logger = logging.getLogger(__name__)


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
        config_file_name: str
            Name of YAML config file for dataset.
            If file does not yet exist, a template file is created.

        Raises
        ------
        FileNotFoundError
            No folder called "0_raw" found
        FileNotFoundError
            No folder called "calibration" found
        """
        self.dataset_dir = Path(dataset_dir)

        # Read and validate YAML config file
        config_file_path = self.dataset_dir / config_file_name
        self.config_file_path = config_file_path
        if not self.config_file_path.exists():
            logger.info(f"No config file found - exporting template file {config_file_name}")
            export_template_yaml(self.config_file_path)
        self.load_config_from_file()  # Reads config from file into self.config

        # Define dataset folder structure
        self.dataset_base_name = self.dataset_dir.name
        self.raw_dir = self.dataset_dir / "0_raw"
        self.quicklook_dir = self.dataset_dir / "quicklook"
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

        # Configure logging
        self._configure_file_logging()

        # Check if data source is raw data or radiance
        self.data_starting_point = self._check_data_starting_point()

        if self.data_starting_point == "raw":
            # Get calibration file paths
            if not self.calibration_dir.exists():
                raise FileNotFoundError(f'Folder "calibration" not found in {self.dataset_dir}')
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

        # Create mosaic file paths
        self.mosaic_rad_gc_path = self.mosaic_dir / (self.dataset_base_name + "_rad_gc_rgb.tiff")
        self.mosaic_refl_gc_path = self.mosaic_dir / (self.dataset_base_name + "_refl_gc_rgb.tiff")

    def load_config_from_file(self):
        """Load or re-load configuration from YAML file"""
        try:
            full_config_dict = read_config(self.config_file_path)
        except IOError:
            logger.error(f"Error parsing config file {self.config_file_path}")
            raise

        try:
            full_config = Config(**full_config_dict)
        except ValidationError as e:
            logger.warning(f"Validation error while processing {self.config_file_path}")
            logger.warning(str(e))
            logger.warning(f"No configuration loaded for {self.dataset_dir}.")
            return
        self.config = full_config.massipipe_options

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
        logger.info(f"File logging for {self.dataset_base_name} initialized.")

    def _check_data_starting_point(self):
        """Determine whether processing starts from raw or radiance data"""
        if self.raw_dir.exists():
            logger.info(f"Found directory {self.raw_dir.name} - processing from raw files.")
            data_starting_point = "raw"
        else:
            if self.radiance_dir.exists():
                logger.info(
                    f"No raw files found, but radiance directory {self.radiance_dir.name} found"
                    + " - processing based on radiance data."
                )
                data_starting_point = "radiance"
            else:
                raise FileNotFoundError(f"Found neither raw or radiance files in dataset dir.")
        return data_starting_point

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
                logger.warning(f"Set of raw files for image {raw_image_path} is incomplete.")
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
        if self.data_starting_point == "raw":
            base_file_names = [
                f"{self.dataset_base_name}_{i:03d}" for i in range(len(self.raw_image_paths))
            ]
        else:  # Radiance as data starting point
            base_file_names = sorted(
                [
                    file.name.split("_radiance")[0]
                    for file in self.radiance_dir.glob("*_radiance*.hdr")
                ]
            )
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
            rad_gc_path = self.radiance_gc_dir / (base_file_name + "_radiance_gc.bip.hdr")
            proc_file_paths.radiance_gc.append(rad_gc_path)

            # Radiance, glint corrected, RGB version
            rad_gc_rgb_path = self.radiance_gc_rgb_dir / (base_file_name + "_radiance_gc_rgb.tiff")
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
            rgc_path = self.reflectance_gc_dir / (base_file_name + "_reflectance_gc.bip.hdr")
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
                raw_image_path.parent / f"{spec_base_name}_downwelling_{image_number}_pre.spec"
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
                "No irradiance calibration file (*.dcp) found in " + f"{self.calibration_dir}"
            )
        else:
            raise ValueError(
                "More than one irradiance calibration file (*.dcp) found in "
                + f"{self.calibration_dir}"
            )

    def create_quicklook_images(self):
        """Create quicklook versions of hyperspectral images"""
        logger.info("---- QUICKLOOK IMAGE GENERATION ----")
        self.quicklook_dir.mkdir(exist_ok=True)
        quicklook_processor = QuickLookProcessor(
            rgb_wl=self.config.general.rgb_wl, percentiles=self.config.quicklook.percentiles
        )

        # Determine whether raw or radiance images are used for quicklook
        if self.raw_dir.exists():
            hyspec_image_paths = self.raw_image_paths
            logger.info(f"Creating quicklook images from raw images")
        else:
            hyspec_image_paths = self.rad_im_paths
            logger.info(f"Creating quicklook images from radiance images")

        for hyspec_image_path, quicklook_image_path in zip(hyspec_image_paths, self.ql_im_paths):
            if quicklook_image_path.exists() and not self.config.quicklook.overwrite:
                logger.info(f"Image {quicklook_image_path.name} exists - skipping.")
                continue

            logger.info(f"Creating quicklook version of {hyspec_image_path.name}")
            try:
                quicklook_processor.create_quicklook_image_file(
                    hyspec_image_path, quicklook_image_path
                )
            except Exception as e:
                logger.warning(f"Error occured while processing {hyspec_image_path}", exc_info=True)
                logger.warning("Skipping file")

    def parse_and_save_imu_data(self):
        """Parse *.lcf and *.times files with IMU data and save as JSON"""
        logger.info("---- IMU DATA PROCESSING ----")
        self.imudata_dir.mkdir(exist_ok=True)
        imu_data_parser = ImuDataParser()
        for lcf_path, times_path, imu_data_path in zip(
            self.lcf_paths, self.times_paths, self.imu_data_paths
        ):
            if imu_data_path.exists() and not self.config.imu_data.overwrite:
                logger.info(f"Image {imu_data_path.name} exists - skipping.")
                continue

            logger.info(f"Processing IMU data from {lcf_path.name}")
            try:
                imu_data_parser.read_and_save_imu_data(lcf_path, times_path, imu_data_path)
            except Exception:
                logger.error(f"Error occured while processing {lcf_path}", exc_info=True)
                logger.error("Skipping file")

    def create_and_save_geotransform(self):
        logger.info("---- GEOTRANSFORM CALCULATION ----")
        self.geotransform_dir.mkdir(exist_ok=True)

        if not any([imu_file.exists() for imu_file in self.imu_data_paths]):
            raise FileNotFoundError("No IMU data files found.")

        # Determine whether raw or radiance images are used
        if self.raw_dir.exists():
            hyspec_image_paths = self.raw_image_paths
        else:
            hyspec_image_paths = self.rad_im_paths

        # Create geotransform for each image
        for hyspec_im_path, imu_data_path, geotrans_path in zip(
            hyspec_image_paths, self.imu_data_paths, self.geotransform_paths
        ):
            if geotrans_path.exists() and not self.config.geotransform.overwrite:
                logger.info(f"Image {geotrans_path.name} exists - skipping.")
                continue

            logger.info(f"Creating and saving geotransform based on {imu_data_path.name}")
            try:
                if imu_data_path.exists() and hyspec_im_path.exists():
                    geotransformer = GeoTransformer(
                        imu_data_path,
                        hyspec_im_path,
                        camera_opening_angle=self.config.geotransform.camera_opening_angle_deg,
                        pitch_offset=self.config.geotransform.pitch_offset_deg,
                        roll_offset=self.config.geotransform.roll_offset_deg,
                        altitude_offset=self.config.geotransform.altitude_offset_m,
                        utm_x_offset=self.config.geotransform.utm_x_offset_m,
                        utm_y_offset=self.config.geotransform.utm_y_offset_m,
                        assume_square_pixels=self.config.geotransform.assume_square_pixels,
                    )
                    geotransformer.save_image_geotransform(geotrans_path)
            except Exception:
                logger.error(f"Error occured while processing {imu_data_path}", exc_info=True)
                logger.error("Skipping file")

    def convert_raw_images_to_radiance(self):
        """Convert raw hyperspectral images (DN) to radiance (microflicks)"""
        logger.info("---- RADIANCE CONVERSION ----")
        self.radiance_dir.mkdir(exist_ok=True)
        radiance_converter = RadianceConverter(
            self.radiance_calibration_file,
            set_saturated_pixels_to_zero=self.config.radiance.set_saturated_pixels_to_zero,
        )
        for raw_image_path, radiance_image_path, geotransform_path in zip(
            self.raw_image_paths, self.rad_im_paths, self.geotransform_paths
        ):
            if radiance_image_path.exists() and not self.config.radiance.overwrite:
                logger.info(f"Image {radiance_image_path.name} exists - skipping.")
                continue

            if not self.config.radiance.add_envi_mapinfo_to_header:
                geotransform_path = None  # path = None -> Mapinfo won't be added/modified

            logger.info(f"Converting {raw_image_path.name} to radiance")
            try:
                radiance_converter.convert_raw_file_to_radiance(
                    raw_image_path, radiance_image_path, geotransform_path=geotransform_path
                )
            except Exception as e:
                logger.warning(f"Error occured while processing {raw_image_path}", exc_info=True)
                logger.warning("Skipping file")

    def convert_raw_spectra_to_irradiance(self):
        """Convert raw spectra (DN) to irradiance (W/(m2*nm))"""
        logger.info("---- IRRADIANCE CONVERSION ----")
        self.radiance_dir.mkdir(exist_ok=True)
        irradiance_converter = IrradianceConverter(self.irradiance_calibration_file)
        for raw_spec_path, irrad_spec_path in zip(self.raw_spec_paths, self.irrad_spec_paths):
            if irrad_spec_path.exists() and not self.config.irradiance.overwrite:
                logger.info(f"Image {irrad_spec_path.name} exists - skipping.")
                continue

            if raw_spec_path.exists():
                logger.info(f"Converting {raw_spec_path.name} to downwelling irradiance")
                try:
                    irradiance_converter.convert_raw_file_to_irradiance(
                        raw_spec_path, irrad_spec_path
                    )
                except Exception:
                    logger.error(f"Error occured while processing {raw_spec_path}", exc_info=True)
                    logger.error("Skipping file")

    def calibrate_irradiance_wavelengths(self):
        """Calibrate irradiance wavelengths using Fraunhofer absorption lines"""
        logger.info("---- IRRADIANCE WAVELENGTH CALIBRATION ----")
        if not (self.radiance_dir.exists()):
            raise FileNotFoundError("Radiance folder with irradiance spectra does not exist")
        wavelength_calibrator = WavelengthCalibrator()
        irradiance_spec_paths = sorted(self.radiance_dir.glob("*.spec.hdr"))
        if irradiance_spec_paths:
            wavelength_calibrator.fit_batch(irradiance_spec_paths)
            for irradiance_spec_path in irradiance_spec_paths:
                logger.info(f"Calibrating wavelengths for {irradiance_spec_path.name}")
                try:
                    wavelength_calibrator.update_header_wavelengths(irradiance_spec_path)
                except Exception:
                    logger.error(
                        f"Error occured while processing {irradiance_spec_path}",
                        exc_info=True,
                    )
                    logger.error("Skipping file")

    def glint_correct_radiance_images(self):
        """Remove water surface reflections of sun and sky light"""
        logger.info("---- RADIANCE GLINT CORRECTION ----")
        self.radiance_gc_dir.mkdir(exist_ok=True)

        # Read glint correction reference information from config
        ref_im_nums = self.config.radiance_gc.reference_image_numbers
        ref_im_ranges = self.config.radiance_gc.reference_image_ranges

        if not (ref_im_nums):
            logger.error("No reference images for sun glint correction specified - aborting.")
            return

        if (ref_im_ranges is not None) and (len(ref_im_nums) != len(ref_im_ranges)):
            raise ValueError(
                "The number of reference image numbers and reference image ranges do not match."
            )

        # Fit glint corrector
        ref_im_paths = [self.rad_im_paths[im_num] for im_num in ref_im_nums]
        glint_corrector = HedleyGlintCorrector(
            smooth_spectra=self.config.radiance_gc.smooth_spectra,
            subtract_dark_spec=self.config.radiance_gc.subtract_dark_spec,
            set_negative_values_to_zero=self.config.radiance_gc.set_negative_values_to_zero,
        )
        logger.info(f"Fitting glint correction model based on image numbers {ref_im_nums}")
        glint_corrector.fit_to_reference_images(ref_im_paths, ref_im_ranges)

        # Run glint correction
        for rad_image, rad_gc_image in zip(self.rad_im_paths, self.rad_gc_im_paths):
            if rad_gc_image.exists() and not self.config.radiance_gc.overwrite:
                logger.info(f"Image {rad_gc_image.name} exists - skipping.")
                continue
            if rad_image.exists():
                logger.info(f"Running glint correction for {rad_image.name}")
                glint_corrector.glint_correct_image_file(rad_image, rad_gc_image)

    def create_glint_corrected_radiance_rgb_geotiff(self):
        """Create georeferenced GeoTIFF versions of glint corrected radiance"""
        logger.info("---- EXPORTING RGB GEOTIFF FOR GLINT CORRECTED RADIANCE ----")
        self.radiance_gc_rgb_dir.mkdir(exist_ok=True)
        georeferencer = SimpleGeoreferencer(rgb_only=True, rgb_wl=self.config.general.rgb_wl)

        if not any([rp.exists() for rp in self.rad_gc_im_paths]):
            logger.warning(f"No radiance images found in {self.radiance_gc_dir}")
        if not any([gtp.exists() for gtp in self.geotransform_paths]):
            logger.warning(f"No geotransform files found in {self.geotransform_dir}")

        for rad_gc_path, geotrans_path, geotiff_path in zip(
            self.rad_gc_im_paths, self.geotransform_paths, self.rad_gc_rgb_im_paths
        ):
            if geotiff_path.exists() and not self.config.radiance_gc_rgb.overwrite:
                logger.info(f"Image {geotiff_path.name} exists - skipping.")
                continue

            if rad_gc_path.exists() and geotrans_path.exists():
                logger.info(f"Georeferencing and exporting RGB version of {rad_gc_path.name}.")
                try:
                    georeferencer.georeference_hyspec_save_geotiff(
                        rad_gc_path,
                        geotrans_path,
                        geotiff_path,
                    )
                except Exception:
                    logger.error(
                        "Error occured while georeferencing RGB version of " f"{rad_gc_path}",
                        exc_info=True,
                    )
                    logger.error("Skipping file")

    def convert_radiance_images_to_reflectance(self):
        """Convert radiance images (microflicks) to reflectance (unitless)"""
        logger.info("---- REFLECTANCE CONVERSION ----")
        self.reflectance_dir.mkdir(exist_ok=True)
        reflectance_converter = ReflectanceConverter(
            wl_min=self.config.reflectance.wl_min,
            wl_max=self.config.reflectance.wl_max,
            conv_irrad_with_gauss=self.config.reflectance.conv_irrad_with_gauss,
            smooth_spectra=self.config.reflectance.smooth_spectra,
            refl_from_mean_irrad=self.config.reflectance.refl_from_mean_irrad,
            irrad_spec_paths=self.irrad_spec_paths,
        )

        if not any([rp.exists() for rp in self.rad_im_paths]):
            raise FileNotFoundError(f"No radiance images found in {self.radiance_dir}")
        if not any([irp.exists() for irp in self.irrad_spec_paths]):
            raise FileNotFoundError(f"No irradiance spectra found in {self.radiance_dir}")

        for rad_path, irrad_path, refl_path in zip(
            self.rad_im_paths, self.irrad_spec_paths, self.refl_im_paths
        ):
            if refl_path.exists() and not self.config.reflectance.overwrite:
                logger.info(f"Image {refl_path.name} exists - skipping.")
                continue
            if rad_path.exists() and irrad_path.exists():
                logger.info(f"Converting {rad_path.name} to reflectance.")
                try:
                    reflectance_converter.convert_radiance_file_to_reflectance(
                        rad_path, irrad_path, refl_path
                    )
                except Exception as e:
                    logger.error(f"Error occured while processing {rad_path}", exc_info=True)
                    logger.error("Skipping file")

    def add_irradiance_to_radiance_header(self):
        """Pre-process irradiance for reflectance calc. and save to radiance header"""
        logger.info("---- WRITING IRRADIANCE TO RADIANCE HEADER ----")
        reflectance_converter = ReflectanceConverter(
            wl_min=None,  # Not used, but set to None to show that ...
            wl_max=None,  # ... original radiance wavelengths are not modified.
            conv_irrad_with_gauss=self.config.reflectance.conv_irrad_with_gauss,
            smooth_spectra=self.config.reflectance.smooth_spectra,
            refl_from_mean_irrad=self.config.reflectance.refl_from_mean_irrad,
            irrad_spec_paths=self.irrad_spec_paths,
        )

        if not any([rp.exists() for rp in self.rad_im_paths]):
            raise FileNotFoundError(f"No radiance images found in {self.radiance_dir}")
        if not any([irp.exists() for irp in self.irrad_spec_paths]):
            raise FileNotFoundError(f"No irradiance spectra found in {self.radiance_dir}")

        for rad_path, irrad_path in zip(self.rad_im_paths, self.irrad_spec_paths):
            if rad_path.exists() and irrad_path.exists():
                logger.info(
                    f"Writing irradiance from {irrad_path.name} "
                    + f"to radiance header {rad_path.name}."
                )
                try:
                    reflectance_converter.add_irradiance_spectrum_to_header(rad_path, irrad_path)
                except Exception as e:
                    logger.error(f"Error occured while processing {rad_path}", exc_info=True)
                    logger.error("Skipping file")

    def glint_correct_reflectance_images(self):
        """Correct for sun and sky glint in reflectance images"""
        logger.info("---- REFLECTANCE GLINT CORRECTION ----")
        self.reflectance_gc_dir.mkdir(exist_ok=True)

        # Calculate reflectance based on glint corrected radiance
        if self.config.reflectance_gc.method == "from_rad_gc":
            logger.info("Calculating glint corrected reflectance from glint corrected radiance")
            reflectance_converter = ReflectanceConverter(
                wl_min=self.config.reflectance.wl_min,
                wl_max=self.config.reflectance.wl_max,
                smooth_spectra=self.config.reflectance_gc.smooth_spectra,
            )

            if not any([rp.exists() for rp in self.rad_gc_im_paths]):
                logger.warning(
                    f"No glint corrected radiance images found in {self.radiance_gc_dir}"
                )

            for rad_gc_path, refl_gc_path in zip(self.rad_gc_im_paths, self.refl_gc_im_paths):
                if refl_gc_path.exists() and not self.config.reflectance_gc.overwrite:
                    logger.info(f"Image {refl_gc_path.name} exists - skipping.")
                    continue
                if rad_gc_path.exists():
                    logger.info(f"Calculating reflectance based on {rad_gc_path.name}.")
                    try:
                        reflectance_converter.convert_radiance_file_with_irradiance_to_reflectance(
                            rad_gc_path, refl_gc_path
                        )
                    except Exception as e:
                        logger.error(
                            f"Error occured while glint correcting {rad_gc_path}",
                            exc_info=True,
                        )
                        logger.error("Skipping file")

        # Calculate glint corrected reflectance based on assumption of flat glint spectrum
        elif self.config.reflectance_gc.method == "flat_spec":
            logger.info("Using flat spectrum method for reflectance glint correction")
            glint_corrector = FlatSpecGlintCorrector(
                smooth_with_savitsky_golay=self.config.reflectance_gc.smooth_spectra
            )

            if not any([rp.exists() for rp in self.refl_im_paths]):
                raise FileNotFoundError(f"No reflectance images found in {self.reflectance_dir}")

            for refl_path, refl_gc_path in zip(self.refl_im_paths, self.refl_gc_im_paths):
                if refl_gc_path.exists() and not self.config.reflectance_gc.overwrite:
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
        else:
            raise ValueError(
                "Invalid reflectance glint correction method: "
                + f"{self.config.reflectance_gc.method}"
            )

    def create_glint_corrected_reflectance_rgb_geotiff(self):
        """Create georeferenced GeoTIFF versions of glint corrected reflectance"""
        logger.info("---- EXPORTING RGB GEOTIFF FOR GLINT CORRECTED REFLECTANCE ----")
        self.reflectance_gc_rgb_dir.mkdir(exist_ok=True)
        georeferencer = SimpleGeoreferencer(rgb_only=True, rgb_wl=self.config.general.rgb_wl)

        if not any([rp.exists() for rp in self.refl_gc_im_paths]):
            logger.warning(f"No reflectance images found in {self.reflectance_gc_dir}")
        if not any([gtp.exists() for gtp in self.geotransform_paths]):
            logger.warning(f"No geotransform files found in {self.geotransform_dir}")

        for refl_gc_path, geotrans_path, geotiff_path in zip(
            self.refl_gc_im_paths, self.geotransform_paths, self.refl_gc_rgb_paths
        ):
            if geotiff_path.exists() and not self.config.reflectance_gc_rgb.overwrite:
                logger.info(f"Image {geotiff_path.name} exists - skipping.")
                continue

            if refl_gc_path.exists() and geotrans_path.exists():
                logger.info(f"Georeferencing and exporting RGB version of {refl_gc_path.name}.")
                try:
                    georeferencer.georeference_hyspec_save_geotiff(
                        refl_gc_path,
                        geotrans_path,
                        geotiff_path,
                    )
                except Exception:
                    logger.error(
                        "Error occured while georeferencing RGB version of " f"{refl_gc_path}",
                        exc_info=True,
                    )
                    logger.error("Skipping file")

    def mosaic_radiance_gc_geotiffs(self):
        """Merge radiance_gc RGB images into mosaic with overviews"""
        logger.info(f"Mosaicing GeoTIFFs in {self.radiance_gc_rgb_dir}")
        self.mosaic_dir.mkdir(exist_ok=True)

        if (self.mosaic_rad_gc_path.exists()) and (
            not self.config.mosaic.radiance_gc_rgb.overwrite
        ):
            logger.info(f"Mosaic {self.mosaic_rad_gc_path} already exists - skipping.")
            return

        if not any([rp.exists() for rp in self.rad_gc_rgb_im_paths]):
            logger.error(f"No images found in {self.radiance_gc_rgb_dir}")
            return

        mosaic_geotiffs(self.rad_gc_rgb_im_paths, self.mosaic_rad_gc_path)
        add_geotiff_overviews(self.mosaic_rad_gc_path, self.config.mosaic.overview_factors)

    def mosaic_reflectance_gc_geotiffs(self):
        """Merge reflectance_gc RGB images into mosaic with overviews"""
        logger.info(f"Mosaicing GeoTIFFs in {self.reflectance_gc_rgb_dir}")
        self.mosaic_dir.mkdir(exist_ok=True)

        if (self.mosaic_refl_gc_path.exists()) and (
            not self.config.mosaic.reflectance_gc_rgb.overwrite
        ):
            logger.info(f"Mosaic {self.mosaic_refl_gc_path} already exists - skipping.")
            return

        if not any([rp.exists() for rp in self.refl_gc_rgb_paths]):
            logger.error(f"No images found in {self.reflectance_gc_dir}")
            return

        mosaic_geotiffs(self.refl_gc_rgb_paths, self.mosaic_refl_gc_path)
        add_geotiff_overviews(self.mosaic_refl_gc_path, self.config.mosaic.overview_factors)

    def delete_existing_products(
        self,
        delete_quicklook: bool = True,
        delete_radiance: bool = True,
        delete_radiance_gc: bool = True,
        delete_reflectance: bool = True,
        delete_reflectance_gc: bool = True,
        delete_geotransform: bool = True,
        delete_imudata: bool = True,
        delete_mosaics: bool = True,
    ):
        """Delete existing image products ("reset" after previous processing)

        Parameters
        ----------
        delete_quicklook : bool, optional
            If true, delete 0b_quicklook folder (if it exists)
        delete_radiance : bool, optional
            If true, delete 1a_radiance folder (if it exists)
            If data "starting point" is radiance and not raw files,
            radiance will not be deleted (delete manually if needed).
        delete_radiance_gc : bool, optional
            If true, delete 1b_radiance folder (if it exists)
        delete_reflectance : bool, optional
            If true, delete 2a_reflectance folder (if it exists)
        delete_reflectance_gc : bool, optional
            If true, delete 2b_reflectance folder (if it exists)
        delete_geotransform : bool, optional
            If true, delete geotransform folder (if it exists)
        delete_imudata : bool, optional
            If true, delete imudata folder (if it exists)
            If data "starting point" is radiance and not raw files,
            imu data will not be deleted (delete manually if needed).
        delete_mosaics : bool, optional
            If true, delete mosaics folder (if it exists)
        """
        if self.quicklook_dir.exists() and delete_quicklook:
            logger.info(f"Deleting {self.quicklook_dir}")
            shutil.rmtree(self.quicklook_dir)
        if (
            self.radiance_dir.exists()
            and delete_radiance
            and not (self.data_starting_point == "radiance")
        ):
            logger.info(f"Deleting {self.radiance_dir}")
            shutil.rmtree(self.radiance_dir)
        if self.radiance_gc_dir.exists() and delete_radiance_gc:
            logger.info(f"Deleting {self.radiance_gc_dir}")
            shutil.rmtree(self.radiance_gc_dir)
        if self.reflectance_dir.exists() and delete_reflectance:
            logger.info(f"Deleting {self.reflectance_dir}")
            shutil.rmtree(self.reflectance_dir)
        if self.reflectance_gc_dir.exists() and delete_reflectance_gc:
            logger.info(f"Deleting {self.reflectance_gc_dir}")
            shutil.rmtree(self.reflectance_gc_dir)
        if self.geotransform_dir.exists() and delete_geotransform:
            logger.info(f"Deleting {self.geotransform_dir}")
            shutil.rmtree(self.geotransform_dir)
        if (
            self.imudata_dir.exists()
            and delete_imudata
            and not (self.data_starting_point == "radiance")
        ):
            logger.info(f"Deleting {self.imudata_dir}")
            shutil.rmtree(self.imudata_dir)
        if self.mosaic_dir.exists() and delete_mosaics:
            logger.info(f"Deleting {self.mosaic_dir}")
            shutil.rmtree(self.mosaic_dir)

    def run_raw_data_processing(self):
        """Run all data processing steps based on raw data"""

        if self.config.imu_data.create:
            try:
                self.parse_and_save_imu_data()
            except Exception:
                logger.error("Error while parsing and saving IMU data", exc_info=True)

        if self.config.radiance.create:
            try:
                self.convert_raw_images_to_radiance()
            except Exception:
                logger.error("Error while converting raw images to radiance", exc_info=True)

        if self.config.irradiance.create:
            try:
                self.convert_raw_spectra_to_irradiance()
            except Exception:
                logger.error("Error while converting raw spectra to irradiance", exc_info=True)

            try:
                self.calibrate_irradiance_wavelengths()
            except Exception:
                logger.error("Error while calibrating irradiance wavelengths", exc_info=True)

        if self.config.radiance.add_irradiance_to_header:
            try:
                self.add_irradiance_to_radiance_header()
            except Exception:
                logger.error("Error while adding irradiance to radiance header", exc_info=True)

    def run_quicklook(self):
        """Create quicklook versions of images (percentile stretched)"""
        if self.config.quicklook.create:
            try:
                self.create_quicklook_images()
            except Exception:
                logger.error("Error while creating quicklook images", exc_info=True)

    def run_geotransform(self):
        """Create geotransform based on IMU data and shape of hyperspectral image"""
        if self.config.geotransform.create:
            try:
                self.create_and_save_geotransform()
            except Exception:
                logger.error("Error while parsing and saving IMU data", exc_info=True)

    def run_reflectance(self):
        """Create reflectance images based on parameters in YAML file"""
        if self.config.reflectance.create:
            try:
                self.convert_radiance_images_to_reflectance()
            except FileNotFoundError:
                logger.warning(
                    "Missing input radiance / irradiance files, "
                    "skipping reflectance conversion.",
                    exc_info=True,
                )
            except Exception:
                logger.error("Error while converting from radiance to reflectance", exc_info=True)

    def run_glint_correction(self):
        """Run glint correction using parameters defined in YAML file

        The processing steps include:
            - Fitting a glint correction model to radiance images
            - Running glint correction and saving glint corrected images
            - Creating RGB GeoTiff versions of glint corrected images

        See massipipe.config.Config and template YAML file for all options.

        """

        if self.config.radiance_gc.create:
            try:
                self.glint_correct_radiance_images()
            except Exception:
                logger.error("Error while glint correcting radiance images", exc_info=True)

        if self.config.radiance_gc_rgb.create:
            try:
                self.create_glint_corrected_radiance_rgb_geotiff()
            except Exception:
                logger.error(
                    "Error while creating RGB GeoTIFFs from glint corrected radiance", exc_info=True
                )

        if self.config.reflectance_gc.create:
            try:
                self.glint_correct_reflectance_images()
            except Exception:
                logger.error("Error while glint correcting radiance images", exc_info=True)

        if self.config.reflectance_gc_rgb.create:
            try:
                self.create_glint_corrected_reflectance_rgb_geotiff()
            except Exception:
                logger.error(
                    "Error while creating RGB GeoTIFFs from glint corrected radiance", exc_info=True
                )

    def run_mosaics(self):
        """Run all mosaicing operations"""
        if self.config.mosaic.radiance_gc_rgb.create:
            try:
                self.mosaic_radiance_gc_geotiffs()
            except Exception:
                logger.error(
                    f"Error occured while mosaicing glint corrected radiance", exc_info=True
                )

        if self.config.mosaic.reflectance_gc_rgb.create:
            try:
                self.mosaic_reflectance_gc_geotiffs()
            except Exception:
                logger.error(
                    f"Error occured while mosaicing glint corrected reflectance", exc_info=True
                )

    def run(self):
        """Run all processing steps"""
        if self.data_starting_point == "raw":
            self.run_raw_data_processing()
        self.run_quicklook()
        self.run_geotransform()
        self.run_reflectance()
        self.run_glint_correction()
        self.run_mosaics()

    def create_template_yaml_config(self):
        """Create YAML file"""


def find_datasets(base_dir: Path, dataset_subdir_search_str="0_raw"):
    """Find dataset paths based on expected subdirectory in dataset"""
    subdirs = base_dir.rglob(dataset_subdir_search_str)
    return sorted([subdir.parent for subdir in subdirs])
