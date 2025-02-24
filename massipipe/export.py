"""Module for exporting dataset to zip archive"""

# Imports
from __future__ import annotations

import logging
import shutil
import zipfile
from pathlib import Path

from massipipe.license import write_license
from massipipe.readme import write_readme

# Get logger
logger = logging.getLogger(__name__)


def configure_export_log_file_handler(file_handler: logging.FileHandler):
    """Configure the export module logger to use the provided file handler."""
    logger.addHandler(file_handler)


def copy_visualization_mosaic(source_mosaic: Path, dest_mosaic_dir: Path):
    """Copy mosaic best suited for visualization to a separate directory.

    Parameters
    ----------
    source_mosaic : Path
        The path to the source mosaic file that needs to be copied.
    dest_mosaic_dir : Path
        The directory where the mosaic file will be copied to.

    Notes
    -----
    This function will create the destination directory if it does not already exist.
    """

    logger.info("---- COPYING MOSAIC USED FOR VISUALIZATION ----")
    dest_mosaic_dir.mkdir(exist_ok=True)
    if source_mosaic.exists():
        logger.info(f"Copying {source_mosaic.name} to {dest_mosaic_dir}")
        shutil.copy2(source_mosaic, dest_mosaic_dir)
    else:
        logger.error(f"Mosaic {source_mosaic} does not exist")


def create_readme_file(dataset_dir: Path):
    """Create a default readme file for the dataset.

    Parameters
    ----------
    dataset_dir : Path
        The directory where the dataset is stored.

    Returns
    -------
    Path
        The path to the created readme file.
    """
    readme_file_path = dataset_dir / "readme.md"
    try:
        logger.info(f"Writing readme file {readme_file_path}")
        write_readme(readme_file_path)
    except IOError:
        logger.error(f"Error while writing readme file {readme_file_path}")
    return readme_file_path


def create_license_file(dataset_dir: Path):
    """Create a default license file for the dataset.

    Parameters
    ----------
    dataset_dir : Path
        The directory where the dataset is stored.

    Returns
    -------
    Path
        The path to the created license file.
    """
    license_file_path = dataset_dir / "license.md"
    try:
        logger.info(f"Writing license file {license_file_path}")
        write_license(license_file_path)
    except IOError:
        logger.error(f"Error while writing license file {license_file_path}")
    return license_file_path


def _add_element_to_archive(dataset_dir: Path, archive: zipfile.ZipFile, element: Path):
    """Add element in dataset (file/dir) to opened archive (zip file)"""
    # TODO: Exclude hidden directories and files, e.g. ".ipynb_checkpoints"
    try:
        if element.exists():
            if element.is_dir():
                for file_path in sorted(element.rglob("*")):
                    logger.info(f"Adding {file_path.relative_to(dataset_dir)} to archive.")
                    archive.write(file_path, arcname=file_path.relative_to(dataset_dir))
            else:
                logger.info(f"Adding {element.relative_to(dataset_dir)} to archive.")
                archive.write(element, arcname=element.relative_to(dataset_dir))
        else:
            logger.warning(f"Element {element.relative_to(dataset_dir)} does not exist.")
    except Exception:
        logger.error(f"Error while writing {element.relative_to(dataset_dir)} to archive.")


def export_dataset_zip(
    dataset_dir: Path,
    quicklook_dir: Path,
    radiance_dir: Path,
    imudata_dir: Path,
    mosaic_visualization_dir: Path,
    config_file_path: Path,
):
    """Export selected parts of dataset to a zip file.

    Parameters
    ----------
    dataset_dir : Path
        The directory containing the dataset to be exported.
    quicklook_dir : Path
        The directory containing quicklook images.
    radiance_dir : Path
        The directory containing radiance data.
    imudata_dir : Path
        The directory containing IMU data.
    mosaic_visualization_dir : Path
        The directory containing mosaic visualizations.
    config_file_path : Path
        The path to the configuration file.

    Returns
    -------
    Path
        The path to the created zip file.

    Notes
    -----
    This function creates a zip archive containing the specified parts of the dataset,
    including quicklook images, radiance data, IMU data, mosaic visualizations, and
    configuration file. It also includes a README and LICENSE file generated for the dataset.
    """

    logger.info("---- EXPORTING DATASET TO ZIP ARCHIVE ----")

    readme_file_path = create_readme_file(dataset_dir)
    license_file_path = create_license_file(dataset_dir)

    # Create zip file export paths
    zip_dir = dataset_dir / "processed"  # Standard folder for SeaBee processed files
    zip_file_path = zip_dir / (dataset_dir.name + ".zip")

    zip_dir.mkdir(exist_ok=True)

    with zipfile.ZipFile(zip_file_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        _add_element_to_archive(dataset_dir, archive, quicklook_dir)
        _add_element_to_archive(dataset_dir, archive, radiance_dir)
        _add_element_to_archive(dataset_dir, archive, imudata_dir)
        _add_element_to_archive(dataset_dir, archive, mosaic_visualization_dir)
        _add_element_to_archive(dataset_dir, archive, config_file_path)
        _add_element_to_archive(dataset_dir, archive, readme_file_path)
        _add_element_to_archive(dataset_dir, archive, license_file_path)

    logger.info(f"Dataset exported to {zip_file_path}")

    return zip_file_path
