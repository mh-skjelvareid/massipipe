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


def copy_visualization_mosaic(source_mosaic: Path, dest_mosaic_dir: Path):
    """Copy mosaic best suited for visualization to separate directory"""
    logger.info("---- COPYING MOSAIC USED FOR VISUALIZATION ----")

    dest_mosaic_dir.mkdir(exist_ok=True)
    if source_mosaic.exists():
        shutil.copy(source_mosaic, dest_mosaic_dir)
    else:
        logger.error(f"Mosaic {source_mosaic} does not exist")


def create_readme_file(dataset_dir: Path):
    """Create a default readme file for the dataset"""
    readme_file_path = dataset_dir / "readme.md"
    try:
        write_readme(readme_file_path)
    except IOError:
        logger.error(f"Error while writing readme file {readme_file_path}")
    return readme_file_path


def create_license_file(dataset_dir: Path):
    """Create a default license file for the dataset"""
    license_file_path = dataset_dir / "license.md"
    try:
        write_license(license_file_path)
    except IOError:
        logger.error(f"Error while writing license file {license_file_path}")
    return license_file_path


def _add_element_to_archive(dataset_dir: Path, archive: zipfile.ZipFile, element: Path):
    """Add element in dataset (file/dir) to opened archive (zip file)"""
    try:
        if element.exists():
            if element.is_dir():
                for file_path in element.rglob("*"):
                    archive.write(file_path, arcname=file_path.relative_to(dataset_dir))
            else:
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
    """Export selected parts of dataset to zip file"""
    logger.info("---- EXPORTING DATASET TO ZIP ARCHIVE ----")

    readme_file_path = create_readme_file(dataset_dir)
    license_file_path = create_license_file(dataset_dir)

    # Create zip file export paths
    zip_dir = dataset_dir / "processed"  # Standard folder for SeaBee processed files
    zip_file_path = zip_dir / (dataset_dir.name + ".zip")

    zip_dir.mkdir(exist_ok=True)

    with zipfile.ZipFile(zip_file_path, mode="w") as archive:
        _add_element_to_archive(dataset_dir, archive, quicklook_dir)
        _add_element_to_archive(dataset_dir, archive, radiance_dir)
        _add_element_to_archive(dataset_dir, archive, imudata_dir)
        _add_element_to_archive(dataset_dir, archive, mosaic_visualization_dir)
        _add_element_to_archive(dataset_dir, archive, config_file_path)
        _add_element_to_archive(dataset_dir, archive, readme_file_path)
        _add_element_to_archive(dataset_dir, archive, license_file_path)

    logger.info(f"Dataset exported to {zip_file_path}")
