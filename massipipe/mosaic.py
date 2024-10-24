# Imports
import logging
import subprocess
import warnings
from pathlib import Path
from typing import Iterable, Sequence, Union

import rasterio
import rasterio.merge

# Get logger
logger = logging.getLogger(__name__)


def mosaic_geotiffs(image_paths: Iterable[Path], mosaic_path: Path):
    """Merge non-rotated geotiffs into mosaic with overviews (rasterio)

    Parameters
    ----------
    image_paths : Iterable[Path]
        Paths to images to merge.
        Note that images with rotated geotransforms can NOT be merged.
    mosaic_path : Path
        Path to output GeoTIFF.
    """
    # Open images
    images_to_merge = []
    for image_path in image_paths:
        try:
            if image_path.exists():
                images_to_merge.append(rasterio.open(image_path, "r"))
        except IOError as e:
            logger.warning(f"Error while reading {image_path} - skipping. ")

    # Merge
    try:
        rasterio.merge.merge(images_to_merge, dst_path=mosaic_path)
    except Exception:
        logger.error("Error while mosaicing geotiffs", exc_info=True)
        raise
    finally:
        for opened_image in images_to_merge:
            opened_image.close()


def add_geotiff_overviews(image_path: Path, overview_factors: Sequence[int] = (2, 4, 8, 16, 32)):
    """Add lower-resolution "overviews" to image file

    Parameters
    ----------
    image_path : Path
        Path to image to which overviews will be added (in-place in same file)
        WARNING: Adding overviews can corrupt the file, make sure that the file
        is backed up or can be easily recreated.
    overview_factors : tuple, optional
        Factors by which to downsample original image when creating overviews
        Typically factors of 2 (2,4,8, etc.).
    """
    logger.info(f"Adding overviews to mosaic {image_path.name}")
    try:
        with rasterio.open(image_path, "r+") as mosaic_dataset:
            mosaic_dataset.build_overviews(overview_factors)
    except Exception:
        logger.error(f"Error while adding overviews to mosaic {image_path.name}", exc_info=True)
        raise


def mosaic_geotiffs_gdal_cli(image_paths: Iterable[Path], mosaic_path: Path):
    """Merge rotated geotiffs into single mosaic with overviews (GDAL CLI)"""
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
        *[str(p) for p in image_paths if p.exists()],
        str(mosaic_path),
    ]
    subprocess.run(gdalwarp_args)


def add_overviews_gdal_cli(image_path: Path):
    """Add overviews to geotiff using GDAL CLI

    Parameters
    ----------
    image_path : Path
        Path to image to which overviews will be added, typically a mosaicked geotiff.

    Notes
    -----
    Overview factors are automatically calculated by GDAL CLI
    """

    # Add image pyramids to file
    logger.info(f"Adding image pyramids to mosaic {image_path.name}")
    # Explanation of gdaladdo options used:
    # -r average: Use averaging when resampling to lower spatial resolution
    # -q: Suppress output (be quiet)
    gdaladdo_args = ["gdaladdo", "-q", "-r", "average", str(image_path)]
    subprocess.run(gdaladdo_args)
