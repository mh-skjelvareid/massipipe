"""Module for GeoTIFF mosaic operations, overview generation, and 8-bit conversion.
This module provides functions to mosaic GeoTIFF images, add overviews using both
Rasterio and GDAL CLI, and convert GeoTIFF images to 8-bit using percentile stretching.
"""

# Imports
import logging
import subprocess
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import rasterio
import rasterio.merge

# Get logger
logger = logging.getLogger(__name__)


def mosaic_geotiffs(image_paths: Iterable[Path], mosaic_path: Path) -> None:
    """
    Merge non-rotated GeoTIFFs into a single mosaic file, and generate overviews using
    Rasterio.

    Parameters
    ----------
    image_paths : Iterable[Path]
        Iterable of paths to input GeoTIFF images. Images with rotated geotransforms are
        not supported.
    mosaic_path : Path
        Path to the output mosaic GeoTIFF file.

    Raises
    ------
    Exception
        If an error occurs during the merging process.
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


def add_geotiff_overviews(
    image_path: Path, overview_factors: Sequence[int] = (2, 4, 8, 16, 32)
) -> None:
    """
    Add lower-resolution overviews to a GeoTIFF image file using Rasterio.

    Parameters
    ----------
    image_path : Path
        Path to the GeoTIFF image to which overviews will be added in-place.
    overview_factors : Sequence[int], optional
        Downsampling factors for the overviews (default: (2, 4, 8, 16, 32)). Typically
        factors of 2 are used (2, 4, 8, etc.).

    Raises
    ------
    Exception
        If an error occurs while building the overviews.
    """
    logger.info(f"Adding overviews to mosaic {image_path.name}")
    try:
        with rasterio.open(image_path, "r+") as mosaic_dataset:
            mosaic_dataset.build_overviews(overview_factors)
    except Exception:
        logger.error(f"Error while adding overviews to mosaic {image_path.name}", exc_info=True)
        raise


def mosaic_geotiffs_gdal_cli(image_paths: Iterable[Path], mosaic_path: Path) -> None:
    """
    Merge rotated GeoTIFF images into a single mosaic file with overviews using the GDAL
    CLI.

    Parameters
    ----------
    image_paths : Iterable[Path]
        Iterable of paths to input GeoTIFF images. Only images that exist will be
        processed.
    mosaic_path : Path
        Destination path for the output mosaic GeoTIFF file.

    Notes
    -----
    This function leverages the 'gdalwarp' CLI tool with options to overwrite, suppress
    output, use nearest neighbor resampling, and specify the GeoTIFF output format.
    """
    # Explanation of gdalwarp options used: -overwrite: Overwrite existing files without
    # error / warning -q: Suppress GDAL output (quiet) -r near: Resampling method:
    # Nearest neighbor -of GTiff: Output format: GeoTiff

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


def add_overviews_gdal_cli(image_path: Path) -> None:
    """
    Add overviews (image pyramids) to a GeoTIFF using the GDAL CLI.

    Parameters
    ----------
    image_path : Path
        Path to the GeoTIFF image where overviews will be added, typically a mosaicked
        file.

    Notes
    -----
    Overview factors are automatically determined by the GDAL CLI. This function uses
    'gdaladdo' with average resampling and quiet mode.
    """
    # Add image pyramids to file
    logger.info(f"Adding image pyramids to mosaic {image_path.name}")

    # Explanation of gdaladdo options used: -r average: Use averaging when resampling to
    # lower spatial resolution -q: Suppress output (be quiet)
    gdaladdo_args = ["gdaladdo", "-q", "-r", "average", str(image_path)]
    subprocess.run(gdaladdo_args)


def convert_geotiff_to_8bit(
    input_image_path: Path,
    output_image_path: Path,
    lower_percentile: float = 2,
    upper_percentile: float = 98,
    require_positive: bool = True,
) -> None:
    """
    Convert a GeoTIFF image to an 8-bit representation using percentile stretching.

    Parameters
    ----------
    input_image_path : Path
        Path to the original GeoTIFF image.
    output_image_path : Path
        Path to save the 8-bit output GeoTIFF image.
    lower_percentile : float, optional
        Lower percentile (default 2) to set the minimum scaling value.
    upper_percentile : float, optional
        Upper percentile (default 98) to set the maximum scaling value.
    require_positive : bool, optional
        If True, enforces a lower bound of zero on the scaling; defaults to True.

    Raises
    ------
    Exception
        If an error occurs during the conversion process.
    """
    no_data_value_8bit = 255

    logger.info(f"Converting {input_image_path.name} to 8-bit geotiff")
    if output_image_path is None:
        output_image_path = input_image_path

    try:
        with rasterio.open(input_image_path) as input_dataset:
            input_image = input_dataset.read()
            output_image = np.zeros(input_image.shape, dtype=np.uint8)

            for band_index in range(input_dataset.count):
                input_band = input_image[band_index]
                nodata_mask = input_band == input_dataset.nodata

                # Determine input range to use in outut
                range_min, range_max = np.percentile(
                    input_band[~nodata_mask], (lower_percentile, upper_percentile)
                )
                if require_positive:
                    range_min = max(0, range_min)

                # Scale using linear interpolation from input to output range
                output_band = np.interp(
                    input_band, (range_min, range_max), (0, no_data_value_8bit - 1)
                )

                # Transfer nodata mask
                output_band[nodata_mask] = no_data_value_8bit

                # Set output band as slice
                output_image[band_index] = output_band

            # Copy metadata, changing only data type and nodata value
            output_profile = input_dataset.profile.copy()
            output_profile["dtype"] = output_image.dtype
            output_profile["nodata"] = no_data_value_8bit

        with rasterio.open(output_image_path, "w", **output_profile) as output_dataset:
            output_dataset.write(output_image)

    except Exception:
        logger.error(f"Error while converting {input_image_path.name} to 8-bit", exc_info=True)
        raise
