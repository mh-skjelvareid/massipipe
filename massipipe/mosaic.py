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


def add_geotiff_overviews(
    image_path: Path, overview_factors: Sequence[int] = (2, 4, 8, 16, 32)
) -> None:
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


def mosaic_geotiffs_gdal_cli(image_paths: Iterable[Path], mosaic_path: Path) -> None:
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


def add_overviews_gdal_cli(image_path: Path) -> None:
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


def convert_geotiff_to_8bit(
    input_image_path: Path,
    output_image_path: Path,
    lower_percentile: float = 2,
    upper_percentile: float = 98,
    require_positive: bool = True,
) -> None:
    """Convert geotiff to 8-bit using percentile stretching

    Parameters
    ----------
    input_image_path : Path
        Path to original geotiff
    output_image_path : Path
        Path to output geotiff (8-bit)
    lower_percentile : float, default 2
        Percentile value from input image, used to set lower end of
        range used when scaling data to 8-bit output.
    upper_percentile : float, default 98
        Percentile value from input image, used to set upper end of
        range used when scaling data to 8-bit output.
    require_positive : bool, default True
        Whether to set a hard limit on the input values (must be positive)
        If True, the minimum value in range included for output is given by
        max(0,lower_percentile_value)
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
