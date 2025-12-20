"""Massipipe: A Python package for batch processing hyperspectral images.

MassiPipe is a data processing pipeline for hyperspectral images. It focuses on
conversion to radiance, conversion to reflectance, and glint correction of images of
shallow water environments.

The package was developed during a project which used a Resonon Pika-L hyperspectral
camera, and parts of the package (radiance and irradiance conversion) are specific to
this camera. However, other parts are more general and can be used with any
hyperspectral image.
"""

# External imports
import logging

# Internal imports
from .config import Config, read_config, update_yaml
from .export import export_dataset_zip
from .georeferencing import (
    FlatTerrainOrthorectifier,
    ImuDataParser,
    ImuGeoTransformer,
    georeferenced_hyspec_to_rgb_geotiff,
)
from .glint import FlatSpecGlintCorrector, HedleyGlintCorrector
from .irradiance import IrradianceConverter
from .mosaic import add_geotiff_overviews, convert_geotiff_to_8bit, mosaic_geotiffs
from .orthorectification import FlatTerrainOrthorectifier_2
from .pipeline import Pipeline, find_datasets
from .quicklook import QuickLookProcessor
from .radiance import RadianceConverter
from .reflectance import ReflectanceConverter
from .utils import (
    closest_wl_index,
    percentile_stretch_image,
    read_envi,
    read_envi_header,
    read_json,
    rgb_subset_from_hsi,
    save_envi,
    save_png,
    savitzky_golay_filter,
    write_envi_header,
)

# Exported API symbols
__all__ = [
    "Config",
    "read_config",
    "ImuDataParser",
    "ImuGeoTransformer",
    "georeferenced_hyspec_to_rgb_geotiff",
    "FlatTerrainOrthorectifier",
    "FlatTerrainOrthorectifier_2",
    "FlatSpecGlintCorrector",
    "HedleyGlintCorrector",
    "IrradianceConverter",
    "add_geotiff_overviews",
    "convert_geotiff_to_8bit",
    "mosaic_geotiffs",
    "Pipeline",
    "find_datasets",
    "QuickLookProcessor",
    "RadianceConverter",
    "ReflectanceConverter",
    "closest_wl_index",
    "percentile_stretch_image",
    "read_envi",
    "read_envi_header",
    "read_json",
    "rgb_subset_from_hsi",
    "save_envi",
    "save_png",
    "savitzky_golay_filter",
    "update_yaml",
    "write_envi_header",
    "export_dataset_zip",
]

# Package version
__version__ = "0.3.1"

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
)
