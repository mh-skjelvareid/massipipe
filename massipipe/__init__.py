"""Massipipe: A Python package for batch processing hyperspectral images.

MassiPipe is a data processing pipeline for hyperspectral images. It focuses on
conversion to radiance, conversion to reflectance, and glint correction of images of
shallow water environments.

The package was developed during a project which used a Resonon Pika-L hyperspectral
camera, and parts of the package are specific to this camera. However, other parts are
more general and can be used with any hyperspectral image.
"""

# Package version
__version__ = "0.2.0"

# Exported API symbols
__all__ = [
    # list of public functions/classes
]

import logging

# Initialize logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
)
