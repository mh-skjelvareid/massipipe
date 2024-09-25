# Imports
import logging
from pathlib import Path
from typing import Union

import massipipe.utils as mpu

# Get logger
logger = logging.getLogger(__name__)


class QuickLookProcessor:
    def __init__(
        self,
        rgb_wl: tuple[float, float, float] = (640.0, 550.0, 460.0),
        percentiles: tuple[float, float] = (2, 98),
    ):
        """Initialize quicklook processor"""
        self.rgb_wl = rgb_wl
        self.percentiles = percentiles

    def create_quicklook_image_file(
        self, raw_path: Union[Path, str], quicklook_path: Union[Path, str]
    ):
        """Create per-band percentile stretched RGB image file from raw hyspec image

        Parameters
        ----------
        raw_path : Union[Path, str]
            Path to raw hyperspectral image (header file)
        quicklook_path : Union[Path, str]
            Path to output PNG image file
        """
        image, wl, _ = mpu.read_envi(raw_path)
        rgb_image, _ = mpu.rgb_subset_from_hsi(image, wl, rgb_target_wl=self.rgb_wl)
        rgb_image = mpu.percentile_stretch_image(
            rgb_image, percentiles=self.percentiles
        )
        mpu.save_png(rgb_image, quicklook_path)
