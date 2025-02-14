# Imports
import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import massipipe.utils as mpu

# Get logger
logger = logging.getLogger(__name__)


class QuickLookProcessor:
    def __init__(
        self,
        rgb_wl: Optional[Tuple[float, float, float]] = None,
        percentiles: Optional[Tuple[float, float]] = None,
    ):
        """Initialize quicklook processor.

        Parameters
        ----------
        rgb_wl : Optional[Tuple[float, float, float]], optional
            Wavelengths (in nm) for generating RGB images.
            If None, uses default (640.0, 550.0, 460.0).
        percentiles : Optional[Tuple[float, float]], optional
            Percentile limits for percentile stretching of images.
            If None, uses default (2, 98).
        """
        self.rgb_wl = rgb_wl if rgb_wl else (640.0, 550.0, 460.0)
        self.percentiles = percentiles if percentiles else (2, 98)

    def create_quicklook_image_file(
        self, raw_path: Union[Path, str], quicklook_path: Union[Path, str]
    ) -> None:
        """Create per-band percentile stretched RGB image file from raw hyperspectral image.

        Parameters
        ----------
        raw_path : Union[Path, str]
            Path to raw hyperspectral image (header file).
        quicklook_path : Union[Path, str]
            Path to output PNG image file.
        """
        try:
            image, wl, _ = mpu.read_envi(raw_path)
            rgb_image, _ = mpu.rgb_subset_from_hsi(image, wl, rgb_target_wl=self.rgb_wl)
            rgb_image = mpu.percentile_stretch_image(rgb_image, percentiles=self.percentiles)
            mpu.save_png(rgb_image, quicklook_path)
        except Exception as e:
            logger.error(f"Error creating quicklook image file from {raw_path}", exc_info=True)
            raise
