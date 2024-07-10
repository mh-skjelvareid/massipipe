import zipfile
from pathlib import Path

import numpy as np
import pytest

import massipipe.processors


def test_radiance_calibration_dataset(example_dataset_dir):
    rad_cal_file = (
        example_dataset_dir / "calibration" / "RadiometricCal100121-278_081220.icp"
    )
    rad_cal_dataset = massipipe.processors.RadianceCalibrationDataset(rad_cal_file)
    dark_frame, _, _, dark_gain, dark_shutter = rad_cal_dataset.get_closest_dark_frame(
        gain=15, shutter=9.274
    )
    assert dark_gain == 15
    assert dark_shutter == 5

    rad_conv_frame, _, _ = rad_cal_dataset.get_rad_conv_frame()
    assert rad_conv_frame.dtype == np.float32
