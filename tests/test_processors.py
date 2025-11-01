import numpy as np
import pytest

import massipipe
from massipipe.radiance import RadianceCalibrationDataset


def test_radiance_calibration_dataset(example_dataset_dir, rad_cal_file_path):
    rad_cal_dataset = RadianceCalibrationDataset(rad_cal_file_path)
    dark_frame, _, _, dark_gain, dark_shutter = rad_cal_dataset.get_closest_dark_frame(
        gain=15, shutter=9.274
    )
    assert (dark_gain == 15) and (dark_shutter == 5)
    assert dark_frame.shape == (1, 900, 600)
    assert dark_frame[0, 400, 300] == 1
    rad_conv_frame, _, _ = rad_cal_dataset.get_rad_conv_frame()
    assert rad_conv_frame.shape == (1, 900, 600)
    assert rad_conv_frame[0, 400, 300] == pytest.approx(10.337038)


def test_radiance_converter(rad_cal_file_path, example_raw_image):
    radiance_converter = massipipe.RadianceConverter(rad_cal_file_path)
    raw_image, raw_wl, raw_meta = example_raw_image
    rad_image = radiance_converter.convert_raw_image_to_radiance(raw_image, raw_meta)
    assert rad_image[10, 300, 100] == 441


def test_imu_data_parser(example_times_lcf_path):
    pass


def test_image_flight_metadata(example_times_lcf_path):
    pass


def test_irradiance_converter(irrad_cal_file_path, example_raw_spec):
    irrad_converter = massipipe.IrradianceConverter(irrad_cal_file_path)
    raw_spec, _, raw_spec_meta = example_raw_spec
    irrad_spec = irrad_converter.convert_raw_spectrum_to_irradiance(raw_spec, raw_spec_meta)
    # assert np.all(irrad_spec < 1.0) # Not true if including all bands (wl < 360 nm)
    np.testing.assert_allclose(irrad_spec.squeeze()[500], 0.74095922)
