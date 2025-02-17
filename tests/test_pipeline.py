import numpy as np
import pytest

import massipipe


def test_pipeline_defaults(example_dataset_dir, example_config_file_name):
    pp = massipipe.Pipeline(example_dataset_dir, example_config_file_name)
    pp.run()
    assert len(pp.raw_image_paths) == 2
    assert len(pp.base_file_names) == 2
    assert len(pp.refl_im_paths) == 2
    assert all(f.exists() for f in pp.rad_im_paths)
    assert all(f.exists() for f in pp.irrad_spec_paths)
    assert all(f.exists() for f in pp.refl_im_paths)
    assert all(f.exists() for f in pp.imu_data_paths)
