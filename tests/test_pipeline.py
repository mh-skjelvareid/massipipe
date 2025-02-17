import numpy as np
import pytest

import massipipe


def test_pipeline_end_to_end(example_dataset_dir):
    pp = massipipe.Pipeline(example_dataset_dir)
    pp.config.radiance_gc.create = True  # Enable creation of radiance images
    pp.config.radiance_gc_rgb.create = True  # Enable creation of radiance RGB images
    pp.config.radiance_gc.reference_image_numbers = [0, 1]  # Add glint correction reference images
    pp.config.mosaic.radiance_gc_rgb.create = True  # Enable creation of glint corr. radiance mosaic

    pp.run()
    assert len(pp.raw_image_paths) == 2
    assert len(pp.base_file_names) == 2
    assert len(pp.refl_im_paths) == 2

    # Check that all expected output files exist
    assert all(f.exists() for f in pp.irrad_spec_paths)
    assert all(f.exists() for f in pp.rad_im_paths)
    assert all(f.exists() for f in pp.rad_rgb_paths)
    assert all(f.exists() for f in pp.refl_im_paths)
    assert all(f.exists() for f in pp.imu_data_paths)
    assert all(f.exists() for f in pp.geotransform_paths)
    assert all(f.exists() for f in pp.rad_gc_im_paths)
    assert all(f.exists() for f in pp.rad_gc_rgb_paths)
    assert pp.mosaic_rad_path.exists()
    assert pp.mosaic_rad_gc_path.exists()


def test_pipeline_export(example_dataset_dir):
    pp = massipipe.Pipeline(example_dataset_dir)
    pp.export()
    zip_file_path = example_dataset_dir / "processed" / (example_dataset_dir.name + ".zip")
    assert zip_file_path.exists()
