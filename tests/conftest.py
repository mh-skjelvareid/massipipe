import shutil
import zipfile
from pathlib import Path

import numpy as np
import pytest

import massipipe.utils as mpu

EXAMPLE_DATA_PATH = (
    Path(__file__).parent
    / "example_data"
    / "massimal_larvik_olbergholmen_202308301001-test_hsi.zip"
)

EXAMPLE_CONFIG_PATH = Path(__file__).parent / "example_data" / "example_config.yaml"


@pytest.fixture(scope="session")
def example_dataset_dir(tmp_path_factory):
    """Create temporary folder with example data"""
    # Create temporary directory and extract contents of ZIP file
    tmp_dir = tmp_path_factory.mktemp("example_data")
    with zipfile.ZipFile(EXAMPLE_DATA_PATH, mode="r") as zip_file:
        zip_file.extractall(path=tmp_dir)
    dataset_dir = tmp_dir.glob("*").__next__()  # Dataset "root" folder, child of temp.dir.

    # Copy YAML config file into dataset directory
    shutil.copy(str(EXAMPLE_CONFIG_PATH), str(dataset_dir))

    return dataset_dir


@pytest.fixture
def example_config_file_name():
    return EXAMPLE_CONFIG_PATH.name


@pytest.fixture
def rad_cal_file_path(example_dataset_dir):
    return example_dataset_dir / "calibration" / "RadiometricCal100121-278_081220.icp"


@pytest.fixture
def irrad_cal_file_path(example_dataset_dir):
    return example_dataset_dir / "calibration" / "FLMS16638_Radiometric_Jan2021.dcp"


@pytest.fixture
def example_raw_image(example_dataset_dir):
    """Image, wavelength vector and metadata for raw image"""
    example_raw_image_path = (
        example_dataset_dir / "0_raw" / "OlbergholmenS1-5" / "OlbergholmenS1_Pika_L_5.bil.hdr"
    )
    return mpu.read_envi(example_raw_image_path)


@pytest.fixture
def example_raw_spec(example_dataset_dir):
    """Spectrum, wavelength vector and metadata for raw spectrum"""
    example_raw_spec_path = (
        example_dataset_dir
        / "0_raw"
        / "OlbergholmenS1-5"
        / "OlbergholmenS1_downwelling_5_pre.spec.hdr"
    )
    return mpu.read_envi(example_raw_spec_path)


@pytest.fixture
def example_times_lcf_path(example_dataset_dir):
    """Paths to example *.times and *.lcf files (IMU data)"""
    example_image_dir = example_dataset_dir / "0_raw" / "OlbergholmenS1-5"
    example_times_path = example_image_dir / "OlbergholmenS1_Pika_L_5.bil.times"
    example_lcf_path = example_image_dir / "OlbergholmenS1_Pika_L_5.lcf"
    return (example_times_path, example_lcf_path)
