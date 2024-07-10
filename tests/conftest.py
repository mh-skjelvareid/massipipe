import zipfile
from pathlib import Path

import numpy as np
import pytest

EXAMPLE_DATA_PATH = (
    Path(__file__).parent
    / "example_data"
    / "massimal_larvik_olbergholmen_202308301001-test_hsi.zip"
)


@pytest.fixture(scope="session")
def example_dataset_dir(tmp_path_factory):
    """Create temporary folder with example data"""
    tmp_dir = tmp_path_factory.mktemp("example_data")
    with zipfile.ZipFile(EXAMPLE_DATA_PATH, mode="r") as zip_file:
        zip_file.extractall(path=tmp_dir)
    dataset_dir = tmp_dir.glob("*").__next__()  # Dataset "root" folder
    return dataset_dir
