import pytest

import massipipe.utils as mpu


def test_wavelength_array_to_header_string():
    wl_str = mpu.wavelength_array_to_header_string([420.32, 500, 581.28849])
    assert wl_str == "{420.320, 500.000, 581.288}"


def test_gps2unix():
    assert mpu.gps2unix(1331120601) == 1647085383
    assert mpu.gps2unix(1000000000) == 1315964785


def test_unix2gps():
    assert mpu.unix2gps(1647085383) == 1331120601
    assert mpu.unix2gps(1315964785) == 1000000000
