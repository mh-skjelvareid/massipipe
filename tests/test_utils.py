import pytest

import massipipe.utils as mpu


def test_gps2unix():
    assert mpu.gps2unix(1331120601) == 1647085383
    assert mpu.gps2unix(1000000000) == 1315964785


def test_unix2gps():
    assert mpu.unix2gps(1647085383) == 1331120601
    assert mpu.unix2gps(1315964785) == 1000000000
