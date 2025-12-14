import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from massipipe.orthorectification import CameraModel


@pytest.fixture
def sample_angle():
    return np.radians(15)  # 30 degrees in radians


@pytest.fixture
def sample_imu_angles(sample_angle):
    """Sample IMU angles for testing"""
    a = sample_angle
    roll = np.array([0, -a, a, 0, 0, 0, 0, a, a, 0, a], dtype=float)
    pitch = np.array([0, 0, 0, -a, a, 0, 0, a, 0, a, a], dtype=float)
    yaw = np.array([0, 0, 0, 0, 0, -a, a, 0, a, a, a], dtype=float)
    return (roll, pitch, yaw)


@pytest.fixture
def sample_imu_data():
    """Sample IMU data for testing"""
    imu_data = {
        "time": [0, 1, 2, 3, 4],
        "roll": [0.0, 0.1, 0.2, 0.1, 0.0],
        "pitch": [0.0, -0.1, -0.2, -0.1, 0.0],
        "yaw": [0.0, 0.5, 1.0, 0.5, 0.0],
        "altitude": [100, 101, 102, 101, 100],
        "latitude": [65, 65, 65, 65, 65],
        "longitude": [5, 5, 5, 5, 5],
    }
    return imu_data


@pytest.fixture
def sample_camera_model(sample_angle):
    return CameraModel(cross_track_fov=2 * sample_angle, n_pix=3)


def test_looking_angles(sample_angle, sample_camera_model):
    looking_angles = sample_camera_model.looking_angles
    assert np.all(looking_angles == np.array([-sample_angle, 0, sample_angle]))


def test_ray_rotation_dimensions(sample_angle, sample_camera_model, sample_imu_angles):
    M = len(sample_imu_angles[0])
    N = len(sample_camera_model.looking_angles)
    ray_rot_mat = sample_camera_model._ray_rotation_matrices(*sample_imu_angles)
    assert ray_rot_mat.shape == (M, N, 3, 3)


def test_ray_rotation_roll(sample_angle, sample_camera_model, sample_imu_angles):
    ray_rot_mat = sample_camera_model._ray_rotation_matrices(*sample_imu_angles)
    a = sample_angle
    # Zero IMU roll at first timepoint
    assert np.allclose(
        ray_rot_mat[0, 0],
        np.array([[1, 0, 0], [0, np.cos(-a), -np.sin(-a)], [0, np.sin(-a), np.cos(-a)]]),
    )
    assert np.allclose(ray_rot_mat[0, 1], Rotation.identity().as_matrix())
    assert np.allclose(
        ray_rot_mat[0, 2],
        np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]]),
    )
    # Non-zero IMU roll at second timepoint
    assert np.allclose(
        ray_rot_mat[1, 0],
        np.array(
            [[1, 0, 0], [0, np.cos(-2 * a), -np.sin(-2 * a)], [0, np.sin(-2 * a), np.cos(-2 * a)]]
        ),
    )
    assert np.allclose(
        ray_rot_mat[1, 1],
        np.array([[1, 0, 0], [0, np.cos(-a), -np.sin(-a)], [0, np.sin(-a), np.cos(-a)]]),
    )
    assert np.allclose(ray_rot_mat[1, 2], Rotation.identity().as_matrix())


def test_ray_rotation_pitch(sample_angle, sample_camera_model, sample_imu_angles):
    pass
    # ray_rot_mat = sample_camera_model._ray_rotation_matrices(*sample_imu_angles)
    # a = sample_angle
    # assert np.allclose(
    #     ray_rot_mat[0, 0],
    #     np.array([[1, 0, 0], [0, np.cos(-a), -np.sin(-a)], [0, np.sin(-a), np.cos(-a)]]),
    # )
    # assert np.all(ray_rot_mat[0, 1] == Rotation.identity().as_matrix())
    # assert np.allclose(
    #     ray_rot_mat[0, 2],
    #     np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]]),
    # )
