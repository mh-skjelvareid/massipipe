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

    for phi, rot_mat_roll in zip([0, -a, a], ray_rot_mat[0:3]):
        for alpha, rot_mat in zip([-a, 0, a], rot_mat_roll):
            expected = np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(alpha + phi), -np.sin(alpha + phi)],
                    [0, np.sin(alpha + phi), np.cos(alpha + phi)],
                ]
            )
            close_mask = np.isclose(rot_mat, expected)
            assert close_mask.all(), f"Matrices differ at indices:\n{np.argwhere(~close_mask)}"


def test_ray_rotation_pitch(sample_angle, sample_camera_model, sample_imu_angles):
    ray_rot_mat = sample_camera_model._ray_rotation_matrices(*sample_imu_angles)
    a = sample_angle

    # Test rotation matrices for pitch / looking angles only (zero roll and yaw)
    for theta, rot_mat_pitch in zip([-a, a], ray_rot_mat[3:5]):
        for phi, rot_mat in zip([-a, 0, a], rot_mat_pitch):
            expected = np.array(
                [
                    [np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi) * np.sin(theta)],
                    [0, np.cos(phi), -np.sin(phi)],
                    [-np.sin(theta), np.sin(phi) * np.cos(theta), np.cos(phi) * np.cos(theta)],
                ]
            )
            close_mask = np.isclose(rot_mat, expected)
            assert close_mask.all(), f"Matrices differ at indices:\n{np.argwhere(~close_mask)}"


def test_ray_rotation_yaw(sample_angle, sample_camera_model, sample_imu_angles):
    ray_rot_mat = sample_camera_model._ray_rotation_matrices(*sample_imu_angles)
    a = sample_angle

    # Test rotation matrices for yaw / looking angles only (zero roll and pitch)
    for psi, rot_mat_yaw in zip([-a, a], ray_rot_mat[5:7]):
        for phi, rot_mat in zip([-a, 0, a], rot_mat_yaw):
            expected = np.array(
                [
                    [np.cos(psi), -np.sin(psi) * np.cos(phi), np.sin(psi) * np.sin(phi)],
                    [np.sin(psi), np.cos(psi) * np.cos(phi), -np.cos(psi) * np.sin(phi)],
                    [0, np.sin(phi), np.cos(phi)],
                ]
            )
            close_mask = np.isclose(rot_mat, expected)
            assert close_mask.all(), f"Matrices differ at indices:\n{np.argwhere(~close_mask)}"


def test_camera_to_ground_simple(sample_camera_model):
    # Simple test case with no rotation and flat terrain
    imu_roll = np.array([0.0])
    imu_pitch = np.array([0.0])
    imu_yaw = np.array([0.0])
    H = 100.0  # 100 meters altitude
    imu_altitude = np.array([H])  # 100 meters altitude

    R_world_cam = sample_camera_model._ray_rotation_matrices(imu_roll, imu_pitch, imu_yaw)
    vectors = sample_camera_model._camera_to_ground_vectors(R_world_cam, imu_altitude)[0]

    for alpha, vector in zip(sample_camera_model.looking_angles, vectors):
        expected_vector = np.array([0, -np.tan(alpha) * H, H])

        assert np.allclose(vector, expected_vector)


def test_ground_offsets_no_yaw(sample_camera_model, sample_imu_angles):
    # Test ground offsets without yaw
    roll, pitch, _ = sample_imu_angles
    yaw = np.zeros_like(roll)
    H = 100.0
    altitude = np.ones_like(roll) * H

    R_world_cam = sample_camera_model._ray_rotation_matrices(roll, pitch, yaw)
    all_vectors = sample_camera_model._camera_to_ground_vectors(R_world_cam, altitude)

    for phi, theta, line_vectors in zip(roll, pitch, all_vectors):
        for alpha, pixel_vector in zip(sample_camera_model.looking_angles, line_vectors):
            expected_vector = H * np.array(
                [np.tan(theta), (-np.tan(alpha + phi) / np.cos(theta)), 1]
            )
            close_mask = np.isclose(pixel_vector, expected_vector)
            assert close_mask.all(), (
                f"{pixel_vector=}, {expected_vector=}"
                f"Vectors differ at indices:\n{np.argwhere(~close_mask)}, {phi=}, {theta=}"
            )
