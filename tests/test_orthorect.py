import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from massipipe.orthorectification import CameraModel, FlatTerrainOrthorectifier


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


def test_ray_rotation_dimensions(sample_camera_model, sample_imu_angles):
    M = len(sample_imu_angles[0])
    N = len(sample_camera_model.looking_angles)
    ray_rot_mat = sample_camera_model._ray_rotation_matrices(*sample_imu_angles)
    assert ray_rot_mat.shape == (M, N, 3, 3)


def test_ray_rotation_roll(sample_camera_model):
    """Test rotation matrices for roll only (zero pitch and yaw)"""
    roll = np.radians([-4, 0, 7, 15])
    pitch = np.zeros_like(roll)
    yaw = np.zeros_like(roll)
    ray_rot_mat = sample_camera_model._ray_rotation_matrices(roll, pitch, yaw)

    for phi, rot_mat_roll in zip(roll, ray_rot_mat):
        for alpha, rot_mat in zip(sample_camera_model.looking_angles, rot_mat_roll):
            expected = np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(alpha + phi), -np.sin(alpha + phi)],
                    [0, np.sin(alpha + phi), np.cos(alpha + phi)],
                ]
            )
            close_mask = np.isclose(rot_mat, expected)
            assert close_mask.all(), f"Matrices differ at indices:\n{np.argwhere(~close_mask)}"


def test_ray_rotation_pitch(sample_camera_model):
    pitch = np.radians([-4, 0, 7, 15])
    roll = np.zeros_like(pitch)
    yaw = np.zeros_like(roll)
    ray_rot_mat = sample_camera_model._ray_rotation_matrices(roll, pitch, yaw)

    # Test rotation matrices for pitch / looking angles only (zero roll and yaw)
    for theta, rot_mat_pitch in zip(pitch, ray_rot_mat):
        for phi, rot_mat in zip(sample_camera_model.looking_angles, rot_mat_pitch):
            expected = np.array(
                [
                    [np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi) * np.sin(theta)],
                    [0, np.cos(phi), -np.sin(phi)],
                    [-np.sin(theta), np.sin(phi) * np.cos(theta), np.cos(phi) * np.cos(theta)],
                ]
            )
            close_mask = np.isclose(rot_mat, expected)
            assert close_mask.all(), f"Matrices differ at indices:\n{np.argwhere(~close_mask)}"


def test_ray_rotation_yaw(sample_camera_model):
    yaw = np.radians([-4, 0, 7, 67, 193, 359])
    roll = np.zeros_like(yaw)
    pitch = np.zeros_like(yaw)
    ray_rot_mat = sample_camera_model._ray_rotation_matrices(roll, pitch, yaw)

    # Test rotation matrices for yaw / looking angles only (zero roll and pitch)
    for psi, rot_mat_yaw in zip(yaw, ray_rot_mat):
        for phi, rot_mat in zip(sample_camera_model.looking_angles, rot_mat_yaw):
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
    """Simple test case with no rotation and flat terrain"""
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


def test_ground_offsets_no_yaw(sample_camera_model):
    """Test ground offsets without yaw"""
    roll = np.radians([-21, -15, 0, 7, 15, 17])
    pitch = np.radians([-9, -8, -2, 0, 10, 13])
    yaw = np.zeros_like(roll)
    H = 100.0
    altitude = np.ones_like(roll) * H

    R_world_cam = sample_camera_model._ray_rotation_matrices(roll, pitch, yaw)
    all_vectors = sample_camera_model._camera_to_ground_vectors(R_world_cam, altitude)

    for phi, theta, line_vectors in zip(roll, pitch, all_vectors):
        for alpha, pixel_vector in zip(sample_camera_model.looking_angles, line_vectors):
            expected_vector = H * np.array(
                [
                    np.tan(theta),
                    (-np.tan(alpha + phi) / np.cos(theta)),
                    1,
                ]
            )
            close_mask = np.isclose(pixel_vector, expected_vector)
            assert close_mask.all(), (
                f"{pixel_vector=}, {expected_vector=}"
                f"Vectors differ at indices:\n{np.argwhere(~close_mask)}, {phi=}, {theta=}"
            )


def test_ground_offsets_with_yaw(sample_camera_model):
    """Test ground offsets with yaw"""
    roll = np.radians([-21, -15, 0, 7, 15, 17])
    pitch = np.radians([-9, -8, -2, 0, 10, 13])
    yaw = np.zeros_like([-4, 0, 37, 88, 178, 352])
    H = 100.0
    altitude = np.ones_like(roll) * H

    R_world_cam = sample_camera_model._ray_rotation_matrices(roll, pitch, yaw)
    all_vectors = sample_camera_model._camera_to_ground_vectors(R_world_cam, altitude)

    for phi, theta, psi, line_vectors in zip(roll, pitch, yaw, all_vectors):
        for alpha, pixel_vector in zip(sample_camera_model.looking_angles, line_vectors):
            expected_vector = H * np.array(
                [
                    np.cos(psi) * np.tan(theta)
                    - np.sin(psi) * (-np.tan(alpha + phi) / np.cos(theta)),
                    np.sin(psi) * np.tan(theta)
                    + np.cos(psi) * (-np.tan(alpha + phi) / np.cos(theta)),
                    1,
                ]
            )
            close_mask = np.isclose(pixel_vector, expected_vector)
            assert close_mask.all(), (
                f"{pixel_vector=}, {expected_vector=}"
                f"Vectors differ at indices:\n{np.argwhere(~close_mask)}, {phi=}, {theta=}, {psi=}"
            )


def test_pixel_ground_positions_simple(sample_camera_model):
    """Test for pixel ground positions in simple no-yaw case"""
    N = 100
    E = -50.0
    H = 100.0
    northing = np.array([N])
    easting = np.array([E])
    altitude = np.array([H])
    roll = np.radians([5])
    pitch = np.radians([-3])
    yaw = np.radians([0])

    calc_positions = sample_camera_model.pixel_ground_positions(
        camera_northing=northing,
        camera_easting=easting,
        camera_altitude=altitude,
        camera_roll=roll,
        camera_pitch=pitch,
        camera_yaw=yaw,
    )

    for alpha, calc_position in zip(sample_camera_model.looking_angles, calc_positions[0, :, :]):
        expected_position = H * np.array(
            [np.tan(pitch), (-np.tan(alpha + roll) / np.cos(pitch))]
        ) + np.array([[N], [E]])
        print(calc_positions)
        assert np.allclose(calc_position, np.squeeze(expected_position), rtol=1e-5)


def test_full_orthorect_no_rot():
    """Test end-to-end pixel location calculation with no rotations

    The camera is set to look straight down from 1000m altitude, so the ground
    pixel locations should form a straight line along the north-south axis. Parameters
    are adjusted so that the GSD is 100m along both axes. The image is placed along the
    middle of UTM zone 1N, just north of the equator. The practical effect of orthorectification
    in this case should be a flip along both axes (180 degree rotation).

    """
    # Define small example image and parameters
    image = np.arange(12).reshape((4, 3))
    H = 1000.0  # 1000 meters altitude
    alpha = 0.1  # Approx 5.7 degrees in radians
    n_pix = 3
    fov = 2 * alpha
    altitude = np.array([H] * 4)
    ll_100m_step = 0.0009047313695974662  # Decimal degrees corr. to approx 100m around equator
    longitude = np.array([-177.0] * 4)  # Center of UTM zone 1N
    latitude = np.array([1.0, 2.0, 3.0, 4.0]) * ll_100m_step  # Around equator, ~100m steps
    roll = np.zeros_like(altitude)
    pitch = np.zeros_like(altitude)
    yaw = np.zeros_like(altitude)
    time = np.array([0.0, 1.0, 2.0, 3.0])

    # Set ground sampling distance slightly larger than raytracing result (avound rounding issues)
    gsd = H * np.tan(alpha) * 1.02

    # Orthorectify
    orthorect = FlatTerrainOrthorectifier(
        camera_cross_track_fov=fov,
        camera_cross_track_n_pixels=n_pix,
        ground_sampling_distance=gsd,
    )
    orthorect_image, area_def, utm_epsg = orthorect.orthorectify_image(
        image,
        time=time,
        latitude=latitude,
        longitude=longitude,
        altitude=altitude,
        roll=roll,
        pitch=pitch,
        yaw=yaw,
    )

    assert orthorect_image.shape == (4, 3)
    assert utm_epsg == 32601  # UTM zone 1N
    assert np.allclose(orthorect_image, np.flipud(np.fliplr(image.astype(np.float32))))


def test_full_orthorect_with_rot():
    """Test end-to-end pixel location calculation with no roll, pitch and yaw

    This tests includes non-zero values for roll, pitch and yaw. To keep the case simple enough
    for quick debugging / analysis, thw followoing angles are chosen
    - Alpha (looking angle): 0.1 radians
    - Roll: 0.05 radians - corresponds to approx 50 meter "left" offset
    - Pitch: 0.2 radians - corresponds to approx. 200 meter forward offset
    - Yaw: pi/2 radians - corresponds to rotation from north to east

    Relatively small roll/pitch angles keeps the ground offsets fairly linear with angle.
    """
    # Define small example image and parameters
    image = np.arange(12).reshape((4, 3))
    H = 1000.0  # 1000 meters altitude
    alpha = 0.1  # Approx 5.7 degrees in radians
    n_pix = 3
    fov = 2 * alpha
    altitude = np.array([H] * 4)
    roll = np.array([alpha / 2] * 4)
    pitch = np.array([2 * alpha] * 4)
    yaw = np.radians([90] * 4)  # Rotate from north to east
    time = np.array([0.0, 1.0, 2.0, 3.0])

    ll_100m_step = 0.0009047313695974662  # Decimal degrees corr. to approx 100m around equator
    longitude = np.array([0.0, 1.0, 2.0, 3.0]) * ll_100m_step - 177  # Moving east in 100 m steps
    latitude = np.array([10 * ll_100m_step] * 4)  # Approx 1000 m north of equator

    # Set ground sampling distance slightly larger than raytracing result (avound rounding issues)
    gsd = H * np.tan(alpha) * 1.05

    # Orthorectify
    orthorect = FlatTerrainOrthorectifier(
        camera_cross_track_fov=fov,
        camera_cross_track_n_pixels=n_pix,
        ground_sampling_distance=gsd,
    )
    orthorect_image, area_def, utm_epsg = orthorect.orthorectify_image(
        image,
        time=time,
        latitude=latitude,
        longitude=longitude,
        altitude=altitude,
        roll=roll,
        pitch=pitch,
        yaw=yaw,
    )

    # Test corners of area definition
    approx_expected_area_extent = np.array(
        [
            500150.0,  # 500 000 + 200 (pitch offset) - 50 (half pixel)
            900.0,  # 1000 - 100 (looking angle) + 50 (roll offset) - 50 (half pixel)
            500550.0,  # 500 000 + 200 (pitch offset) + 100*3 (along-track steps)
            1200.0,  # 1000 + 100 (looking angle) + 50 (roll offset) + 50 (half pixel)
        ]
    )
    assert np.allclose(np.array(area_def.area_extent), approx_expected_area_extent, atol=20)

    # Pixels should correspond to 90 degree counterclockwise rotation
    assert orthorect_image.shape == (3, 4)
    assert orthorect_image[0, 0] == image[0, -1]
    assert orthorect_image[-1, 0] == image[0, 0]
    assert orthorect_image[0, -1] == image[-1, -1]
    assert orthorect_image[-1, -1] == image[-1, 0]

    # Test for correct UTM CRS
    assert utm_epsg == 32601  # UTM zone 1N
