from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

import massipipe.utils as mpu


class CameraModel:
    def __init__(
        self,
        opening_angle_deg: float,
        n_pix: int,
        R_to_imu_from_cam: NDArray | None = None,
        euler_to_imu_from_cam: NDArray | None = None,
    ):
        """
        Initialize the orthorectification parameters.

        Parameters
        ----------
        opening_angle_deg : float
            The opening angle of the camera in degrees.
        n_pix : int
            The number of pixels per image line.
        R_to_imu_from_cam : NDArray, optional
            Direction Cosine Matrix (DCM) representing the rotation from the camera
            to the IMU. If provided, `euler_to_imu_from_cam` must not be provided.
        euler_to_imu_from_cam : NDArray, optional
            Euler angles (yaw, pitch, roll) representing the rotation from the camera
            to the IMU. If provided, `cam_to_imu_rot_dcm` must not be provided.

        Raises
        ------
        ValueError
            If both `cam_to_imu_rot_dcm` and `cam_to_imu_rot_euler` are provided.
        """

        self.opening_angle_deg = opening_angle_deg
        self.n_pix = n_pix

        if R_to_imu_from_cam is not None:
            if euler_to_imu_from_cam is not None:
                raise ValueError("Provide rotation as angles or DCM, not both.")
            else:
                self.R_to_imu_from_cam = Rotation.from_matrix(R_to_imu_from_cam)
        elif euler_to_imu_from_cam is not None:
            yaw, pitch, roll = euler_to_imu_from_cam
            self.R_to_imu_from_cam = Rotation.from_euler("zyx", [yaw, pitch, roll])
        else:
            self.R_to_imu_from_cam = Rotation.from_euler("zyx", [0.0, 0.0, 0.0])  # No rotation

    # def camera_rotations(self, yaw: NDArray, pitch: NDArray, roll: NDArray):
    #     """
    #     Computes the camera rotations by applying a transformation from IMU rotations.

    #     Parameters
    #     ----------
    #     yaw : NDArray
    #         Array of yaw angles (rotation around the Z-axis) in radians.
    #     pitch : NDArray
    #         Array of pitch angles (rotation around the Y-axis) in radians.
    #     roll : NDArray
    #         Array of roll angles (rotation around the X-axis) in radians.

    #     Returns
    #     -------
    #     Rotation
    #         A batch of M rotation matrices representing the camera rotations.
    #         R_cam_to_world = R_imu_to_world @ R_cam_to_imu
    #     """
    #     imu_rot = Rotation.from_euler("zyx", np.column_stack([yaw, pitch, roll]))
    #     return imu_rot * self.cam_to_imu_rot  # Matrix multiplications (batch of M)

    def calc_pixel_looking_angles(self) -> NDArray:
        """Calculate the looking angles for each pixel in the sensor.

        Returns
        -------
        numpy.ndarray
            A 1D array of angles (in radians) corresponding to each pixel
        """

        edge = np.tan(np.radians(self.opening_angle_deg / 2))
        return np.arctan(np.linspace(-edge, edge, self.n_pix))

    def calc_ray_rotation_matrices(self, yaw: NDArray, pitch: NDArray, roll: NDArray) -> NDArray:
        """
        Compute the ray rotation matrices for a camera given its rotation and pixel looking angles.

        Parameters
        ----------
        cam_rot : Rotation
            A `scipy.spatial.transform.Rotation` object representing the camera's rotation.
            The rotation is expected to be in the form of a matrix with shape (M, 3, 3),
            where M is the number of rotation matrices.

        Returns
        -------
        NDArray
            A NumPy array of shape (M, N, 3, 3) representing the ray rotation matrices.
            M is the number of camera rotations, and N is the number of pixel looking angles.
        """

        # Create rotation objects
        R_to_world_from_imu = Rotation.from_euler("zyx", np.column_stack([yaw, pitch, roll]))
        R_to_world_from_cam = R_to_world_from_imu * self.R_to_imu_from_cam
        R_to_cam_from_pixel = Rotation.from_euler("x", self.calc_pixel_looking_angles())

        # Convert to matrices
        R_world_cam = R_to_world_from_cam.as_matrix()  # (M,3,3)
        R_cam_pixel = R_to_cam_from_pixel.as_matrix()  # (N,3,3)

        # Matrix multiplication (last two dims) via broadcasting
        return R_world_cam[:, np.newaxis, :, :] @ R_cam_pixel[np.newaxis, :, :, :]  # (M,N,3,3)

    def intersect_rays_ground(
        self,
        R_to_world_from_pixel: NDArray,
        cam_altitude: NDArray,
    ) -> NDArray:
        """
        Intersects rays with the ground plane to calculate the northing and easting offsets.

        Parameters
        ----------
        R_to_world_from_pixel : NDArray
            An array of shape (N, M, 3, 3) representing the rotation matrices for each pixel.
        cam_alt : NDArray
            A 1D array of shape (M,) representing the altitudes of the camera relative to the ground.

        Returns
        -------
        NDArray
            A 3D array of shape (M, N, 2) containing the direction vectors from the camera to the ground.

        """

        # Parameterization: t corresponds to number of unit vectors to reach ground
        with np.errstate(divide="ignore", invalid="ignore"):
            t = cam_altitude[:, np.newaxis] / R_to_world_from_pixel[:, :, 2, 2]  # (M,N)

        # Direction vector corresponds to last column of rotation matrix
        # (multiplication with unit vector along z-axis, [0,0,1])
        d_hat = R_to_world_from_pixel[:, :, 2, 0:2]  # (M,N,2)

        # Extend direction vector down to ground plane
        return d_hat * t[:, :, np.newaxis]

    def combine_camera_pos_ray_offsets(
        self, camera_northing: NDArray, camera_easting: NDArray, ray_vectors: NDArray
    ) -> NDArray:
        """
        Combine camera positions and ray offsets to compute pixel positions in UTM coordinates.
        """
        # Stack camera positions
        r_camera = np.column_stack((camera_northing, camera_easting))[:, np.newaxis, :]  # (M,1,2)

        # Extract ray offsets in northing and easting
        d_xy = ray_vectors[:, :, 0:2]  # (M,N,2)

        # Combine camera positions and ray offsets via broadcasting
        return r_camera + d_xy

    def pixel_ground_positions(
        self,
        camera_northing: NDArray,
        camera_easting: NDArray,
        camera_altitude: NDArray,
        camera_yaw: NDArray,
        camera_pitch: NDArray,
        camera_roll: NDArray,
    ) -> NDArray:
        # Check that all inputs have same length
        M = len(camera_northing)
        if not all(
            len(x) == M
            for x in [camera_easting, camera_yaw, camera_pitch, camera_roll, camera_altitude]
        ):
            raise ValueError("All input arrays must have the same length")

        # Get camera rotations
        R_to_world_from_pixel = self.calc_ray_rotation_matrices(
            camera_yaw, camera_pitch, camera_roll
        )

        # Calculate ray direction vectors
        ray_vecs = self.intersect_rays_ground(R_to_world_from_pixel, camera_altitude)

        # Calculate pixel world coordinates
        return self.combine_camera_pos_ray_offsets(camera_northing, camera_easting, ray_vecs)

    # def pixel_ground_positions_from_json(self, imu_json_path: Path | str):
    #     imu_data = mpu.read_json(imu_json_path)
    #     expected_keys = ["time", "latitude", "longitude", "altitude", "yaw", "pitch", "roll"]
    #     if not all(x in imu_data for x in expected_keys):
    #         raise ValueError(f"IMU JSON file must correspond to dict with keys {expected_keys}")
    #     return self.pixel_ground_positions(
    #         imu_data["latitude"],
    #         imu_data["longitude"],
    #         imu_data["yaw"],
    #         imu_data["pitch"],
    #         imu_data["roll"],
    #         imu_data["altitude"],
    #     )
