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
        imu_to_cam_rot_dcm: NDArray | None = None,
        imu_to_cam_rot_euler: NDArray | None = None,
        estimate_yaw_from_positions: bool = False,
    ):
        """
        Initialize the orthorectification parameters.

        Parameters
        ----------
        opening_angle_deg : float
            The opening angle of the camera in degrees.
        n_pix : int
            The number of pixels per image line.
        imu_to_cam_rot_dcm : NDArray, optional
            Direction Cosine Matrix (DCM) representing the rotation from the IMU
            to the camera. If provided, `imu_to_cam_rot_euler` must not be provided.
        imu_to_cam_rot_euler : NDArray, optional
            Euler angles (yaw, pitch, roll) representing the rotation from the IMU
            to the camera. If provided, `imu_to_cam_rot_dcm` must not be provided.
        estimate_yaw_from_positions : bool, optional
            Whether to estimate the yaw angle from positions. Default is False.

        Raises
        ------
        ValueError
            If both `imu_to_cam_rot_dcm` and `imu_to_cam_rot_euler` are provided.
        """

        self.opening_angle_deg = opening_angle_deg
        self.n_pix = n_pix

        if imu_to_cam_rot_dcm is not None:
            if imu_to_cam_rot_euler is not None:
                raise ValueError("Provide rotation as angles or DCM, not both.")
            else:
                self.imu_to_cam_rot = Rotation.from_matrix(imu_to_cam_rot_dcm)
        elif imu_to_cam_rot_euler is not None:
            yaw, pitch, roll = imu_to_cam_rot_euler
            self.imu_to_cam_rot = Rotation.from_euler("zyx", [yaw, pitch, roll])
        else:
            self.imu_to_cam_rot = Rotation.from_euler("zyx", [0.0, 0.0, 0.0])
        self.estimate_yaw_from_positions = estimate_yaw_from_positions

    def camera_rotations(self, yaw: NDArray, pitch: NDArray, roll: NDArray):
        """
        Computes the camera rotations by applying a transformation from IMU rotations.

        Parameters
        ----------
        yaw : NDArray
            Array of yaw angles (rotation around the Z-axis) in radians.
        pitch : NDArray
            Array of pitch angles (rotation around the Y-axis) in radians.
        roll : NDArray
            Array of roll angles (rotation around the X-axis) in radians.

        Returns
        -------
        Rotation
            A batch of M rotation matrices representing the camera rotations.
        """
        imu_rot = Rotation.from_euler(
            "zyx",
            np.column_stack([yaw, pitch, roll]),
        )

        return self.imu_to_cam_rot * imu_rot  # Batch of M rotations

    def pixel_looking_angles(self) -> NDArray:
        """Calculate the looking angles for each pixel in the sensor.

        Returns
        -------
        numpy.ndarray
            A 1D array of angles (in radians) corresponding to each pixel
        """

        edge = np.tan(np.radians(self.opening_angle_deg / 2))
        return np.arctan(np.linspace(-edge, edge, self.n_pix))

    def ray_rotations(self, cam_rot: Rotation) -> NDArray:
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
        # TODO: Double-check the order of rotation matrices here
        cam_rot_mat = cam_rot.as_matrix()  # (M, 3, 3)
        pixel_rot_mat = Rotation.from_euler("x", self.pixel_looking_angles()).as_matrix()
        return cam_rot_mat[:, np.newaxis, :, :] @ pixel_rot_mat[np.newaxis, ::, :, :]  # (M,N,3,3)

    def intersect_rays_ground(self, ray_rot: NDArray, cam_alt: NDArray) -> NDArray:
        """
        Intersects rays with the ground plane to calculate the northing and easting offsets.

        Parameters
        ----------
        ray_rot : NDArray
            A 3D array of shape (N, M, 3, 3) representing the rotation matrices for the rays.
        cam_alt : NDArray
            A 1D array of shape (M,) representing the altitudes of the camera relative to the ground.

        Returns
        -------
        NDArray
            A 3D array of shape (M, N, 2) containing the northing and easting offsets for each ray.

        """

        # TODO: Double-check the multiplcation order for rotation matrices here
        # The nadir ray should be rotated first in the camera frame, then to the IMU frame,
        # then to the world frame

        # Create unit vectors for each ray
        ray_nadir = np.array([[0.0, 0.0, 1.0]]).T  # (3,1)
        rays = ray_rot @ ray_nadir  # (N,M,3,1)

        # Parameterization: t corresponds to number of unit vectors to reach ground
        with np.errstate(divide="ignore", invalid="ignore"):
            t = cam_alt[:, np.newaxis] / rays[:, :, 2, 1]  # (M,N)

        # Calculate northing and easting offsets
        return rays[:, :, 0:2, 0] * t[:, :, np.newaxis]  # (M,N,2)

    def combine_camera_pos_ray_offsets(
        self, camera_lat: NDArray, camera_long: NDArray, ray_offsets: NDArray
    ) -> tuple[NDArray, int]:
        """
        Combine camera positions and ray offsets to compute pixel positions in UTM coordinates.

        Parameters
        ----------
        camera_lat : NDArray
            Array of camera latitude positions (in degrees).
        camera_long : NDArray
            Array of camera longitude positions (in degrees).
        ray_offsets : NDArray
            Array of ray offsets in UTM coordinates (shape: (M, N, 2)).

        Returns
        -------
        tuple[NDArray, int]
            A tuple containing:
            - pixel_utm_ne (NDArray): Array of pixel positions in UTM coordinates (shape: (M, N, 2)).
            - utm_epsg (int): EPSG code of the UTM zone used for the conversion.
        """

        cam_utm_e, cam_utm_n, utm_epsg = mpu.convert_long_lat_to_utm(camera_long, camera_lat)
        cam_utm_ne = np.column_stack((cam_utm_n, cam_utm_e))[:, np.newaxis, :]  # (M,1,2)
        pixel_utm_ne = cam_utm_ne + ray_offsets  # (M,N,2)
        return pixel_utm_ne, utm_epsg

    def pixel_ground_positions(
        self,
        camera_lat: NDArray,
        camera_long: NDArray,
        camera_yaw: NDArray,
        camera_pitch: NDArray,
        camera_roll: NDArray,
        camera_alt: NDArray,
    ) -> tuple[NDArray, int]:
        # Check that all inputs have same length
        M = len(camera_lat)
        if not all(
            len(x) == M for x in [camera_long, camera_yaw, camera_pitch, camera_roll, camera_alt]
        ):
            raise ValueError("All input arrays must have the same length")

        # Get camera rotations
        cam_rot = self.camera_rotations(camera_yaw, camera_pitch, camera_roll)

        # Calculate ray rotation matrices
        ray_rot = self.ray_rotations(cam_rot)

        # Calculate ray intersections (northing/easting offsets relative to camera)
        ray_offsets = self.intersect_rays_ground(ray_rot, camera_alt)

        # Calculate pixel world coordinates
        return self.combine_camera_pos_ray_offsets(camera_lat, camera_long, ray_offsets)

    def pixel_ground_positions_from_json(self, imu_json_path: Path | str):
        imu_data = mpu.read_json(imu_json_path)
        expected_keys = ["time", "latitude", "longitude", "altitude", "yaw", "pitch", "roll"]
        if not all(x in imu_data for x in expected_keys):
            raise ValueError(f"IMU JSON file must correspond to dict with keys {expected_keys}")
        return self.pixel_ground_positions(
            imu_data["latitude"],
            imu_data["longitude"],
            imu_data["yaw"],
            imu_data["pitch"],
            imu_data["roll"],
            imu_data["altitude"],
        )
