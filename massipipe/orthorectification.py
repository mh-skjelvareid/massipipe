import numpy as np
import rasterio
from numpy.typing import NDArray
from pyproj import CRS, Proj
from pyresample.geometry import AreaDefinition, SwathDefinition
from pyresample.kd_tree import resample_nearest
from rasterio.plot import reshape_as_raster
from rasterio.profiles import DefaultGTiffProfile
from rasterio.transform import from_bounds
from scipy.spatial.transform import Rotation

import massipipe.utils as mpu


class CameraModel:
    def __init__(
        self,
        cross_track_fov: float,
        n_pix: int,
        R_to_imu_from_cam: NDArray | None = None,
        euler_to_imu_from_cam: NDArray | None = None,
    ):
        """
        Initialize the orthorectification parameters.

        Parameters
        ----------
        cross_track_fov_deg : float
            The cross-track field of view (angular extent) of the camera, in radians.
        n_pix : int
            The number of pixels per image line.
        R_to_imu_from_cam : NDArray, optional
            Direction Cosine Matrix (DCM) representing the rotation from the camera
            to the IMU. If provided, `euler_to_imu_from_cam` must not be provided.
        euler_to_imu_from_cam : NDArray, optional
            Euler angles (yaw, pitch, roll), in radians, representing the rotation
            from the camera to the IMU. If provided, `cam_to_imu_rot_dcm` must not
            be provided.

        Notes
        -----
        - If neither `cam_to_imu_rot_dcm` nor `cam_to_imu_rot_euler` are provided,
        an identity rotation is assumed (i.e., camera and IMU frames are aligned).

        Raises
        ------
        ValueError
            If both `cam_to_imu_rot_dcm` and `cam_to_imu_rot_euler` are provided.
        """

        self.opening_angle_deg = cross_track_fov
        self.n_pix = n_pix

        if (R_to_imu_from_cam is not None) and (euler_to_imu_from_cam is not None):
            raise ValueError("Provide rotation as angles or DCM, not both.")

        if R_to_imu_from_cam is not None:
            self.R_to_imu_from_cam = Rotation.from_matrix(R_to_imu_from_cam)
        elif euler_to_imu_from_cam is not None:
            yaw, pitch, roll = euler_to_imu_from_cam
            self.R_to_imu_from_cam = Rotation.from_euler("zyx", [yaw, pitch, roll])
        else:
            self.R_to_imu_from_cam = Rotation.from_euler("zyx", [0.0, 0.0, 0.0])  # No rotation

    @property
    def pixel_looking_angles(self) -> NDArray:
        """Calculate the looking angles for each pixel in the sensor.

        Returns
        -------
        numpy.ndarray
            A 1D array of angles (in radians) corresponding to each pixel
        """

        edge = np.tan(self.opening_angle_deg / 2)
        return np.arctan(np.linspace(-edge, edge, self.n_pix))

    def _ray_rotation_matrices(self, yaw: NDArray, pitch: NDArray, roll: NDArray) -> NDArray:
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
        R_to_cam_from_pixel = Rotation.from_euler("x", self.pixel_looking_angles)

        # Convert to matrices
        R_world_cam = R_to_world_from_cam.as_matrix()  # (M,3,3)
        R_cam_pixel = R_to_cam_from_pixel.as_matrix()  # (N,3,3)

        # Matrix multiplication (last two dims) via broadcasting
        return R_world_cam[:, np.newaxis, :, :] @ R_cam_pixel[np.newaxis, :, :, :]  # (M,N,3,3)

    def _camera_to_ground_vectors(
        self,
        R_to_world_from_pixel: NDArray,
        camera_altitude: NDArray,
    ) -> NDArray:
        """Calculate vectors from the camera to the ground for each pixel.

        Parameters
        ----------
        R_to_world_from_pixel : NDArray
            An array of shape (N, M, 3, 3) representing the rotation matrices for each pixel.
        camera_altitude : NDArray
            A 1D array of shape (M,) representing the altitude of the camera relative to the ground
            for each image row.

        Returns
        -------
        NDArray
            A 3D array of shape (M, N, 3) containing the direction vectors from the camera to the
            ground for eah pixel.

        """

        # Parameterization: t corresponds to number of unit vectors to reach ground
        with np.errstate(divide="ignore", invalid="ignore"):
            t = camera_altitude[:, np.newaxis] / R_to_world_from_pixel[:, :, 2, 2]  # (M,N)

        # Direction vector corresponds to last column of rotation matrix
        # (multiplication with unit vector along z-axis, [0,0,1])
        d_hat = R_to_world_from_pixel[:, :, 2, 0:2]  # (M,N,2)

        # Extend direction vector down to ground plane
        return d_hat * t[:, :, np.newaxis]

    def _combine_cam_pos_with_ray_vec(
        self, camera_northing: NDArray, camera_easting: NDArray, ray_vectors: NDArray
    ) -> NDArray:
        """Combine camera positions and ray offsets to compute pixel ground positions."""
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
        """Calculate pixel ground positions in UTM coordinates.

        Parameters
        ----------
        camera_northing : NDArray
            Camera northing positions, shape (M,).
        camera_easting : NDArray
            Camera easting positions, shape (M,).
        camera_altitude : NDArray
            Canera altitude positions, shape (M,).
        camera_yaw : NDArray
            Camera yaw angles in radians, zero at north, pi/2 at east. Shape (M,).
        camera_pitch : NDArray
            Camera pitch angles in radians, zero at horizon, positive looking up. Shape (M,).
        camera_roll : NDArray
            Camera roll angles in radians, zero at horizon, pos. for "right wing down". Shape (M,).

        Returns
        -------
        NDArray
            Pixel ground positions, shape (M, N, 2), where M is the number of
            camera positions and N is the number of pixels.

        Raises
        ------
        ValueError
            If input arrays do not have the same length.
        """
        # Verify that all inputs have same length
        M = len(camera_northing)
        if not all(
            len(x) == M
            for x in [camera_easting, camera_yaw, camera_pitch, camera_roll, camera_altitude]
        ):
            raise ValueError("All input arrays must have the same length")

        # Get camera rotations
        R_to_world_from_pixel = self._ray_rotation_matrices(camera_yaw, camera_pitch, camera_roll)

        # Calculate ray direction vectors
        ray_vecs = self._camera_to_ground_vectors(R_to_world_from_pixel, camera_altitude)

        # Calculate pixel world coordinates
        return self._combine_cam_pos_with_ray_vec(camera_northing, camera_easting, ray_vecs)


class Resampler:
    def __init__(self, radius_of_influence, nodata):
        self.radius = radius_of_influence
        self.nodata = nodata

    def _area_definition(self, pixel_utm_positions: NDArray, utm_epsg: int) -> AreaDefinition:
        """Create AreaDefinition based on pixel coordinates

        Parameters
        ----------
        pixel_utm_positions : NDArray
            UTM coordinates (easting, northing) for each pixel in the input image.
            Shape (n_lines, n_samples, 2).
        utm_epsg : int
            EPSG code for the UTM zone.

        Returns
        -------
        AreaDefinition
            Pyresample AreaDefinition object created based on outer x/y bounds of pixel
            positions. The resolution is estimated median spacing of x and y coordinates
            in the pixel positions.
        """
        gsd = estimate_resolution_from_pixel_coordinates(pixel_utm_positions)
        area_extent = area_extent_from_pixel_coordinates(pixel_utm_positions)
        x_min, y_min, x_max, y_max = area_extent
        width = int((x_max - x_min) / gsd)
        height = int((y_max - y_min) / gsd)

        area_def = AreaDefinition(
            area_id="utm_grid",
            description="UTM orthorectified area",
            proj_id="utm",
            projection=CRS.from_epsg(utm_epsg),
            width=width,
            height=height,
            area_extent=area_extent,
        )
        return area_def

    def _swath_definition(self, pixel_utm_coordinates: NDArray, utm_epsg: int) -> SwathDefinition:
        """Create swath definition based on spatial pixel coordinates for image

        Parameters
        ----------
        pixel_utm_coordinates : NDArray
            UTM coordinates (northing, easting) for each pixel in the input image.
            Shape (n_lines, n_samples, 2).
        utm_epsg : int
            EPSG code for UTM zone.

        Returns
        -------
        SwathDefinition
            pyresample SwathDefinition object, created based on utm pixel coordinates
            converted to longitude and latitude.
        """
        # Convert UTM coordinates to long/lat (note the order switch)
        proj = Proj(utm_epsg)
        pixel_lon, pixel_lat = proj(
            pixel_utm_coordinates[:, :, 1], pixel_utm_coordinates[:, :, 0], inverse=True
        )
        # Define swath (NOTE: MUST BE LONG/LAT)
        swath_def = SwathDefinition(
            lons=pixel_lon,
            lats=pixel_lat,
        )
        return swath_def

    def resample(self, image: NDArray, area_def, swath_def, gsd: float) -> NDArray:
        """Resample swath to grid using nearest neighbor resampling.

        Parameters
        ----------
        image : NDArray, shape (n_lines, n_samples, n_bands)
            Image to be resampled (original swath data).
        area_def : AreaDefinition
            The target area definition for the resampled image.
        swath_def : SwathDefinition
            The swath of the original image.
        gsd : float
            Ground sampling distance (in meters) of the resampled image.

        Returns
        -------
        NDArray, shape (n_northing, n_easting, n_bands)
            Resampled image.
        """

        radius = self.radius or 2 * gsd
        return np.array(
            resample_nearest(
                swath_def,
                image,
                area_def,
                radius_of_influence=radius,
                fill_value=self.nodata,
            )
        )


class GeoTiffWriter:
    def __init__(self, nodata):
        self.nodata = nodata

    def create_profile(
        self, image: NDArray, area_extent: tuple[float, float, float, float], epsg: int
    ) -> dict:
        h, w, b = image.shape
        tr = from_bounds(*area_extent, width=w, height=h)
        profile = dict(DefaultGTiffProfile())
        profile.update(
            {
                "height": h,
                "width": w,
                "count": b,
                "dtype": str(image.dtype),
                "crs": CRS.from_epsg(epsg),
                "transform": tr,
                "nodata": self.nodata,
            }
        )
        return profile

    def save(self, image: NDArray, wavelengths: NDArray, profile: dict, path: str):
        band_names = [f"{wl:.3f}" for wl in wavelengths]
        with rasterio.open(path, "w", **profile) as ds:
            for i, name in enumerate(band_names, 1):
                ds.set_band_description(i, name)
            ds.write(reshape_as_raster(image))


def estimate_resolution_from_pixel_coordinates(
    pixel_utm_coordinates: NDArray,
) -> float:
    """Estimate ground sampling distance (GSD) from pixel UTM coordinates.

    Parameters
    ----------
    pixel_utm_coordinates : np.ndarray
        Array of shape (n_rows, n_cols, 2) containing UTM coordinates (northing, easting)
        for each pixel.

    Returns
    -------
    gsd : float
        Estimated ground sampling distance (GSD) in meters.
    """
    # Calculate differences between adjacent pixels
    delta_x = np.diff(pixel_utm_coordinates[:, :, 1], axis=1)
    delta_y = np.diff(pixel_utm_coordinates[:, :, 0], axis=0)

    # Estimate GSD as the median of the absolute differences
    gsd_x = np.median(np.abs(delta_x))
    gsd_y = np.median(np.abs(delta_y))

    # Return the larger GSD rounded to the next significant digit
    return mpu.round_float_up(max(float(gsd_x), float(gsd_y)))


def area_extent_from_pixel_coordinates(pixel_utm_positions: NDArray) -> tuple:
    """Calculate area extent from pixel UTM coordinates.

    Parameters
    ----------
    pixel_utm_positions : NDArray
        UTM coordinates (northing, easting) for each pixel in the input image.
        Shape (n_lines, n_samples, 2).

    Returns
    -------
    tuple
        Area extent as (min_x, min_y, max_x, max_y).
    """
    utm_x_min = np.min(pixel_utm_positions[:, :, 1])
    utm_x_max = np.max(pixel_utm_positions[:, :, 1])
    utm_y_min = np.min(pixel_utm_positions[:, :, 0])
    utm_y_max = np.max(pixel_utm_positions[:, :, 0])
    return (utm_x_min, utm_y_min, utm_x_max, utm_y_max)
