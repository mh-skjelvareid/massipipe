from pathlib import Path

import numpy as np
import rasterio
from numpy.typing import NDArray
from pyproj import CRS, Proj
from pyresample.geometry import AreaDefinition, SwathDefinition
from pyresample.kd_tree import resample_nearest
from rasterio.plot import reshape_as_raster
from rasterio.profiles import DefaultGTiffProfile
from rasterio.transform import Affine, from_bounds
from scipy.interpolate import make_smoothing_spline
from scipy.spatial.transform import Rotation

import massipipe.utils as mpu


class CameraModel:
    def __init__(
        self,
        cross_track_fov: float,
        n_pix: int,
        R_imu_from_cam: NDArray | None = None,
        euler_imu_from_cam: NDArray | None = None,
        altitude_correction: float = 0,
    ):
        """
        Initialize the orthorectification parameters.

        Parameters
        ----------
        cross_track_fov : float
            The cross-track field of view (angular extent) of the camera, in radians.
        n_pix : int
            The number of pixels per image line.
        R_imu_from_cam : NDArray, optional
            Direction Cosine Matrix (DCM) representing the (intrinsic) rotation from
            the camera to the IMU. If provided, `euler_to_imu_from_cam` must not be provided.
        euler_imu_from_cam : NDArray, optional
            Euler angles (roll, pitch, yaw), in radians, representing the (intrinsic) rotation
            from the camera to the IMU. If provided, `cam_to_imu_rot_dcm` must not
            be provided.
        altitude_correction : float, optional
            Correction term added to altitude measurements to account for systematic offsets,
                true_altitude = imu_altitude + altitude_correction

        Notes
        -----
        - If neither `R_imu_from_cam` nor `euler_imu_from_cam` are provided,
        an identity rotation is assumed (i.e., camera and IMU frames are aligned).

        Raises
        ------
        ValueError
            If both `R_imu_from_cam` and `euler_imu_from_cam` are provided.
        """

        if cross_track_fov >= np.pi:
            raise ValueError("Cross-track FOV must be less than pi radians (180 degrees).")
        self.cross_track_fov = cross_track_fov
        self.n_pix = n_pix

        if (R_imu_from_cam is not None) and (euler_imu_from_cam is not None):
            raise ValueError("Provide rotation as angles or DCM, not both.")

        # Create rotation from camera to IMU
        # NOTE: "xyz" order corresponds to R_z(yaw) @ R_y(pitch) @ R_x(roll)
        if R_imu_from_cam is not None:
            self.R_imu_from_cam = Rotation.from_matrix(R_imu_from_cam)
        elif euler_imu_from_cam is not None:
            self.R_imu_from_cam = Rotation.from_euler("xyz", euler_imu_from_cam)
        else:
            self.R_imu_from_cam = Rotation.identity()  # No rotation
        self.altitude_correction = altitude_correction

    @property
    def looking_angles(self) -> NDArray:
        """Calculate the looking angles for each pixel in the sensor.

        Returns
        -------
        numpy.ndarray
            A 1D array of angles (in radians) corresponding to each pixel
        """

        edge = np.tan(self.cross_track_fov / 2)
        return np.arctan(np.linspace(-edge, edge, self.n_pix))

    def _ray_rotation_matrices(self, roll: NDArray, pitch: NDArray, yaw: NDArray) -> NDArray:
        """
        Compute the ray rotation matrices for a camera given its rotation and pixel looking angles.

        Parameters
        ----------
        roll : NDArray
            A 1D array of shape (M,) representing the roll angles (in radians) of the camera.
        pitch : NDArray
            A 1D array of shape (M,) representing the pitch angles (in radians) of the camera.
        yaw : NDArray
            A 1D array of shape (M,) representing the yaw angles (in radians) of the camera.

        Returns
        -------
        NDArray
            A NumPy array of shape (M, N, 3, 3) representing the ray rotation matrices,
            from the pixel / camera frame to the world frame.
            M is the number of camera rotations, and N is the number of pixel looking angles.
        """

        # Create rotation to camera frame from pixel
        R_cam_from_pixel = Rotation.from_euler("x", self.looking_angles)

        # Create rotation from IMU to world
        # NOTE: "xyz" order corresponds to R_z(yaw) @ R_y(pitch) @ R_x(roll)
        R_world_from_imu = Rotation.from_euler("xyz", np.column_stack([roll, pitch, yaw]))

        # Apply correction to misalignment between camera and IMU
        R_world_from_cam = R_world_from_imu * self.R_imu_from_cam

        # Convert to matrices
        R_world_from_cam = R_world_from_cam.as_matrix()  # (M,3,3)
        R_cam_from_pixel = R_cam_from_pixel.as_matrix()  # (N,3,3)

        # Combine rotations via broadcasting and matrix multiplication (last two dims)
        return (
            R_world_from_cam[:, np.newaxis, :, :] @ R_cam_from_pixel[np.newaxis, :, :, :]
        )  # (M,N,3,3)

    def _camera_to_ground_vectors(
        self,
        R_world_from_pixel: NDArray,
        camera_altitude: NDArray,
    ) -> NDArray:
        """Calculate vectors from the camera to the ground for each pixel.

        Parameters
        ----------
        R_world_from_pixel : NDArray
            An array of shape (N, M, 3, 3) representing the rotation matrices for each pixel.
        camera_altitude : NDArray
            A 1D array of shape (M,) representing the IMU measurements of altitude of the camera
            relative to the ground for each image row.

        Returns
        -------
        NDArray
            A 3D array of shape (M, N, 3) containing the direction vectors from the camera to the
            ground for eah pixel.

        """

        # Correct camera altitude by adding altitude correction
        camera_altitude = camera_altitude + self.altitude_correction

        # Parameterization: t corresponds to number of unit vectors to reach ground
        with np.errstate(divide="ignore", invalid="ignore"):
            t = camera_altitude[:, np.newaxis] / R_world_from_pixel[:, :, 2, 2]  # (M,N)

        # Direction vector corresponds to last column of rotation matrix
        # (multiplication with unit vector along z-axis, [0,0,1])
        d_hat = R_world_from_pixel[:, :, :, 2]  # (M,N,3)

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
        camera_roll: NDArray,
        camera_pitch: NDArray,
        camera_yaw: NDArray,
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
        camera_roll : NDArray
            Camera roll angles in radians, zero at horizon, pos. for "right wing down". Shape (M,).
        camera_pitch : NDArray
            Camera pitch angles in radians, zero at horizon, positive looking up. Shape (M,).
        camera_yaw : NDArray
            Camera yaw angles in radians, zero at north, pi/2 at east. Shape (M,).

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
            for x in [camera_easting, camera_roll, camera_pitch, camera_yaw, camera_altitude]
        ):
            raise ValueError("All input arrays must have the same length")

        # Get camera rotations
        R_world_from_pixel = self._ray_rotation_matrices(camera_roll, camera_pitch, camera_yaw)

        # Calculate ray direction vectors
        ray_vecs = self._camera_to_ground_vectors(R_world_from_pixel, camera_altitude)

        # Calculate pixel world coordinates
        return self._combine_cam_pos_with_ray_vec(camera_northing, camera_easting, ray_vecs)


class Resampler:
    def __init__(self, nodata: float | None):
        """Initialize resampler.

        Parameters
        ----------
        nodata : float
            Value to use for "no data" pixels in resampled image.
        """
        self.nodata = nodata

    def _area_definition(
        self, pixel_utm_positions: NDArray, utm_epsg: int, gsd: float
    ) -> AreaDefinition:
        """Create AreaDefinition based on pixel coordinates"""
        # Determine area extent and resolution
        area_extent = area_extent_from_pixel_coordinates(pixel_utm_positions)

        # Calculate width and height in pixels
        x_min, y_min, x_max, y_max = area_extent
        width = int((x_max - x_min) / gsd)
        height = int((y_max - y_min) / gsd)
        if width <= 0 or height <= 0:
            raise ValueError(
                "Incompatible GSD and pixel coordinates - "
                "calculated width or height is non-positive."
            )

        # Create area definition
        return AreaDefinition(
            area_id="utm_grid",
            description="UTM orthorectified area",
            proj_id="utm",
            projection=CRS.from_epsg(utm_epsg),
            width=width,
            height=height,
            area_extent=area_extent,
        )

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

    def resample(
        self,
        image: NDArray,
        pixel_utm_coordinates: NDArray,
        utm_epsg: int,
        ground_sampling_distance: float | None = None,
        radius_of_influence: float | None = None,
    ) -> tuple[NDArray, AreaDefinition]:
        """Resample swath to grid using nearest neighbor resampling.

        Parameters
        ----------
        image : NDArray, shape (n_lines, n_samples, n_bands)
            Image to be resampled (original swath data).
        pixel_utm_coordinates : NDArray
            UTM coordinates (northing, easting) for each pixel in the input image.
            Shape (n_lines, n_samples, 2).
        utm_epsg : int
            EPSG code for UTM zone.
        ground_sampling_distance : float | None, optional
            Ground sampling distance (in meters) of the resampled image. If None, GSD is
            estimated from pixel coordinates.
        radius_of_influence : float | None, optional
            Radius of neighbor pixels considered for nearest-neighbor resampling (meters). If None,
            radius of influence is set to 5 times the ground sampling distance.

        Returns
        -------
        tuple[NDArray, AreaDefinition]
            resampled_image : NDArray, shape (n_northing, n_easting, n_bands)
                Resampled image.
            area_def : AreaDefinition
                Pyresample AreaDefinition for the resampled grid.
        """
        gsd = ground_sampling_distance or resolution_from_pixel_coordinates(pixel_utm_coordinates)
        radius = radius_of_influence or 5 * gsd

        area_def = self._area_definition(pixel_utm_coordinates, utm_epsg, gsd)
        swath_def = self._swath_definition(pixel_utm_coordinates, utm_epsg)

        image_resampled = np.array(
            resample_nearest(
                swath_def,
                image,
                area_def,
                radius_of_influence=radius,
                fill_value=self.nodata,  # type: ignore
            )
        )

        return image_resampled, area_def


class GeoTiffWriter:
    def __init__(self, nodata):
        self.nodata = nodata

    def _create_profile(self, image: NDArray, transform: Affine, epsg: int) -> dict:
        """Create GeoTOFF profile for image array, using existing transform & CRS."""
        h, w, b = image.shape

        profile = dict(DefaultGTiffProfile())
        profile.update(
            {
                "height": h,
                "width": w,
                "count": b,
                "dtype": str(image.dtype),
                "crs": CRS.from_epsg(epsg),
                "transform": transform,
                "nodata": self.nodata,
            }
        )
        return profile

    def _save(self, image: NDArray, wavelengths: NDArray, profile: dict, path: str | Path):
        """Save array as GeoTIFF using a rasterio profile, labelling channels with wavelengths."""
        band_names = [f"{wl:.3f}" for wl in wavelengths]
        with rasterio.open(path, "w", **profile) as ds:
            for i, name in enumerate(band_names, 1):
                ds.set_band_description(i, name)
            ds.write(reshape_as_raster(image))

    def save_image(
        self, image: NDArray, wavelengths: NDArray, transform: Affine, epsg: int, path: str | Path
    ):
        """Save orthorectified image as GeoTIFF file.

        Parameters
        ----------
        image : NDArray
            Orthorectified image to be saved, shape (n_northing, n_easting, n_bands).
        wavelengths : NDArray
            Wavelengths corresponding to each band in the image, shape (n_bands,).
        profile : dict
            Rasterio profile for the GeoTIFF file.
        path : str
            Path to the output GeoTIFF file.
        """

        # Create GeoTIFF profile
        geotiff_profile = self._create_profile(image, transform, epsg)

        # Save orthorectified image as GeoTIFF
        self._save(image, wavelengths, geotiff_profile, path)


class FlatTerrainOrthorectifier:
    """Line-by-line orthorectification of hyperspectral images assuming flat terrain"""

    def __init__(
        self,
        camera_cross_track_fov: float,
        camera_cross_track_n_pixels: int,
        R_imu_from_camera_dcm: NDArray | None = None,
        euler_imu_from_camera: NDArray | None = None,
        camera_altitude_correction: float = 0,
        radius_of_influence: float | None = None,
        ground_sampling_distance: float | None = None,
        nodata_fill_value: float = np.nan,
        estimate_yaw_from_positions: bool = False,
        imu_roll_direction_is_right_wing_down: bool = True,
        image_columns_increase_with_positive_y: bool = True,
    ):
        """Initialize orthorectifier with constants

        Parameters
        ----------
        camera_cross_track_fov : float,
            Opening angle (radians) of pushbroom camera
        camera_cross_track_n_pixels : int
            Number of spatial pixels in pushbroom camera
        R_imu_from_camera_dcm : NDArray | None, optional
            Direction Cosine Matrix (DCM) representing rotation from camera to IMU. If provided,
            `imu_camera_rotation_euler` must not also be provided.
        euler_imu_from_camera : NDArray | None, optional
            Euler angles (roll, pitch, yaw), in radians, representing rotation from camera to IMU.
            If provided, `imu_camera_rotation_dcm` must not also be provided.
        camera_altitude_correction : float, optional
            Correction term added to altitude measurements to account for systematic offsets
            between the true altitude above ground and the altitude measured by the IMU.
        radius_of_influence : float | None, optional
            Radius of neigbor pixels considered for nearest-neighbor resampling
            (meters). If None, radius of influence is set to twice the ground sampling distance.
        ground_sampling_distance : float | None, optional
            Ground sampling distance (in meters) for the orthorectified image. If None, GSD is
            estimated from pixel coordinates.
        nodata_fill_value : float, optional
            Which value to use for "no data" (outside swath), by default np.nan
        estimate_yaw_from_positions: bool
            Whether to estimate yaw from positions rather than using yaw measurements
            from IMU. Useful if yaw measurements are inaccurate.
        imu_roll_direction_is_right_wing_down : bool, optional
            Whether roll angles are defined as positive for "right wing down".
            If False, roll angles are assumed to be positive for "right wing up".
        image_columns_increase_with_positive_y : bool, optional
            Whether image columns indices increase with increasing y-axis coordinates.
            If False, column indices are assumed to increase with decreasing y-axis coordinates.
            A NED coordinate system in which the y-axis points to the right relative to direction
            of travel is used.
        """

        # Create helper objects; camera model, resampler, file writer
        self.camera_model = CameraModel(
            cross_track_fov=camera_cross_track_fov,
            n_pix=camera_cross_track_n_pixels,
            R_imu_from_cam=R_imu_from_camera_dcm,
            euler_imu_from_cam=euler_imu_from_camera,
            altitude_correction=camera_altitude_correction,
        )
        self.resampler = Resampler(
            nodata=nodata_fill_value,
        )
        self.file_writer = GeoTiffWriter(nodata=nodata_fill_value)

        # Set other parameters
        self.radius_of_influence = radius_of_influence
        self.gsd = ground_sampling_distance
        self.estimate_yaw_from_positions = estimate_yaw_from_positions
        self.imu_roll_direction_is_right_wing_down = imu_roll_direction_is_right_wing_down
        self.image_columns_increase_with_positive_y = image_columns_increase_with_positive_y

    def orthorectify_image(
        self,
        image: NDArray,
        time: NDArray,
        latitude: NDArray,
        longitude: NDArray,
        altitude: NDArray,
        roll: NDArray,
        pitch: NDArray,
        yaw: NDArray | None = None,
    ) -> tuple[NDArray, AreaDefinition, int]:
        """Orthorectify hyperspectral image using IMU data

        Parameters
        ----------
        image : NDArray
            Hyperspectral image to be orthorectified, shape (n_lines, n_samples, n_bands).
        time : NDArray
            Time vector, shape (M,)
        latitude : NDArray
            Latitude in decimal degrees, shape (M,)
        longitude : NDArray
            Longitude in decimal degrees, shape (M,)
        altitude : NDArray
            Altitude above ground in meters, shape (M,)
        roll : NDArray
            Roll angles in radians, shape (M,)
        pitch : NDArray
            Pitch angles in radians, shape (M,)
        yaw : NDArray | None, optional
            Yaw angles in radians, shape (M,). If None, yaw is estimated from positions.

        Returns
        -------
        tuple[NDArray, AreaDefinition, int]
            ortho_image : NDArray
                Orthorectified image, shape (n_northing, n_easting, n_bands).
            area_def : AreaDefinition
                Pyresample AreaDefinition for the orthorectified image.
            utm_epsg : int
                EPSG code for UTM zone of orthorectified image.

        """
        # Estimate yaw from positions if specified
        easting, northing, utm_epsg = mpu.convert_long_lat_to_utm(longitude, latitude)

        if (yaw is None) or self.estimate_yaw_from_positions:
            yaw = heading_from_positions(time, northing, easting)

        # Adjust roll direction if specified
        if not self.imu_roll_direction_is_right_wing_down:
            roll = -roll

        # Adjust for image column direction if specified
        if not self.image_columns_increase_with_positive_y:
            image = np.flip(image, axis=1)

        # Calculate pixel ground positions
        pixel_utm_coordinates = self.camera_model.pixel_ground_positions(
            camera_northing=northing,
            camera_easting=easting,
            camera_altitude=altitude,
            camera_roll=roll,
            camera_pitch=pitch,
            camera_yaw=yaw,
        )

        # Resample to orthorectified grid
        ortho_image, area_def = self.resampler.resample(
            image,
            pixel_utm_coordinates,
            utm_epsg,
            ground_sampling_distance=self.gsd,
            radius_of_influence=self.radius_of_influence,
        )

        return ortho_image, area_def, utm_epsg

    def orthorectify_image_file(
        self,
        image_path: Path | str,
        imu_data_path: Path | str,
        geotiff_path: Path | str,
        rgb_only: bool = True,
    ) -> None:
        """Orthorectify hyperspectral image using IMU data and save result to file

        Parameters
        ----------
        image_path : Path | str
            Path to hyperspectral image header.
        imu_data_path : Path | str
            Path to IMU data file (JSON format).
        geotiff_path : Path | str
            Path to (output) GeoTIFF file.
        rgb_only : bool, optional
            Whether to only use RGB bands from hyperspectral image for orthorectification.
        """
        # Read image
        image, wl, _ = mpu.read_envi(image_path)
        if rgb_only:
            image, wl = mpu.rgb_subset_from_hsi(image, wl)

        # Read IMU data
        imu_data = mpu.read_json(imu_data_path)

        # Orthorectify image
        ortho_image, area_def, utm_epsg = self.orthorectify_image(
            image,
            time=np.array(imu_data["time"]),
            latitude=np.array(imu_data["latitude"]),
            longitude=np.array(imu_data["longitude"]),
            altitude=np.array(imu_data["altitude"]),
            roll=np.array(imu_data["roll"]),
            pitch=np.array(imu_data["pitch"]),
            yaw=np.array(imu_data["yaw"]),
        )
        transform = from_bounds(
            *area_def.area_extent, height=ortho_image.shape[0], width=ortho_image.shape[1]
        )

        # Save orthorectified image as GeoTIFF
        self.file_writer.save_image(ortho_image, wl, transform, utm_epsg, geotiff_path)


def resolution_from_pixel_coordinates(
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


def heading_from_positions(time: NDArray, northing: NDArray, easting: NDArray) -> NDArray:
    """Calculate heading from smoothed positions

    Parameters
    ----------
    time : NDArray
        Time vector, shape (M,)
    northing : NDArray
        Northing in meters, shape (M,)
    easting : NDArray
        Easting in meters, shape (M,)

    Returns
    -------
    NDArray
        Heading (yaw) vector, in radians, shape (M,)

    Raises
    ------
    ValueError
        If input vectors are not all equal length
    """

    M = len(time)
    if not all(len(x) == M for x in (northing, easting)):
        raise ValueError("All input vectors must be of equal length.")

    # Identify valid timestamps (strictly increasing)
    valid = np.ones(M, dtype=bool)
    valid[1:] = np.diff(time) > 0

    if valid.sum() < 2:
        raise ValueError("Not enough unique timestamps for spline fitting.")

    # Fit splines using only valid data
    sn = make_smoothing_spline(time[valid], northing[valid])
    se = make_smoothing_spline(time[valid], easting[valid])

    # Compute derivatives at valid timestamps
    dn_dt_valid = sn.derivative()(time[valid])
    de_dt_valid = se.derivative()(time[valid])

    heading_valid = np.arctan2(de_dt_valid, dn_dt_valid)

    # Linear interpolation for invalid timestamps (np.interp)
    heading = np.empty(M)
    heading[valid] = heading_valid

    if not np.all(valid):
        heading[~valid] = np.interp(
            time[~valid],
            time[valid],
            heading_valid,
        )

    return heading
