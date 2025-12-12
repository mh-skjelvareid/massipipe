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
        R_to_imu_from_cam: NDArray | None = None,
        euler_to_imu_from_cam: NDArray | None = None,
        altitude_offset: float = 0,
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
        altitude_offset : float, optional
            Offset between the true altitude above ground and the altitude measured by
            the IMU, in meters.
                imu_altitude = true_altitude + altitude_offset
                true_altitude = imu_altitude - altitude_offset

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
        self.altitude_offset = altitude_offset

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
            A 1D array of shape (M,) representing the IMU measurements of altitude of the camera
            relative to the ground for each image row. The CameraModel's altitude offset is used to
            calculate true altitude as:
                true_altitude = camera_altitude - camera_model.altitude_offset

        Returns
        -------
        NDArray
            A 3D array of shape (M, N, 3) containing the direction vectors from the camera to the
            ground for eah pixel.

        """

        # Adjust camera altitude using altitude offset
        camera_altitude = camera_altitude - self.altitude_offset

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
        gsd = resolution_from_pixel_coordinates(pixel_utm_positions)
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

    def _create_profile(self, image: NDArray, transform: Affine, epsg: int) -> dict:
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


class FlatTerrainOrthorectifier_2:
    """Line-by-line orthorectification of hyperspectral images assuming flat terrain"""

    def __init__(
        self,
        camera_fov: float,
        camera_cross_track_n_pixels: int,
        imu_camera_rotation_dcm: NDArray | None = None,
        imu_camera_rotation_euler: NDArray | None = None,
        camera_altitude_offset: float = 0,
        radius_of_influence: float | None = None,
        nodata_fill_value: float = np.nan,
        ground_sampling_distance: float | None = None,
        estimate_yaw_from_positions: bool = False,
    ):
        """Initialize orthorectifier with constants

        Parameters
        ----------
        camera_fov : float,
            Opening angle (radians) of pushbroom camera
        camera_cross_track_n_pixels : int
            Number of spatial pixels in pushbroom camera
        imu_camera_rotation_dcm : NDArray | None, optional
            Direction Cosine Matrix (DCM) representing rotation from camera to IMU. If provided,
            `imu_camera_rotation_euler` must not also be provided.
        imu_camera_rotation_euler : NDArray | None, optional
            Euler angles (yaw, pitch, roll), in radians, representing rotation from camera to IMU.
            If provided, `imu_camera_rotation_dcm` must not also be provided.
        camera_altitude_offset : float, optional
            Offset (in meters) between the real altitude and that measured by the IMU.
            If the altitude above ground is larger than the IMU altitude, the offset is positive.
        radius_of_influence : float | None, optional
            Radius of neigbor pixels considered for nearest-neighbor resampling
            (meters). If None, radius of influence is set to twice the ground sampling distance.
        nodata_fill_value : _type_, optional
            Which value to use for "no data" (outside swath), by default np.nan
        ground_sampling_distance : float | None, optional
            Ground sampling distance (in meters) for the orthorectified image. If None, GSD is
            estimated from pixel coordinates.
        estimate_yaw_from_positions: bool
            Whether to estimate yaw from positions rather than using yaw measurements
            from IMU. Useful if yaw measurements are inaccurate.
        """

        self.camera_model = CameraModel(
            cross_track_fov=camera_fov,
            n_pix=camera_cross_track_n_pixels,
            R_to_imu_from_cam=imu_camera_rotation_dcm,
            euler_to_imu_from_cam=imu_camera_rotation_euler,
            altitude_offset=camera_altitude_offset,
        )
        self.resampler = Resampler(
            radius_of_influence=radius_of_influence, nodata=nodata_fill_value
        )
        self.file_writer = GeoTiffWriter(nodata=nodata_fill_value)
        self.gsd = ground_sampling_distance
        self.estimate_yaw_from_positions = estimate_yaw_from_positions

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

        # Calculate pixel ground positions
        pixel_utm_positions = self.camera_model.pixel_ground_positions(
            camera_northing=northing,
            camera_easting=easting,
            camera_altitude=altitude,
            camera_yaw=yaw,
            camera_pitch=pitch,
            camera_roll=roll,
        )

        # Create area and swath definitions
        area_def = self.resampler._area_definition(pixel_utm_positions, utm_epsg)
        swath_def = self.resampler._swath_definition(pixel_utm_positions, utm_epsg)

        # Resample image to grid
        gsd = self.gsd or resolution_from_pixel_coordinates(pixel_utm_positions)
        ortho_image = self.resampler.resample(image, area_def, swath_def, gsd=gsd)

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

    # Fit smoothing splines
    sn = make_smoothing_spline(time, northing)
    se = make_smoothing_spline(time, easting)

    # First derivatives (velocity components)
    dn_dt = sn.derivative()(time)
    de_dt = se.derivative()(time)

    # calculate heading
    return np.arctan2(de_dt, dn_dt)  # arctan (easting / northing)
