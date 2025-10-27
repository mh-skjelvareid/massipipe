"""
This is example code demonstrating orthorectification based on (x,y) coordinates for
each pixel, done purely with rasterio and in-memory (no need to create VRT file for
GDAL).

NOTE: This solution loops over bands, calculating the nearest-neighbor interpolation for
each band individually. Calling GDAL directly would avoid this, but mixing rasterio and
GDAL is a bit risky. Make sure to close datasets of one kind before opening others.
"""

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

# ---------------------------------------------------------------------
# 1. Load your ENVI hyperspectral cube and coordinate arrays
# ---------------------------------------------------------------------
src_path = "input_image.img"  # ENVI file (.hdr next to it)
utm_x = np.load("utm_x.npy")  # shape (lines, samples)
utm_y = np.load("utm_y.npy")

with rasterio.open(src_path) as src:
    data = src.read()  # shape (bands, lines, samples)
    src_profile = src.profile
    src_dtype = src.dtypes[0]
    src_count, src_height, src_width = src.count, src.height, src.width

# ---------------------------------------------------------------------
# 2. Define the target (regular) grid
# ---------------------------------------------------------------------
# Choose pixel size (in meters)
xres, yres = 1.0, 1.0

# Determine bounds from UTM coordinates
x_min, x_max = utm_x.min(), utm_x.max()
y_min, y_max = utm_y.min(), utm_y.max()

dst_width = int(np.ceil((x_max - x_min) / xres))
dst_height = int(np.ceil((y_max - y_min) / yres))

# Construct affine transform for output grid (north-up)
dst_transform = from_origin(x_min, y_max, xres, yres)

# Output CRS
dst_crs = "EPSG:32611"  # Adjust to your UTM zone!

print(f"Output grid: {dst_width}×{dst_height} @ {xres} m")

# ---------------------------------------------------------------------
# 3. Allocate output array
# ---------------------------------------------------------------------
rectified = np.empty((src_count, dst_height, dst_width), dtype=src_dtype)

# ---------------------------------------------------------------------
# 4. Perform geolocation-based reprojection
# ---------------------------------------------------------------------
for i in range(src_count):
    reproject(
        source=data[i],
        destination=rectified[i],
        src_crs=dst_crs,  # CRS of geolocation array
        src_geoloc_array=(utm_x, utm_y),  # per-pixel coords
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        dst_resolution=(xres, yres),
        resampling=Resampling.bilinear,
    )

print("Reprojection complete.")

# ---------------------------------------------------------------------
# 5. Save output to GeoTIFF
# ---------------------------------------------------------------------
out_path = "rectified_rasterio.tif"
profile = src_profile.copy()
profile.update(
    {
        "driver": "GTiff",
        "height": dst_height,
        "width": dst_width,
        "count": src_count,
        "transform": dst_transform,
        "crs": dst_crs,
        "dtype": src_dtype,
        "compress": "deflate",
    }
)

with rasterio.open(out_path, "w", **profile) as dst:
    dst.write(rectified)

print(f"✅ Georectified cube written to {out_path}")
