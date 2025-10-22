# Plan for implementing orthorectification
MassiPipe currently has simple orthorectification, using affine transforms, but no full
orthorectification that uses the IMU measurements of camera angles for each image line. 

Orthorectifying an image can be divided into two main operations:
1. Calculating the geolocation of each pixel on the ground
2. Resampling / interpolating the image to a new regular grid (e.g. equispaced UTM grid)

The first operation is fairly specific to the camera and IMU used. The second is more
general and can be implemented using available libraries. 

## Calculating pixel locations
- Calculate per-line camera center (pitch/roll) based on 
    - fixed pitch/roll offsets
    - and pitch/roll IMU measurements
- Calculate roll "fan" representing rays angles for every spatial pixel
- Calculate pixel coordinates (UTM) based on
    - per-line along-track and across-track unit vectors
    - per-line position (UTM) 
    - pixel angles (pitch vector / roll matrix)

## Resampling to regular grid

### Via GDAL
- Save pixel coordinates as two GeoTIFF layers, e.g. "utm_x.tif", "utm_y.tif"
- Create VRT file linking image file and coordinates
- Use GDAL warp to resample

Advantages:
- Fast, production-scale
- Using files seems like a useful way to log "state" along the way
- Already using rasterio / GDAL, no new dependencies

Disadvantages:
- Not implementable in "pure" rasterio (must "import gdal")
- Creating VTR file seems a bit clunky

### Via pyresample ++
- Define pixel coordinates via pyresample.geometry.SwathDefinition
- Define output image via pyresample.geometry.AreaDefinition
- Use pyresample.kd_tree to resample

Advantages:
- Clean, compact syntax 

Disadvantages:
- In-memory processing, limits image file size
- Requires new dependencies
 
