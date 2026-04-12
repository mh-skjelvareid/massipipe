# MassiPipe development roadmap

*Author: Martin Skjelvareid*  
*Last updated: 2026-04-12*  

## Purpose
This document is intended as an overview of long-term development goals for MassiPipe. Concrete feature request issues will be made based on this document and the developers' own priorities and time available. 

## Background: Massimal project
MassiPipe was originally developed for processing data from the [MASSIMAL research project](https://en.uit.no/project/massimal) and publishing the dataset on the [NIRD Research Data Archive](https://archive.sigma2.no/) and on the [SeaBee Geovisualization Portal](https://geonode.seabee.sigma2.no/).

The initial goal of the project was to process large amounts of data from a [Resonon system for UAV hyperspectral imaging](https://resonon.com/hyperspectral-airborne-remote-sensing-system). Important features for the initial version included

- Being open source and Python-based, making it easily accessible and free 
- Standardized file naming and folder structure
- Robust processing (gracefully handing errors)  
- Georeferencing (simple affine transform)
- Mosaicing of multiple images, with image pyramids

The project is now over, and the full dataset is available  [DOI:10.11582/2025.00041](https://doi.org/10.11582/2025.00041). Processed versions of hyperspectral, multispectral and RGB image datasets can be browsed in an interactive map (see [SeaBee datasets](https://geonode.seabee.sigma2.no/catalogue/#/search?q=massimal&f=dataset)). 

The project is now in a position to continue as a more general-purpose tool for airborne hyperspectral imaging. However, being able to generate higher-level image products from published Massimal data (e.g. converting radiance images to reflectance) remains a priority. 

The remainder of this document describes ideas for further development of MassiPipe. 

## Integrating full orthorectification
The original version of MassiPipe was based on a simplified georeferencing method which only applied an affine transform (scaling and rotation) to map the image from pixel coordinates to UTM coordinates. Code for full orthorectification (assuming flat terrain, see [FlatTerrainOrthorectifier](api.md#massipipe.orthorectification.FlatTerrainOrthorectifier)) has subsequently been developed, but has not yet been integrated into the pipeline. The new orthorectifier should replace the simplified one.

Additional improvements to orthorectification could include:  

- Lever arm correction (taking offset between IMU and camera into account)
- Support for any terrain topography (based on elevation model)
- Support for using [quaternions](https://en.wikipedia.org/wiki/Quaternion) for describing rotation (not just [Euler angles](https://en.wikipedia.org/wiki/Euler_angles))


## Compatibility with additional imaging systems
MassiPipe currently supports only Resonon hyperspectral cameras, and the specific formats that Resonon uses for calibration files, IMU data, and downwelling irradiance measurements. The Resonon-specific parts are limited to 

- Converting raw hyperspectral images to radiance
- Converting raw spectra to downwelling irradiance
- Processing / interpolating IMU data to get camera position and orientation for each image line 

These operations could be bundled into a camera-specific part of the pipeline -- and the same could be done for other cameras. The pipeline should define a standardized data format for calbrated images (see NetCDF below), so that subsequent parts of the pipeline do not depend on camera specifics. 


## Using NetCDF / xarray / CF-compliance
MassiPipe is currently based on multiple file formats:

- ENVI files, for hyperspectral images and irradiance spectra. ENVI files consist of two files for each image; a binary data file and a text header file.
- JSON files, for IMU data
- (GeoTIFFs, for georeferenced images) 

Having multiple files for storing different parts of the same dataset is clunky and could lead to errors with missing data. 

Also, an important reason for calibrating hyperspectral images is to have proper radiometric units (e.g. watt per steradian per square metre per nanometre for [spectral radiance](https://en.wikipedia.org/wiki/Spectral_radiance)). However, the units are usually not encoded as part of ENVI files or GeoTIFFs, and have to be read from external documentation. 

[NetCDF](https://en.wikipedia.org/wiki/NetCDF) provides a more modern file format, where the values and axes of multidimensional arrays have standardized names and units are built in to the file ("self-describing"). A NetCDF file could be used to store both the hyperspectral image, the corresponding irradiance spectrum, and IMU data / geographic coordinates / CRS, all in one clearly labelled dataset. NetCDF files can be created and read in Python with [xarray](https://docs.xarray.dev/en/stable/getting-started-guide/why-xarray.html) (and other libraries as well).

NetCDF follows the so-called [CF conventions](https://en.wikipedia.org/wiki/Climate_and_Forecast_Metadata_Conventions) for Earth sciences data.

Using NetCDF as an output format from MassiPipe would make the processed data more self-contained and easier to use by others. However, GeoTIFFs may still be relevant for visualization and processing in GIS tools. 


## Atmospheric correction
The initial version of MassiPipe was made for UAV images aquired at approx. 50-60 m altitude, at which atmospheric effects are more or less negligible. However, for images aquired from manned aircraft, which fly at altitudes of hundreds or thousands of meters, the light reaching the camera is significantly affected by scattering and absorption in the air. 

[Atmospheric correction](https://en.wikipedia.org/wiki/Atmospheric_correction) methods attempt to counter these effects, so that the true reflectance of the ground can be measured, are called. MassiPipe could potentially include some such methods (via external libraries, see e.g. [Py6S](https://py6s.readthedocs.io/en/latest/index.html) and [acolite](https://github.com/acolite/acolite)) 
