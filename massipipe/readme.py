# Imports
import logging
from pathlib import Path

# Get logger
logger = logging.getLogger(__name__)

TEXT = """# MASSIMAL hyperspectral image dataset

## Table of contents
1. [The MASSIMAL research project](#the-massimal-research-project)
2. [Hyperspectral imaging system](#hyperspectral-imaging-system)
3. [UAV platform for hyperspectral imaging](#uav-platform-for-hyperspectral-imaging)
4. [Field operations](#field-operations)

## The MASSIMAL research project 
This dataset was collected as part of the "MASSIMAL" project (Mapping of Algae
and Seagrass using Spectral Imaging and Machine Learning). The project was
conducted in the period 2020-2024, and data collection and field work was
performed at various locations along the Norwegian coast in the period 2021-2023. 

The project was financed by the Norwegian Research Council (8 MNOK) and by UiT
the Arctic University of Norway (600 kNOK), and was a collaboration between 

- UiT the Arctic University of Norway ("UiT")
- Norwegian Institute for Water Research ("NIVA")
- Nord University ("Nord")

The following researchers were the main contributors to the project:

- Martin Hansen Skjelvareid (principal investigator), UiT
- Katalin Blix, UiT
- Eli Rinde, NIVA
- Kasper Hancke, NIVA
- Maia Røst Kile, NIVA
- Galice Guillaume Hoarau, Nord

All UAV flights were piloted by Sigfinn Andersen at FlyLavt AS, Bodø.

Additional information about the project can be found on the following websites:
- [UiT project page](https://en.uit.no/project/massimal)
- [Cristin research database project page](https://app.cristin.no/projects/show.jsf?id=2054355)
- [Norwegian Research Council project
  page](https://prosjektbanken.forskningsradet.no/project/FORISS/301317)
- [SeaBee data portal with Massimal data](https://geonode.seabee.sigma2.no/catalogue/#/search?q=massimal&f=dataset)

## Hyperspectral imaging system
Hyperspectral imaging was performed using a "[Airborne Remote Sensing System](https://resonon.com/hyperspectral-airborne-remote-sensing-system)"
manufactured by Resonon. The system was configured with the following components:

### Hyperspectral camera
A [Pika-L](https://resonon.com/Pika-L) hyperspectral camera fitted with a lens with 8 mm
focal length and 36.5 degree field (see [lens
options](https://resonon.com/objective-lenses)) of view was used for all image
acquisitions.  The Pika-L is a pushbroom sensor; light entering through a narrow slit is
reflected by a diffraction grating on to a digital image sensor. The sensor has 900x600
pixels, and the diffraction grating splits the spectral components of the light, so that
the spectral range of 390-1030 nm spans the 600 pixels along the short axis of the
sensor. By default, pairs of spectral pixels are binned, resulting in 300 spectral
channels. In the specifications of the camera, Resonon lists that it has a spectral
range of 400-1000 nm and 281 spectral channels, indicating that the channels at the high
and low ends of the spectra are usually discarded. The 900 pixels along the long axis of
the sensor corresponds to 900 spatial pixels. Each image frame corresponds to imaging of
a single line on the ground, and to create a 2D image where each pixel has 300 spectral
channels, the camera has to be moved across the area of interest. The along-track
sampling distance on the ground is given by the speed of the camera multiplied by the
image acquisition time for each frame.

The spectral resolution (FWHM) has a mean value of approximately 2.7 nm. The table
below lists the FWHM for a selection of channels. The data is from Resonons general
design documents. Each individual camera may have slight deviations from this, but
Resonon claims that the data fits experiments well. Note the local minimum of 2.24 nm at
approx. 470 nm and the local maximum of 2.94 nm at approx. 745 nm.

| Wavelength (nm) | FWHM (nm) |
| :-------------: | :-------: |
|             400 |      2.97 |
|             415 |      2.55 |
|             430 |      2.36 |
|             445 |      2.27 |
|             460 |      2.24 |
|             475 |      2.24 |
|             490 |      2.26 |
|             505 |      2.29 |
|             520 |      2.33 |
|             535 |      2.39 |
|             550 |      2.44 |
|             565 |      2.50 |
|             580 |      2.57 |
|             595 |      2.63 |
|             610 |      2.68 |
|             625 |      2.74 |
|             640 |      2.79 |
|             655 |      2.83 |
|             670 |      2.86 |
|             685 |      2.89 |
|             700 |      2.92 |
|             715 |      2.93 |
|             730 |      2.94 |
|             745 |      2.94 |
|             760 |      2.94 |
|             775 |      2.93 |
|             790 |      2.91 |
|             805 |      2.89 |
|             820 |      2.87 |
|             835 |      2.84 |
|             850 |      2.80 |
|             865 |      2.77 |
|             880 |      2.73 |
|             895 |      2.69 |
|             910 |      2.65 |
|             925 |      2.61 |
|             940 |      2.57 |
|             955 |      2.53 |
|             970 |      2.50 |
|             985 |      2.47 |
|            1000 |      2.49 |

### On-board computer
A small on-board computer made by Resonon was used for controlling the camera, logging sensor data, and communicating with the ground station. The computer ran a Linux-based imaging firmware. 

### Inertial measurement unit (IMU)
An SBG Ellipse 2N inertial measurement unit was connected to the onboard computer. The IMU consists of 3 accelerometers and 3 gyroscopes to measure translational and angular accelerations of the camera, a GNSS receiver for measuring position and velocity, a barometric altimeter for aiding altitude measurement, and a magnetometer aiding in measurement of heading. The sensor data are combined in an extended Kalman filter to produce estimates of camera position (latitude, longitude, altitude) and orientation (pitch, roll, yaw). The specified accuracy for the GNSS receiver was 2.0 m CEP. Note that the GNSS receiver was not able to use real-time post-processing kinematic positioning (RTK/PPK). 

### Downwelling irradiance measurement
A Flame-S-VIS-NIR spectrometer with a CC-3-DA cosine collector manufactured by Ocean
Insight was used to measure downwelling spectral irradiance for each hyperspectral image
that was acquired. The spectrometer has a spectral range of 350-1000 nm, with optical
resolution of 1.33 nm and spectral sampling of 0.3-0.4 nm. The spectrometer was mounted
directly to one of the arms of the UAV [see image]. Ideally, the cosine collected should
be mounted pointing directly upwards, measuring the downwelling irradiance from a 180
degree hemispheric field of view. When mounted on the UAV, the orientation of the cosine
collector follows that of the UAV, which needs to roll and/or pitch to maneuver. This
necessarily affects the irradiance measurement to some degree, but the effects are
expected to be small as long as the movements are small. However, in some cases the
sensor movements may have large effects on the measurement, e.g. if the sun is very low.
The sensor was mounted with a slight backwards tilt, to compensate for the forwards tilt
of the UAV during imaging.  

Note that for some datasets, downwelling irradiance was not measured due to a technical
failure (poor cable connection between spectrometer and on-board computer).

## UAV platform for hyperspectral imaging

### Multirotor UAV
The UAV was a [Matrice 600 Pro](https://www.dji.com/no/support/product/matrice600-pro)
manufactured by DJI, a hexacopter design with 6 propellers, each 21 inches long. The UAV
weighed 9.5 kg without payload, and approximately 14 kg with the full
payload. A typical flight lasted approximately 10 minutes, with a safety margin of at
least 30 % remaining battery capacity. Six TB47S batteries were used to power the UAV,
and three sets of batteries were used to enable multiple flights and battery changing in
the field. The arms and propellers of the UAV can be folded for transport. When fully
extended, the UAV has a "wingspan" and height of approximately 1.65 and 0.75 meters,
respectively. During transportation, the UAV was folded and transported in a case
measuring 0.8x0.7x0.7 meters.

### Gimbal
The hyperspectral camera and the onboard computer were mounted to a DJI [Ronin-MX
gimbal](https://www.dji.com/no/ronin-mx). The gimbal has 3 axes (pitch, roll and yaw)
and decoupled the movement of the UAV from that of the camera. The camera was kept
pointing directly downwards ("nadir") with the line of sight perpendicular to the
forward movement of the drone. For yawing movements, the gimbal was set to follow the
heading of the UAV, but with low angular accelerations. 

### Radio modem
For most missions, a lightweight radio modem (Digi XBee, 2.4 GHz) was used for
communication between the ground station and the on-board computer. 


## Field operations
### Ground station
All datasets were acquired with a UAV ground station close to the area of interest
(usually 500 meters or less), with clear visual line of sight to the UAV. In some cases
a ground station could be set up close to the road, and in some cases all equipment had
to be transported via a small boat.  

### Mission setup
Most imaging missions were performed by defining a target area of interest, creating a
[KML file](https://en.wikipedia.org/wiki/Keyhole_Markup_Language) with a polygon
describing the area, and uploading the KML file to the UAV on-board computer. The
Airborne Remote Sensing System senses when the UAV is inside the terget area, and
automatically starts and stops the recording accordingly. The KML file was also used in
flight planning software to create way points for flight lines. In 2021 and 2022, DJI
Pilot was used for flight planning, while in 2023,
[UgCS](https://www.sphengineering.com/flight-planning/ugcs) was used.

### Splitting flight lines into multiple images
The Airborne Remote Sensing System is set up so that if an image reaches the limit of
2000 lines, the image is saved, and additional data is recorded into a new image. The practical effect of this is that images along a single continuous flight line are
split into multiple images - typically 5-10 images per line. 

### Autoexposure
The Airborne Remote Sensing System includes an autoexposure feature. With this, the
camera gain and shutter is automatically adjusted, based on test images acquired by the
camera, to bring the distribution of values into a suitable part of the camera dynamic
range. Autoexposure was used on a per-image basis, meaning that gain and shutter were
re-calculated between each image.  

Note that using autoexposure occationally resulted in suboptimal gain and shutter values.
For example, if the UAV was above (dark) water at the time of autoexposure and then flew
over (bright) land before the autoexposure could be recalculated, parts of the image
became saturated, resulting in invalid pixels. 

## Massipipe data processing pipeline
The dataset has been processed by
[MassiPipe](https://github.com/mh-skjelvareid/massipipe), a data processing pipeline
developed as part of the MASSIMAL project. Developent of the pipeline has been based on
data from the Resonon Pika-L hyperspectral camera, but many elements of the pipeline (e.g.
reflectance calculation, sun glint correction) are general and can be applied to any
hyperspectral image.  

MassiPipe can be used to automatically generate additional image products, based on the
radiance images distributed in the dataset. See details in the description of dataset
contents below. 

## Dataset contents
The following is a general description which is valid for most of the datasets produced
by the MASSIMAL project. Some deviations from this description may exist for special
cases, usually caused by technical problems during field work.

The same base file name is used for multiple types of data (e.g. IMU data and images),
this enables identification of which files belong together. 


### Configuration file - config.seabee.yaml 
The configuration file contains metadata about the dataset, and the parameters for
processing the dataset using MassiPipe. The file name includes "seabee" because the
dataset was perpared for publication and interactive exploration through the SeaBee data
portal, run by [SeaBee](https://seabee.no/) (Norwegian Infrastructure for Drone-based
Research, Mapping and Monitoring in the Coastal Zone).

For explanation of the MassiPipe processing parameters, see the [MassiPipe
documentation](https://mh-skjelvareid.github.io/massipipe/). The configuration file can
be edited to re-run the processing with modified parameters, and/or produce additional
image products. 

### Quicklook
The "quicklook" images are color images saved in the PNG format, to be used for getting
a quick overview of the dataset. The data displayed as red, green and blue (RGB)
corresponds to slices of the hyperspectral radiance image extracted at the "RGB
wavelengths" set in the config file. Each color channel has also been individually
contrast stretched, using the 2nd and 98th percentiles of the original data to set the
lower and upper ends of the range of values displayed. Note that since the image
statistics change from image to image, identical objects or nature types may appear
different in different images. The images are also not georeferenced. It is not
recommended to use the quicklook images for any type of analysis - use the hyperspectral
data instead. 

### IMU data
IMU data is stored as JSON files with 7 fields:
- **time**: Time represented as a single floating-point value, describing the number of
  seconds passed since January 1st 1970 (["UNIX
  time"](https://en.wikipedia.org/wiki/Unix_time)) 
- **roll**: Camera roll measured in radians. Positive values correspond to
  "right wing up", or pointing the camera to the right side of the flight line. 
- **pitch**: Camera pitch measured in radians. 
- **yaw**: Camera heading, measured in radians. Zero at due north, $\pi/2$ at due east.
- **longitude**: Longitude in decimal degrees, positive for east longitude
- **latitude**: Latitude in decimal degrees, positive for northern hemisphere
- **altitude**: Altitude in meters relative to the WGS-84 ellipsiod.

All values have been interpolated to match the lines of the hyperspectral image. E.g., if the
image contains 2000 lines, each field has 2000 corresponding values. 

*Note that while roll, pitch, longitude and latitude are relatively accurate, yaw and
altitude are less so.* The magnetometer which the IMU used to measure yaw (heading) turned out to
have poor accuracy, despite repeated calibrations. Absolute altitude values also appear
to be accurate only to approximately +/- 10 m, probably due to limited accuracy when
estimating altitude from GPS / GNSS. However, relative altitude values for the same
flight (same dataset) are fairly consistent, probably because altitude estimation was
aided by a barometric pressure sensor.  


### Radiance hyperspectral images

### Downwelling irradiance

### Mosaic


 

"""


def write_readme(readme_file_path: Path):
    """Write standard readme text to file"""
    try:
        with open(readme_file_path, mode="w", encoding="utf-8") as readme_file:
            readme_file.write(TEXT)
    except Exception:
        logger.error(f"Error while writing readme text to {readme_file_path}", exc_info=True)
