## MassiPipe
MassiPipe is a data processing pipeline for hyperspectral data acquired with the Airborne Remote Sensing System from Resonon, using a [Pika L](https://resonon.com/Pika-L) hyperspectral camera and a [Flame-S-VIS-NIR](https://www.oceanoptics.com/blog/flame-series-general-purpose-spectrometers/) spectrometer with a cosine corrector for measuring downwelling irradiance.

The pipeline was (originally) developed to process large amounts of data acquired in the ["Mapping Algae and Seagrass using Spectral Imaging and Machine Learning" (MASSIMAL)](https://en.uit.no/project/massimal) project. 

Resonon provides a free version of their hyperspectral processing software [Spectronon](https://resonon.com/software) that can do many of the processes that are implemented in the pipeline. However, this typically requires a lot of manual configuration for each step in the processing chain. The pipeline allows all the steps to be automated.

The pipeline also has the following improvements / additional features compared to Spectronon:

* Wavelength calibration of irradiance spectra using Fraunhofer lines.
* Parsing and interpolating IMU data to match image line timestamps
* Sun/sky glint correction (useful for images taken over water)
* Simplified georeferencing procedure using affine transform (faster iteration when parameters are changed)
* Combining GeoTIFFs into single large "mosaic", with "pyramids" allowing fast image viewing at different resolutions. 

## Installation
MassiPipe uses a number of external libraries. Some of these are easily installed using pip, but GDAL (the [Geospatial Data Abstraction Library](https://gdal.org/en/latest/)) and [rasterio](https://rasterio.readthedocs.io/en/stable/index.html) (which builds on GDAL) can be more difficult to install. We recommend using conda for creating a virtual environment and installing GDAL and rasterio, and then installing the MassiPipe package with pip.

Create environment with GDAL and rasterio, installing from conda-forge channel (change "massipipe" environment name and python version to your preference):

    conda create -n massipipe -c conda-forge python=3.10 gdal rasterio

Download massipipe from the [massipipe GitHub repository](https://github.com/mh-skjelvareid/massipipe) (download as zip or use git clone). Navigate to the root folder of the repo and install using pip ("." indicates installation of package in current directory):

    conda activate massipipe
    pip install .

If you're a developer, you may want to install massipipe in "editable" mode instead, and also install optional dependencies that are used in development:
    
    pip install -e .[dev]

To register the virtual environment for use with Jupyter notebooks, run

    conda activate massipipe
    python -m ipykernel install --user --name=massipipe



## Quick start
The dataset shold be organized as follows:
``` { .text .no-copy }
    ├── 0_raw
    │   ├── <Raw data folder 1>
    │   │   ├── <ImageSetName>_downwelling_1_pre.spec
    │   │   ├── <ImageSetName>_downwelling_1_pre.spec.hdr
    │   │   ├── <ImageSetName>_Pika_L_1.bil
    │   │   ├── <ImageSetName>_Pika_L_1.bil.hdr
    │   │   ├── <ImageSetName>_Pika_L_1.bil.times
    │   │   └── <ImageSetName>_Pika_L_1.lcf
    │   ├── <Raw data folder 2>
    │   └── ...
    └── calibration
    │   ├── <downwelling_calibration_file>.dcp
    │   └── <radiance_calibration_file>.icp
    └── config.yaml
```

Note that the structure of the 0_raw folder is the structure created directly by the
Pika L camera. The config.yaml file is used to modify the processing of the specific
dataset. If no such config file is present, a template with defaults is created.

Create a pipeline processor for a dataset at path `dataset_dir`, without YAML config
file, using only default parameters:
``` { .python}
    from massipipe.pipeline import PipelineProcessor
    processor = PipelineProcessor(dataset_dir)
    processor.run()
```
A config file with name "seabee.config.yaml" is created in the root directory of
the dataset. 

It is possible to modify the YAML config file to change the input parameters and then
run the processing. See separate section on configuration file below.

Create a pipeline processor and run based on input parameters in seabee.config.yaml file :
``` { .python}
    from massipipe.pipeline import PipelineProcessor
    processor = PipelineProcessor(dataset_dir,config_file_name='seabee.config.yaml')
    processor.run()
```

After processing, the dataset has the following structure (files not shown), with processed files in folders `0b_quicklook`, `1a_radiance`, `1b_radiance_gc`, `2a_reflectance`, `2b_reflectance_gc`, `imudata`, and `mosaics`:
``` { .text .no-copy }
    ├── 0_raw
    │   ├── <Raw data folder 1>
    │   ├── <Raw data folder 2>
    │   └── ...
    ├── 0b_quicklook
    ├── 1a_radiance
    ├── 1b_radiance_gc
    │   └── rgb
    ├── 2a_reflectance
    ├── 2b_reflectance_gc
    │   └── rgb
    ├── calibration
    │   ├── downwelling_calibration_spectra
    │   └── radiance_calibration_frames
    ├── geotransform
    ├── imudata
    ├── logs
    └── mosaics
```

Note that if some data is missing (e.g. downwelling irradiance), some of the data
products will not be created. 

## Configuration file

The text block below shows an example YAML configuration file with default parameters. A
short explanation is given for each line. For more detailed explanations, see
documentation / docstrings for the individual methods used. 

Note that the first lines in the YAML file (above "massipipe options") are included to
be compatible with the [YAML format established by
SeaBee](https://seabee-no.github.io/documentation/data-upload.html#sec-config-file).
These lines are not directly used by Massipipe, and can be left as default values,
unless the dataset is used for publishing via SeaBee. 
```{.yaml}
grouping: grouping_name                 # E.g. project name and/or larger area
area: area_name                         # Name of specific location
datetime: '197001010000'                # 'yyyymmddHHMM' or 'yyyymmdd'. Note: Quotes required
nfiles: 1                               # [not used for hyperspectral?]
organisation: organization_name         # Responsible organisation
creator_name: creator_name              # [Optional]. Data collector/pilot
mosaic: false                           # [not used for hyperspectral?]
classify: false                         # [not used for hyperspectral?]
theme: Habitat                          # SeaBee "theme" ('Seabirds', 'Mammals' or 'Habitat')
spectrum_type: HSI                      # [Optional]. Sensor type ('RGB', 'MSI' or 'HSI')
massipipe_options:                      # Massimal-specific options
  general:                              # 
    rgb_wl: null                        # RGB rendering wavelength (nm). Default [640,550,460]
  quicklook:                            # Simple PNG image product
    create: true                        # Whether to create
    overwrite: false                    # Whether to overwrite existing
    percentiles: null                   # Percentiles for contrast stretching. Default [2,98]
  imu_data:                             # JSON files with IMU data for each image
    create: true                        # Whether to create
    overwrite: false                    # Whether to overwrite existing
  geotransform:                         # JSON with simple affine transform for georeferencing++
    create: true                        # Whether to create
    overwrite: true                     # Whether to overwrite existing
    camera_opening_angle_deg: 36.5      # Opening angle of camera ("field of view")
    pitch_offset_deg: 0.0               # Camera forward tilt relative to nadir
    roll_offset_deg: 0.0                # Camera rightward tilt relative to nadir
    altitude_offset_m: 0.0              # Offset to add to measured altitude
    utm_x_offset_m: 0.0                 # Easting offset, pos. if measured data east of true value
    utm_y_offset_m: 0.0                 # Northing offset, pos. if measured data north of true value
    assume_square_pixels: true          # Whether to estimate altitude to achieve square pixels
  radiance:                             # Calibrated radiance in microflicks
    create: true                        # Whether to create
    overwrite: false                    # Whether to overwrite existing
    set_saturated_pixels_to_zero: true  # Set all pixel values to zero if any bands are saturated
    add_irradiance_to_header: true      # Adding calibrated irradiance to header (if available)
  radiance_rgb:                         # RGB GeoTIFF based on radiance cube
    create: true                        # Whether to create
    overwrite: false                    # Whether to overwrite existing
  radiance_gc:                          # Glint corrected version of radiance
    create: true                        # Whether to create
    overwrite: false                    # Whether to overwrite existing
    smooth_spectra: false               # Whether to smooth with Savitzky–Golay filter
    subtract_dark_spec: false           # Whether to subtract estimated "dark spectrum"
    set_negative_values_to_zero: false  # Negative radiance values (after g.c.) set to zero
    reference_image_numbers: null       # Ref. images (indexed from 0) for glint corrector. E.g. [0,14]
    reference_image_ranges: null        # Image ranges for each ref.im. List of 4-element lists.
                                        # [start_y,end_y,start_x,end_x]
                                        # Example: [[0,500,0,900],[1300,1750,300,870]]
  radiance_gc_rgb:                      # RGB GeoTIFF based on radiance_gc 
    create: true                        # Whether to create
    overwrite: false                    # Whether to overwrite existing
  irradiance:                           # Calibrated downwelling irradiance (if available)
    create: true                        # Whether to create
    overwrite: false                    # Whether to overwrite existing
  reflectance:                          # Reflectance (calc. with Lambertian reflector assumption)
    create: true                        # Whether to create
    overwrite: false                    # Whether to overwrite existing
    wl_min: 400                         # Minimum wavelength included (nm)
    wl_max: 930                         # Maximum wavelength included (nm)
    conv_irrad_with_gauss: true         # Smooth measured irradiance
    fwhm_irrad_smooth: 3.5              # FWHM of irrad. smoothing gaussian kernel
    smooth_spectra: false               # Smooth output with Savitzky–Golay filter
    refl_from_mean_irrad: false         # Use mean dataset irradiance for refl. calc. 
  reflectance_gc:                       # Glint corrected reflectance
    create: true                        # Whether to create
    overwrite: false                    # Whether to overwrite existing
    smooth_spectra: true                # Smooth output with Savitzky–Golay filter
    method: from_rad_gc                 # "from_rad_gc" or "flat_spec"
                                        # "from_rad_gc" calulates reflectance from radiance_gc
                                        # "flat_spec" assumes flat glint spectrum
  reflectance_gc_rgb:                   # RGB GeoTIFF of relectance_gc
    create: true                        # Whether to create
    overwrite: false                    # Whether to overwrite existing
  mosaic:                               # Mosaic of RGB GeoTIFFs
    overview_factors: [2, 4, 8, 16, 32] # GeoTIFF overview factors ("pyramids")
    reflectance_gc_rgb:                 # Mosaic for reflectance_gc_rgb
      create: false                     # Whether to create
      overwrite: true                   # Whether to overwrite existing
```