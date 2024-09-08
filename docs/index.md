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

Note that the structure of the 0_raw folder is the structure created directly by the Pika L camera. The config.yaml file will be used to modify the processing of the specific dataset. This is work in progress, and the details and syntax of the config file have not been established yet. Most processing steps do not depend on the config file, and will run even if an empty config file is used.

Create a pipeline processor for a dataset at path `dataset_dir`, and run all default processing steps:
``` { .python .no-copy }
    from massipipe.pipeline import PipelineProcessor
    processor = PipelineProcessor(dataset_dir,config_file_name='config.yaml')
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
    ├── imudata
    ├── logs
    └── mosaics
```

Note that if some data is missing (e.g. downwelling irradiance), some of the data products will not be created. 