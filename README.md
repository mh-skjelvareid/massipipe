[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14748766.svg)](https://doi.org/10.5281/zenodo.14748766)

# MassiPipe - a data processing pipeline for MASSIMAL hyperspectral images
Pipeline and library for processing hyperspectral images. The pipeline was developed as
part of the MASSIMAL project; "Mapping of Algae and Seagrass using Spectral Imaging and
MAchine Learning". The project acquired a large set of hyperspectral images of
shallow-water areas in Norway, which were used for mapping marine habitats. 

The pipeline was developed based on data from a Resonon airborne remote sensing system,
with a [Pika L](https://resonon.com/Pika-L) hyperspectral camera and a
[Flame-S-VIS-NIR](https://www.oceanoptics.com/blog/flame-series-general-purpose-spectrometers/)
spectrometer with a cosine corrector for measuring downwelling irradiance. Parts of the
pipeline are custom-built for this system. However, other parts can be used for any
hyperspectral image in the [ENVI file
format](https://www.nv5geospatialsoftware.com/docs/ENVIImageFiles.html). 


## Citation
If you use this software in your own work, please cite it as

- Title: "MassiPipe - a data processing pipeline for MASSIMAL hyperspectral images"
- Author: Martin Hansen Skjelvareid
- DOI: [10.5281/zenodo.14748766](https://doi.org/10.5281/zenodo.14748766)

## Installation
MassiPipe uses a number of external libraries. Some of these are easily installed using
pip, but GDAL (the [Geospatial Data Abstraction Library](https://gdal.org/en/latest/))
and [rasterio](https://rasterio.readthedocs.io/en/stable/index.html) (which builds on
GDAL) can be more difficult to install. We recommend using conda for creating a virtual
environment and installing GDAL and rasterio, and then installing the MassiPipe package
with pip.

Create environment with GDAL and rasterio, installing from conda-forge channel (change
"massipipe" environment name and python version to your preference):

    conda create -n massipipe -c conda-forge python=3.10 gdal rasterio

Download massipipe from the [massipipe GitHub
repository](https://github.com/mh-skjelvareid/massipipe) (download as zip or use git
clone). Navigate to the root folder of the repo and install using pip ("." indicates
installation of package in current directory):

    conda activate massipipe
    pip install .

If you're a developer, you may want to install massipipe in "editable" mode instead, and
also install optional dependencies that are used in development:
    
    pip install -e .[dev]

To register the virtual environment for use with Jupyter notebooks, run

    conda activate massipipe
    python -m ipykernel install --user --name=massipipe

## Documentation
[MassiPipe documentation](https://mh-skjelvareid.github.io/massipipe/)

## Quick start
### Processing published (radiance) data
Datasets which have already been processed by MassiPipe and have been exported/published
are organized as follows:
``` { .text .no-copy }
    ├── 1a_radiance
    │   ├── rgb
    |   |   ├── <DatasetName>_<ImageNumber>_radiance_rgb.tiff
    |   |   └── ...
    │   ├── <DatasetName>_<ImageNumber>_irradiance.spec
    │   ├── <DatasetName>_<ImageNumber>_irradiance.spec.hdr
    │   ├── <DatasetName>_<ImageNumber>_radiance.bip
    │   ├── <DatasetName>_<ImageNumber>_radiance.bip.hdr
    │   └── ...
    └── imudata
    │   ├── <DatasetName>_<ImageNumber>_imudata.json
    │   └── ...
    └── orthomosaic
    │   └── <DatasetName>_<ImageType>_rgb.tiff
    └── config.seabee.yaml
```

The config.seabee.yaml contains the parameters used to process the data before
publication. To create additional image products based on the dataset (e.g. reflectance
images), either modify the YAML file directly, or load the original parameters and
modify them before running the pipeline. 

The example below shows how a pipeline is used to process a published dataset at path
`dataset_dir`, using the "config" object to specify that reflectance images and glint
corrected radiance images should be created:
``` { .python}
    from massipipe import Pipeline
    pipeline = Pipeline(dataset_dir)
    pipeline.config.reflectance.create = True
    pipeline.config.radiance_gc.create = True
    pipeline.run()
```

After processing, the images are can be found in the folders "1b_radiance_gc" and
"2a_reflectance".

### Processing raw data
The pipeline can use raw images (not yet calibrated to spectral radiance units) combined
with a set of calibration files as the starting point for processing. Raw datasets
should be organized as follows:
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
    └── config.seabee.yaml
```

Note that the structure of the 0_raw folder is the structure created directly by the
Pika L camera. The config.seabee.yaml file is used to modify the processing of the
specific dataset. If no such config file is present, a template with defaults is created
when a Pipeline object is instantiated.

Use a pipeline to process a dataset at path `dataset_dir`, using only default
parameters:
``` { .python}
    from massipipe import Pipeline
    pipeline = Pipeline(dataset_dir)
    pipeline.run()
```
A config file with name "seabee.config.yaml" is created in the root directory of the
dataset. The pipeline runs and produces all default output image products ("quicklook"
images, calibrated irradiance spectra, calibrated radiance images, RGB mosaic)

After processing, the dataset has the following structure (files not shown), with
processed files in folders `1a_radiance`, `geotransform`, `imudata`, `mosaics`, and
`quicklook`. Logs from each processing are saved to the `logs` folder.
``` { .text .no-copy }
    ├── 0_raw
    │   ├── <Raw data folder 1>
    │   ├── <Raw data folder 2>
    │   └── ...
    ├── 1a_radiance
    │   ├── <DatasetName>_<ImageNumber>_irradiance.spec
    │   ├── <DatasetName>_<ImageNumber>_irradiance.spec.hdr
    │   ├── <DatasetName>_<ImageNumber>_radiance.bip
    │   ├── <DatasetName>_<ImageNumber>_radiance.bip.hdr
    │   └── ...
    ├── calibration
    │   ├── downwelling_calibration_spectra
    │   └── radiance_calibration_frames
    ├── geotransform
    ├── imudata
    ├── logs
    ├── mosaics
    ├── quicklook
    └── config.seabee.yaml
```

Note that if some data is missing (e.g. downwelling irradiance), some of the data
products will not be created. 



## The MASSIMAL research project 
This dataset was collected as part of the "MASSIMAL" project (Mapping of Algae and
Seagrass using Spectral Imaging and Machine Learning). The project was conducted in the
period 2020-2024, and data collection and field work was performed at various locations
along the Norwegian coast in the period 2021-2023. 

The project was financed by the Norwegian Research Council (8 MNOK) and by UiT the
Arctic University of Norway (600 kNOK), and was a collaboration between 

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
- [Cristin research database project
  page](https://app.cristin.no/projects/show.jsf?id=2054355)
- [Norwegian Research Council project
  page](https://prosjektbanken.forskningsradet.no/project/FORISS/301317)
- [SeaBee data portal with Massimal
  data](https://geonode.seabee.sigma2.no/catalogue/#/search?q=massimal&f=dataset)

## License
MassiPipe is distributed under the terms described in the LICENSE file. Please refer to
the LICENSE file for complete details about usage and distribution rights.


## Contributing
Contributions of all kinds are welcome! If you'd like to contribute to this project,
please fork the repository and submit a pull request with your proposed changes. Before
starting work on a significant change, consider opening an issue to discuss your ideas. 

Bug reports, feature suggestions, documentation improvements, and any other ideas for
improving MassiPipe are welcome. If you have a dataset collected using a different
hyperspectral camera than that used here (Resonon Pika L) and want to use MassiPipe to
process the data, get in touch - we are interested in example datasets for expanding the
pipeline to other hyperspcetral cameras. Thanks!



