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

## Quick start
The dataset shold be organized as follows:

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
        ├── <downwelling_calibration_file>.dcp
        └── <radiance_calibration_file>.icp


Create a pipeline processor for a dataset at path `dataset_dir`, and run all default processing steps:

    from massipipe.pipeline import PipelineProcessor
    processor = PipelineProcessor(dataset_dir)
    processor.run()

After processing, the dataset has the following structure (files not shown), with processed files in folders `1_radiance`, `2_reflectance`, `2b_reflectance_gc`, and `mosaics`:

    ├── 0_raw
    │   ├── OlbergholmenS1-5
    │   ├── OlbergholmenS1-7
    │   └── OlbergholmenS1-8
    ├── 1_radiance
    ├── 2a_reflectance
    ├── 2b_reflectance_gc
    │   └── rgb_geotiff
    ├── calibration
    │   ├── downwelling_calibration_spectra
    │   └── radiance_calibration_frames
    ├── logs
    └── mosaics
