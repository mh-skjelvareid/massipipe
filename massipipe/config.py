"""Massipipe configuration module.

Provides functions for reading, writing, and exporting YAML configuration files,
and defines Pydantic models for Massipipe processing options.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import yaml
from pydantic import (
    BaseModel,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    field_validator,
)


def read_config(yaml_path: Union[Path, str]) -> Any:
    """Parse YAML config file, accepting only basic YAML tags"""
    with open(yaml_path, "r") as stream:
        config = yaml.safe_load(stream)
    return config


def write_config(data: dict, yaml_path: Union[Path, str]) -> None:
    """Write config data formatted as dictionary to YAML file"""
    with open(yaml_path, "w") as yaml_file:
        yaml.safe_dump(data, yaml_file, default_flow_style=False, sort_keys=False)


#### DEFINE PYDANTIC CONFIG STRUCTURE ####


class MpGeneral(BaseModel):
    """General options for Massipipe processing"""

    rgb_wl: Optional[Tuple[PositiveFloat, PositiveFloat, PositiveFloat]] = (640.0, 550.0, 460.0)


class MpQuickLook(BaseModel):
    """Configuration for creating quicklook image"""

    create: bool = True
    overwrite: bool = False
    percentiles: Optional[Tuple[NonNegativeFloat, PositiveFloat]] = None


class MpImuData(BaseModel):
    """Configuration for processing IMU data"""

    create: bool = True
    overwrite: bool = False


class MpGeoTransform(BaseModel):
    """Configuration for creating geotransform from IMU data"""

    create: bool = True
    overwrite: bool = False
    camera_opening_angle_deg: PositiveFloat = 36.5
    pitch_offset_deg: float = 0.0
    roll_offset_deg: float = 0.0
    altitude_offset_m: float = 0.0
    utm_x_offset_m: float = 0.0
    utm_y_offset_m: float = 0.0
    assume_square_pixels: bool = True


class MpRadiance(BaseModel):
    """Configuration for converting raw data to radiance"""

    create: bool = True
    overwrite: bool = False
    set_saturated_pixels_to_zero: bool = True
    add_envi_mapinfo_to_header: bool = True
    add_irradiance_to_header: bool = True


class MpRadianceRgb(BaseModel):
    """Comfiguration for creating RGB image from radiance data"""

    create: bool = True
    overwrite: bool = False


class MpRadianceGc(BaseModel):
    """Configuration for converting radiance to glint corrected radiance"""

    create: bool = False
    overwrite: bool = False
    smooth_spectra: bool = True
    subtract_dark_spec: bool = False
    set_negative_values_to_zero: bool = False
    reference_image_numbers: Optional[List[NonNegativeInt]] = None
    reference_image_ranges: Optional[
        List[Tuple[NonNegativeInt, PositiveInt, NonNegativeInt, PositiveInt]]
    ] = None


class MpRadianceGcRgb(BaseModel):
    """Configuration for creating RGB image from glint corrected radiance data"""

    create: bool = False
    overwrite: bool = False


class MpIrradiance(BaseModel):
    """Configuration for converting raw spectrum to irradiance"""

    create: bool = True
    overwrite: bool = False


class MpReflectance(BaseModel):
    """Configuration for converting radiance to reflectance"""

    create: bool = False
    overwrite: bool = False
    wl_min: float = 400
    wl_max: float = 930
    conv_irrad_with_gauss: bool = True
    fwhm_irrad_smooth: float = 3.5
    smooth_spectra: bool = False
    refl_from_mean_irrad: bool = False


class MpReflectanceGc(BaseModel):
    """Configuration for creating glint corrected reflectance"""

    create: bool = False
    overwrite: bool = False
    smooth_spectra: bool = True
    method: Literal["from_rad_gc", "flat_spec"] = "from_rad_gc"


class MpReflectanceGcRgb(BaseModel):
    """Configuration for creating RGB image from glint corrected reflectance data"""

    create: bool = False
    overwrite: bool = False


class MpMosaicRadiance(BaseModel):
    """Configuration for creating radiance mosaic image"""

    create: bool = True
    overwrite: bool = False


class MpMosaicRadianceGc(BaseModel):
    """Configuration for creating glint corrected radiance mosaic image"""

    create: bool = False
    overwrite: bool = False


class MpMosaicReflectanceGc(BaseModel):
    """Configuration for creating glint corrected reflectance mosaic image"""

    create: bool = False
    overwrite: bool = False


class MpMosaic(BaseModel):
    """General configuration for creating mosaic images"""

    overview_factors: Sequence[PositiveInt] = [2, 4, 8, 16, 32]
    visualization_mosaic: Literal["radiance", "radiance_gc"] = "radiance"
    radiance_rgb: MpMosaicRadiance = MpMosaicRadiance()
    radiance_gc_rgb: MpMosaicRadianceGc = MpMosaicRadianceGc()
    reflectance_gc_rgb: MpMosaicReflectanceGc = MpMosaicReflectanceGc()


class MassipipeOptions(BaseModel):
    """Configuration for Massipipe processing"""

    general: MpGeneral = MpGeneral()
    quicklook: MpQuickLook = MpQuickLook()
    imu_data: MpImuData = MpImuData()
    geotransform: MpGeoTransform = MpGeoTransform()
    radiance: MpRadiance = MpRadiance()
    radiance_rgb: MpRadianceRgb = MpRadianceRgb()
    radiance_gc: MpRadianceGc = MpRadianceGc()
    radiance_gc_rgb: MpRadianceGcRgb = MpRadianceGcRgb()
    irradiance: MpIrradiance = MpIrradiance()
    reflectance: MpReflectance = MpReflectance()
    reflectance_gc: MpReflectanceGc = MpReflectanceGc()
    reflectance_gc_rgb: MpReflectanceGcRgb = MpReflectanceGcRgb()
    mosaic: MpMosaic = MpMosaic(
        radiance_rgb=MpMosaicRadiance(),
        radiance_gc_rgb=MpMosaicRadianceGc(),
        reflectance_gc_rgb=MpMosaicReflectanceGc(),
    )


class Config(BaseModel):
    """Top-level configuration including standard SeaBee fields"""

    grouping: str = "grouping_name"
    area: str = "area_name"
    datetime: str = "197001010000"
    nfiles: PositiveInt = 1
    organisation: str = "organization_name"
    project: str = "project_name"
    creator_name: str = "creator_name"
    mosaic: bool = False
    classify: bool = False
    theme: str = "Habitat"
    spectrum_type: Literal["RGB", "MSI", "HSI"] = "HSI"
    massipipe_options: MassipipeOptions = MassipipeOptions()

    # Validation of datetime string
    # Note that string is not converted to datetime object,
    # this is to keep original string for template generation.
    @field_validator("datetime", mode="before")
    @classmethod
    def datetime_validate(cls, datetime_str: str):
        if len(datetime_str) == 8:  # Expect YYYYmmdd
            try:
                _ = datetime.strptime(datetime_str, "%Y%m%d")
            except ValueError:
                raise
        elif len(datetime_str) == 12:  # Expect YYYYmmddHHMM
            try:
                _ = datetime.strptime(datetime_str, "%Y%m%d%H%M")
            except ValueError:
                raise
        else:
            raise ValueError(f"Invalid datetime string, use YYYYmmdd or YYYYmmddHHMM")
        return datetime_str


def get_config_template() -> Config:
    """Generate template configuration based on Pydantic schema"""
    template_config = Config()
    return template_config


def nested_config_to_dict(config: BaseModel) -> Dict[str, Any]:
    """Convert nested Pydantic configuration to nested dictionary (recursively)"""
    config_dict = dict(config)
    for key, value in config_dict.items():
        if isinstance(value, BaseModel):
            config_dict[key] = nested_config_to_dict(value)
    return config_dict


def export_template_yaml(yaml_path: Union[Path, str]) -> None:
    """Export YAML template based on Pydantic schema"""
    template_config = get_config_template()
    template_dict = nested_config_to_dict(template_config)
    write_config(template_dict, yaml_path)
