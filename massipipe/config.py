from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional, Sequence, Tuple, Union

import yaml
from pydantic import BaseModel, NonNegativeInt, PositiveFloat, PositiveInt, field_validator


def read_config(yaml_path: Union[Path, str]):
    """Parse YAML config file, accepting only basic YAML tags"""
    with open(yaml_path, "r") as stream:
        config = yaml.safe_load(stream)
    return config


def write_config(data: dict, yaml_path: Union[Path, str]):
    """Write config data formatted as dictionary to YAML file"""
    with open(yaml_path, "w") as yaml_file:
        yaml.safe_dump(data, yaml_file, default_flow_style=False, sort_keys=False)


#### DEFINE PYDANTIC CONFIG STRUCTURE ####


class MpGeneral(BaseModel):
    rgb_wl: Optional[tuple[PositiveInt, PositiveInt, PositiveInt]] = None


class MpQuickLook(BaseModel):
    create: bool = True
    overwrite: bool = False
    percentiles: Optional[tuple[NonNegativeInt, PositiveInt]] = None


class MpImuData(BaseModel):
    create: bool = True
    overwrite: bool = False


class MpGeoTransform(BaseModel):
    create: bool = True
    overwrite: bool = True
    camera_opening_angle_deg: PositiveFloat = 36.5
    pitch_offset_deg: float = 0.0
    roll_offset_deg: float = 0.0
    altitude_offset_m: float = 0.0
    utm_x_offset_m: float = 0.0
    utm_y_offset_m: float = 0.0
    assume_square_pixels: bool = True


class MpRadiance(BaseModel):
    create: bool = True
    overwrite: bool = False
    set_saturated_pixels_to_zero: bool = True
    add_irradiance_to_header: bool = True


class MpRadianceRgb(BaseModel):
    create: bool = True
    overwrite: bool = False


class MpRadianceGc(BaseModel):
    create: bool = True
    overwrite: bool = False
    smooth_spectra: bool = False
    subtract_dark_spec: bool = False
    set_negative_values_to_zero: bool = False
    reference_image_numbers: Optional[List[NonNegativeInt]] = None
    reference_image_ranges: Optional[
        List[Tuple[NonNegativeInt, PositiveInt, NonNegativeInt, PositiveInt]]
    ] = None


class MpRadianceGcRgb(BaseModel):
    create: bool = True
    overwrite: bool = False


class MpIrradiance(BaseModel):
    create: bool = True
    overwrite: bool = False


class MpReflectance(BaseModel):
    create: bool = True
    overwrite: bool = False
    wl_min: float = 400
    wl_max: float = 930
    conv_irrad_with_gauss: bool = True
    fwhm_irrad_smooth: float = 3.5
    smooth_spectra: bool = False
    refl_from_mean_irrad: bool = False


class MpReflectanceGc(BaseModel):
    create: bool = True
    overwrite: bool = False
    smooth_spectra: bool = True
    method: Literal["from_rad_gc", "flat_spec"] = "from_rad_gc"


class MpReflectanceGcRgb(BaseModel):
    create: bool = True
    overwrite: bool = False


class MpMosaicCreateOverwrite(BaseModel):
    create: bool = False
    overwrite: bool = True


class MpMosaic(BaseModel):
    overview_factors: Sequence[PositiveInt] = [2, 4, 8, 16, 32]
    # radiance_rgb: MpMosaicCreateOverwrite # Not yet implemented
    # radiance_gc_rgb: MpMosaicCreateOverwrite
    # reflectance_rgb: MpMosaicCreateOverwrite # Not yet implemented
    reflectance_gc_rgb: MpMosaicCreateOverwrite


class MassipipeOptions(BaseModel):
    general: MpGeneral
    quicklook: MpQuickLook
    imu_data: MpImuData
    geotransform: MpGeoTransform
    radiance: MpRadiance
    radiance_rgb: MpRadianceRgb
    radiance_gc: MpRadianceGc
    radiance_gc_rgb: MpRadianceGcRgb
    irradiance: MpIrradiance
    reflectance: MpReflectance
    reflectance_gc: MpReflectanceGc
    reflectance_gc_rgb: MpReflectanceGcRgb
    mosaic: MpMosaic


class Config(BaseModel):
    grouping: str
    area: str
    datetime: str
    nfiles: PositiveInt
    organisation: str
    creator_name: str
    mosaic: bool = False
    classify: bool = False
    theme: str = "Habitat"
    spectrum_type: Literal["RGB", "MSI", "HSI"] = "HSI"
    massipipe_options: MassipipeOptions

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


def get_config_template():
    template_config = Config(
        grouping="grouping_name",
        area="area_name",
        datetime="197001010000",
        nfiles=1,
        organisation="organization_name",
        creator_name="creator_name",
        massipipe_options=MassipipeOptions(
            general=MpGeneral(),
            quicklook=MpQuickLook(),
            imu_data=MpImuData(),
            geotransform=MpGeoTransform(),
            radiance=MpRadiance(),
            radiance_rgb=MpRadianceRgb(),
            radiance_gc=MpRadianceGc(),
            radiance_gc_rgb=MpRadianceGcRgb(),
            irradiance=MpIrradiance(),
            reflectance=MpReflectance(),
            reflectance_gc=MpReflectanceGc(),
            reflectance_gc_rgb=MpReflectanceGcRgb(),
            mosaic=MpMosaic(
                # radiance_rgb=MpMosaicCreateOverwrite(),
                # radiance_gc_rgb=MpMosaicCreateOverwrite(),
                # reflectance_rgb=MpMosaicCreateOverwrite(),
                reflectance_gc_rgb=MpMosaicCreateOverwrite(),
            ),
        ),
    )
    return template_config


def nested_config_to_dict(config):
    """Convert nested Pydantic configuration to nested dictionary (recursively)"""
    config_dict = dict(config)
    for key, value in config_dict.items():
        if isinstance(value, BaseModel):
            config_dict[key] = nested_config_to_dict(value)
    return config_dict


def export_template_yaml(yaml_path: Union[Path, str]):
    """Export YAML template based on Pydantic schema"""
    template_config = get_config_template()
    template_dict = nested_config_to_dict(template_config)
    write_config(template_dict, yaml_path)


if __name__ == "__main__":
    pass
    # export_template_yaml("example_config.yaml")
