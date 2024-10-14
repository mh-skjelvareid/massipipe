from datetime import datetime
from typing import List, Literal, Optional, Tuple, Union

import yaml
from pydantic import BaseModel, PositiveFloat, PositiveInt, field_validator


def parse_config(yaml_path):
    """Parse YAML config file, accepting only basic YAML tags"""
    with open(yaml_path, "r") as stream:
        config = yaml.safe_load(stream)
    return config


class MpGeneral(BaseModel):
    rgb_wl: Optional[tuple[int, int, int]] = None


class MpQuickLook(BaseModel):
    create: bool = True
    overwrite: bool = False
    percentiles: Optional[tuple[int, int]] = None


class MpImuData(BaseModel):
    create: bool = True
    overwrite: bool = False


class MpGeoTransform(BaseModel):
    create: bool = True
    overwrite: bool = True
    camera_opening_angle_deg: Optional[PositiveFloat] = None
    pitch_offset_deg: Optional[float] = None
    roll_offset_deg: Optional[float] = None
    altitude_offset_m: Optional[float] = None
    utm_x_offset_m: Optional[float] = None
    utm_y_offset_m: Optional[float] = None
    assume_square_pixels: bool = True


class MpRadiance(BaseModel):
    create: bool = True
    overwrite: bool = False
    set_saturated_pixels_to_zero: Optional[bool] = True
    add_map_info: Optional[bool] = True


class MpRadianceRgb(BaseModel):
    create: bool = True
    overwrite: bool = False


class MpRadianceGc(BaseModel):
    create: bool = True
    overwrite: bool = False
    smooth_spectra: bool = False
    subtract_dark_spec: bool = True
    reference_image_numbers: Optional[List[PositiveInt]] = None
    reference_image_ranges: Optional[
        List[Tuple[PositiveInt, PositiveInt, PositiveInt, PositiveInt]]
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
    wl_min: Optional[float] = None
    wl_max: Optional[float] = None
    conv_irrad_with_gauss: bool = True
    fwhm_irrad_smooth: Optional[float] = None
    smooth_spectra: bool = False
    add_map_info: bool = True
    refl_from_mean_irrad: bool = False


class MpReflectanceGc(BaseModel):
    create: bool = True
    overwrite: bool = False
    smooth_spectra: bool = False
    wl_min: Optional[float] = None
    wl_max: Optional[float] = None


class MpReflectanceGcRgb(BaseModel):
    create: bool = True
    overwrite: bool = False


class MpMosaicCreateOverwrite(BaseModel):
    create: bool = True
    overwrite: bool = True


class MpMosaic(BaseModel):
    overview_factors: Optional[Union[list[int], tuple[int]]] = None
    radiance_rgb: MpMosaicCreateOverwrite
    radiance_gc_rgb: MpMosaicCreateOverwrite
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
    datetime: datetime
    nfiles: PositiveInt | None
    organisation: str
    mosaic: bool = False
    classify: bool = False
    theme: str = "Habitat"
    spectrum_type: Literal["RGB", "MSI", "HSI"] = "HSI"
    creator_name: str
    massipipe_options: MassipipeOptions

    @field_validator("datetime", mode="before")
    def datetime_validate(cls, datetime_str: str):
        if len(datetime_str) == 8:  # Expect YYYYmmdd
            return datetime.strptime(datetime_str, "%Y%m%d")
        elif len(datetime_str) == 12:  # Expect YYYYmmddHHMM
            return datetime.strptime(datetime_str, "%Y%m%d%H%M")
        else:
            raise ValueError(f"Invalid datetime string, use YYYYmmdd or YYYYmmddHHMM")
