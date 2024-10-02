from datetime import datetime
from typing import List, Literal, Optional, Tuple, Union

import yaml
from pydantic import BaseModel, PositiveFloat, PositiveInt


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
    smooth_spectra: Optional[bool] = False
    subtract_dark_spec: Optional[bool] = True
    reference_image_numbers: Optional[List[PositiveInt]] = None
    reference_image_ranges: Optional[
        List[Tuple[PositiveInt, PositiveInt, PositiveInt, PositiveInt]]
    ] = None


class MassipipeOptions(BaseModel):
    general: MpGeneral
    quicklook: MpQuickLook


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
