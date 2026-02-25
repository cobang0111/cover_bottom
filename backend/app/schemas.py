from pydantic import BaseModel
from typing import List, Literal, Optional

class Parameters(BaseModel):
    material: str
    pcb_length: float
    led_housing_gap: float
    led_spacing: float
    led_count: int
    led_watt: float
    led_length_x: float
    led_height_y: float
    led_depth_z: float

class Hotspot(BaseModel):
    no: int
    x: float
    y: float
    z: float
    temperature: float
    spec: float
    judge: Literal["OK", "NG"]

class ResultPayload(BaseModel):
    metadata: dict
    parameters: Parameters
    results: Optional[dict]