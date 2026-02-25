from typing import List, Literal, Tuple, Optional, Dict
from pydantic import BaseModel

class ViewParams(BaseModel):
    step_file: Optional[str] = None
    z_threshold: float = -45.2
    z_outline: float = -38.2
    width: int | None = None   # 고정 너비가 필요하면 지정(없으면 자동 비율)

class ViewResponse(BaseModel):
    image_base64: str
    cad_xlim: Tuple[float, float]
    cad_ylim: Tuple[float, float]
    pixel_width: int
    pixel_height: int
    view_id: str | None = None  # 추가

class PxRect(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

class AnalyzeRequest(BaseModel):
    cad_xlim: Tuple[float, float]
    cad_ylim: Tuple[float, float]
    pixel_width: int
    pixel_height: int
    required_rects: List[PxRect] = []
    forbidden_rects: List[PxRect] = []

class VizLinePx(BaseModel):
    id: str
    x1: float
    y1: float
    x2: float
    y2: float

class VizRectPx(BaseModel):
    id: str
    x1: float
    y1: float
    x2: float
    y2: float

class Overlays(BaseModel):
    non_zero_vrects: List[VizRectPx] = []
    fixed_zero_vrects: List[VizRectPx] = []
    add_unconditional_lines: List[VizLinePx] = []

class AnalyzeResponse(BaseModel):
    add_unconditional: List[List[str]]
    fixed_zero: List[str]
    non_zero: List[str]
    not_equal: List[List[str]] = [] 
    fixed_value: Dict[str, float] = {} 
    must_equal: List[List[str]] = []
    upper_bound: Dict[str, float] = {}
    forbidden_zones: List[Dict[str, float]] = []
    h1_p_components: Dict[str, float] | None = None
    overlays: Overlays | None = None

class OptimizeRequest(BaseModel):
    add_unconditional: List[List[str]] = []
    fixed_zero: List[str] = []
    non_zero: List[str] = []
    not_equal: List[List[str]] = []
    fixed_value: Dict[str, float] = {}
    must_equal: List[List[str]] = []
    upper_bound: Dict[str, float] = {}
    forbidden_zones: List[Dict[str, float]] = []
    h1_p_components: Optional[Dict[str, float]] = None
    # 선택: 빠른 테스트용 GA 옵션
    max_generations: Optional[int] = 100
    p_crossover: Optional[float] = 0.9
    p_mutation: Optional[float] = 0.2
    num_random_individuals: Optional[int] = 300
    early_stop_patience: Optional[int] = 10
    view_id: str | None = None

class OptimizeResponse(BaseModel):
    mask_base64: str
    min_fitness: float
    contour_base64: str | None = None
    overlay_base64: str | None = None
    combined_base64: str | None = None
    max_displacement: float | None = None


class H1pRequest(BaseModel):
    view_id: str
    box: PxRect

class H1pResponse(BaseModel):
    value: float
    matched_y: float
    base: float

class AnalyzeOccRequest(BaseModel):
    view_id: str
    required_rects: List[PxRect] = []
    forbidden_rects: List[PxRect] = []
    h1p_box: Optional[PxRect] = None
    rev_forming_box: Optional[PxRect] = None