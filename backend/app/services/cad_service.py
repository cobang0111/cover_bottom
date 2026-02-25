import io
import base64
from typing import List, Tuple
from PIL import Image
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Rectangle

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import topods
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_WIRE, TopAbs_EDGE
from OCC.Core.BRep import BRep_Tool
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepTools import BRepTools_WireExplorer

# 기존 규칙 테이블 (필요 부분만 그대로 복사)
# 수평 규칙의 선분 정의
HORIZONTAL_RULES = [
    {'id': 'h1_1_down', 'start_point': (497.65, 163.06), 'end_point': (255.0, 163.06)},
    {'id': 'h1_2_down', 'start_point': (220.0, 163.06), 'end_point': (155.0, 163.06)},
    {'id': 'h1_3_down', 'start_point': (80.0, 163.06), 'end_point': (-80.0, 163.06)},
    {'id': 'h1_4_down', 'start_point': (-220.0, 163.06), 'end_point': (-155.0, 163.06)},
    {'id': 'h1_5_down', 'start_point': (-497.65, 163.06), 'end_point': (-255.0, 163.06)},
    {'id': 'h2_1_up', 'start_point': (497.65, 75.0), 'end_point': (255.0, 75.0)},
    {'id': 'h2_2_up', 'start_point': (220.0, 75.0), 'end_point': (155.0, 75.0)},
    {'id': 'h2_3_up', 'start_point': (80.0, 75.0), 'end_point': (-80.0, 75.0)},
    {'id': 'h2_4_up', 'start_point': (-220.0, 75.0), 'end_point': (-155.0, 75.0)},
    {'id': 'h2_5_up', 'start_point': (-497.65, 75.0), 'end_point': (-255.0, 75.0)},
    {'id': 'h2_1_down', 'start_point': (497.65, 15.0), 'end_point': (403.67, 15.0)},
    {'id': 'h2_2_down', 'start_point': (368.67, 15.0), 'end_point': (180.0, 15.0)},
    {'id': 'h2_3_down', 'start_point': (80.0, 15.0), 'end_point': (-80.0, 15.0)},
    {'id': 'h2_4_down', 'start_point': (-368.67, 15.0), 'end_point': (-180.0, 15.0)},
    {'id': 'h2_5_down', 'start_point': (-497.65, 15.0), 'end_point': (-403.67, 15.0)},
    {'id': 'h3_1_up', 'start_point': (497.65, -14.5), 'end_point': (403.67, -14.5)},
    {'id': 'h3_2_up', 'start_point': (368.67, -14.5), 'end_point': (180.0, -14.5)},
    {'id': 'h3_3_up', 'start_point': (80.0, -14.5), 'end_point': (-80.0, -14.5)},
    {'id': 'h3_4_up', 'start_point': (-368.67, -14.5), 'end_point': (-180.0, -14.5)},
    {'id': 'h3_5_up', 'start_point': (-497.65, -14.5), 'end_point': (-403.67, -14.5)},
    {'id': 'h3_1_down', 'start_point': (497.65, -70.5), 'end_point': (367.5, -70.5)},
    {'id': 'h3_2_down', 'start_point': (317.5, -70.5), 'end_point': (180.0, -70.5)},
    {'id': 'h3_3_down', 'start_point': (80.0, -70.5), 'end_point': (-80.0, -70.5)},
    {'id': 'h3_4_down', 'start_point': (-317.5, -70.5), 'end_point': (-180.0, -70.5)},
    {'id': 'h3_5_down', 'start_point': (-497.65, -70.5), 'end_point': (-367.5, -70.5)},
    {'id': 'h4_1_up', 'start_point': (497.65, -162.8), 'end_point': (367.5, -162.8)},
    {'id': 'h4_2_up', 'start_point': (317.5, -162.8), 'end_point': (180.0, -162.8)},
    {'id': 'h4_3_up', 'start_point': (80.0, -162.8), 'end_point': (-80.0, -162.8)},
    {'id': 'h4_4_up', 'start_point': (-317.5, -162.8), 'end_point': (-180.0, -162.8)},
    {'id': 'h4_5_up', 'start_point': (-497.65, -162.8), 'end_point': (-367.5, -162.8)},
    {'id': 'h4_1_down', 'start_point': (493.65, -220.96), 'end_point': (427.5, -220.96)},
    {'id': 'h4_2_down', 'start_point': (347.5, -220.96), 'end_point': (200.5, -220.96)},
    {'id': 'h4_3_down', 'start_point': (180, -251.96), 'end_point': (-180, -251.96)},
    {'id': 'h4_4_down', 'start_point': (-347.5, -220.96), 'end_point': (-200.5, -220.96)},
    {'id': 'h4_5_down', 'start_point': (-493.65, -220.96), 'end_point': (-427.5, -220.96)},
    {'id': 'h5_1', 'start_point': (567.95, -285.96), 'end_point': (493.65, -285.96)},
    {'id': 'h5_2', 'start_point': (427.5, -291.96), 'end_point': (347.5, -291.96)},
    {'id': 'h5_3', 'start_point': (160.0, -293.17), 'end_point': (98.0, -293.17)},
    {'id': 'h5_4', 'start_point': (-160.0, -293.17), 'end_point': (-98.0, -293.17)},
    {'id': 'h5_5', 'start_point': (-427.5, -291.96), 'end_point': (-347.5, -291.96)},
    {'id': 'h5_6', 'start_point': (-567.95, -285.96), 'end_point': (-493.65, -285.96)},
]

# 수직 규칙의 선분 정의
VERTICAL_RULES = [
    {'id': 'v1_1', 'x_range': (465, 570), 'y_range': (75, 163.06)},
    {'id': 'v2_1', 'x_range': (220, 464), 'y_range': (75, 163.06)},
    {'id': 'v3_1', 'x_range': (0, 219), 'y_range': (75, 163.06)},
    {'id': 'v4_1', 'x_range': (-219, -1), 'y_range': (75, 163.06)},
    {'id': 'v5_1', 'x_range': (-464, -220), 'y_range': (75, 163.06)},
    {'id': 'v6_1', 'x_range': (-570, -465), 'y_range': (75, 163.06)},
    {'id': 'v1_2', 'x_range': (465, 570), 'y_range': (-14.5, 15)},
    {'id': 'v2_2', 'x_range': (220, 464), 'y_range': (-14.5, 15)},
    {'id': 'v3_2', 'x_range': (0, 219), 'y_range': (-14.5, 15)},
    {'id': 'v4_2', 'x_range': (-219, -1), 'y_range': (-14.5, 15)},
    {'id': 'v5_2', 'x_range': (-464, -220), 'y_range': (-14.5, 15)},
    {'id': 'v6_2', 'x_range': (-570, -465), 'y_range': (-14.5, 15)},
    {'id': 'v1_3', 'x_range': (465, 570), 'y_range': (-162.8, -70.5)},
    {'id': 'v2_3', 'x_range': (220, 464), 'y_range': (-162.8, -70.5)},
    {'id': 'v3_3', 'x_range': (0, 219), 'y_range': (-162.8, -70.5)},
    {'id': 'v4_3', 'x_range': (-219, -1), 'y_range': (-162.8, -70.5)},
    {'id': 'v5_3', 'x_range': (-464, -220), 'y_range': (-162.8, -70.5)},
    {'id': 'v6_3', 'x_range': (-570, -465), 'y_range': (-162.8, -70.5)},
    {'id': 'v1_4', 'x_range': (465, 570), 'y_range': (-285.96, -220.96)},
    {'id': 'v2_4', 'x_range': (220, 464), 'y_range': (-291.96, -220.96)},
    {'id': 'v3_4', 'x_range': (0, 219), 'y_range': (-293.17, -251.96)},
    {'id': 'v4_4', 'x_range': (-219, -1), 'y_range': (-293.17, -251.96)},
    {'id': 'v5_4', 'x_range': (-464, -220), 'y_range': (-291.96, -220.96)},
    {'id': 'v6_4', 'x_range': (-570, -465), 'y_range': (-285.96, -220.96)},
]

def _read_step_and_render(step_file: str, z_threshold: float, z_outline: float, width_hint: int | None):
    reader = STEPControl_Reader()
    status = reader.ReadFile(step_file)
    if status != 1:
        raise RuntimeError(f"STEP 파일 읽기 실패: {step_file}")
    reader.TransferRoots()
    shape = reader.Shape()

    target_faces = []
    outline_faces = []
    tol = 1e-6

    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while face_explorer.More():
        face = topods.Face(face_explorer.Current())
        props = GProp_GProps()
        brepgprop.SurfaceProperties(face, props)
        zc = props.CentreOfMass().Z()
        if zc <= z_threshold:
            target_faces.append(face)
        if abs(zc - z_outline) < tol:
            outline_faces.append(face)
        face_explorer.Next()

    if not target_faces and not outline_faces:
        raise RuntimeError(f"면 정보 없음 (z_threshold={z_threshold}, z_outline={z_outline})")

    fig = plt.figure(figsize=(20, 10), dpi=150)
    ax = fig.add_subplot(111)
    ax.set_aspect('equal', 'box')
    ax.axis('off')
    ax.invert_xaxis()

    for face in target_faces:
        all_verts, all_codes = [], []
        wire_explorer = TopExp_Explorer(face, TopAbs_WIRE)
        while wire_explorer.More():
            wire = topods.Wire(wire_explorer.Current())
            we = BRepTools_WireExplorer(wire)
            verts = []
            while we.More():
                p = BRep_Tool.Pnt(we.CurrentVertex())
                verts.append((p.X(), p.Y()))
                we.Next()
            if verts:
                all_verts.extend(verts)
                all_codes.append(Path.MOVETO)
                all_codes.extend([Path.LINETO] * (len(verts) - 1))
            wire_explorer.Next()
        if all_verts:
            path = Path(all_verts, all_codes)
            patch = PathPatch(path, facecolor='black', edgecolor='none')
            ax.add_patch(patch)

    # 외곽 사각형
    all_outline_verts = []
    for face in outline_faces:
        wire_explorer = TopExp_Explorer(face, TopAbs_WIRE)
        while wire_explorer.More():
            wire = topods.Wire(wire_explorer.Current())
            we = BRepTools_WireExplorer(wire)
            while we.More():
                p = BRep_Tool.Pnt(we.CurrentVertex())
                all_outline_verts.append((p.X(), p.Y()))
                we.Next()
            wire_explorer.Next()

    if all_outline_verts:
        xs = [v[0] for v in all_outline_verts]
        ys = [v[1] for v in all_outline_verts]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        rect = Rectangle((min_x, min_y), max_x - min_x, max_y - min_y,
                         linewidth=1.5, edgecolor='black', facecolor='none', linestyle='-')
        ax.add_patch(rect)

    ax.autoscale_view()
    cad_xlim = ax.get_xlim()
    cad_ylim = ax.get_ylim()

    # 해상도 비율 조정
    if width_hint:
        # width_hint 기준으로 높이 맞춤
        fig_w = width_hint / 100  # dpi=100 가정 시. (dpi=150이면 조정 필요)
        fig.set_size_inches(fig_w, fig_w/2, forward=True)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    pil = Image.open(buf).convert("RGBA")
    w, h = pil.size
    raw = io.BytesIO()
    pil.save(raw, format='PNG')
    raw.seek(0)
    b64 = base64.b64encode(raw.read()).decode('ascii')
    return b64, cad_xlim, cad_ylim, w, h

def _px_to_cad(x_px: float, y_px: float, cad_xlim: Tuple[float, float], cad_ylim: Tuple[float, float], pw: int, ph: int):
    cx = cad_xlim[0] + (x_px / pw) * (cad_xlim[1] - cad_xlim[0])
    cy = cad_ylim[1] - (y_px / ph) * (cad_ylim[1] - cad_ylim[0])
    return cx, cy

def _cad_to_px(cx: float, cy: float, cad_xlim, cad_ylim, pw: int, ph: int):
    px = (cx - cad_xlim[0]) / (cad_xlim[1] - cad_xlim[0]) * pw
    py = (cad_ylim[1] - cy) / (cad_ylim[1] - cad_ylim[0]) * ph
    return px, py


def render_view(step_file: str, z_threshold: float, z_outline: float, width_hint: int | None):
    return _read_step_and_render(step_file, z_threshold, z_outline, width_hint)

def create_cad_background_and_get_bounds(step_path: str, save_path: str):
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.BRepBndLib import brepbndlib
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
    from OCC.Display.SimpleGui import init_display

    if not os.path.exists(step_path):
        raise FileNotFoundError(step_path)

    reader = STEPControl_Reader()
    if reader.ReadFile(step_path) != 1:
        raise RuntimeError(f"STEP 파일 읽기 실패: {step_path}")
    reader.TransferRoots()
    shape = reader.OneShape()

    bbox = Bnd_Box()
    brepbndlib.Add(shape, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    cad_extent = (xmin, xmax, ymin, ymax)

    display, _, _, _ = init_display(size=(1920, 1080), display_triedron=False)
    display.EraseAll()
    display.set_bg_gradient_color([255,255,255], [255,255,255])
    dark = Quantity_Color(0.03, 0.03, 0.03, Quantity_TOC_RGB)
    display.DisplayShape(shape, update=True, color=dark)
    display.View.SetProj(0, 0, -1)
    display.View.SetUp(0, 1, 0)
    display.FitAll()
    display.View.Dump(save_path)
    print(f"CAD 배경 이미지 저장 완료: {save_path}")
    return save_path, cad_extent


# 새로 추가 (파일 하단에 배치)
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.GeomAbs import GeomAbs_Line
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCC.Display.SimpleGui import init_display

def render_occ_and_extract(step_file: str, z_threshold: float, z_outline: float, width_hint: int | None = 1920, min_area: float = 1500.0):
    reader = STEPControl_Reader()
    if reader.ReadFile(step_file) != 1:
        raise RuntimeError(f"STEP 파일 읽기 실패: {step_file}")
    reader.TransferRoots()
    shape = reader.Shape()

    target_faces, outline_faces = [], []
    tol = 1e-6
    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while face_explorer.More():
        face = topods.Face(face_explorer.Current())
        props = GProp_GProps()
        brepgprop.SurfaceProperties(face, props)
        zc = props.CentreOfMass().Z()
        area = props.Mass()
        if (area > min_area) and (zc <= z_threshold):
            target_faces.append(face)
        if abs(zc - z_outline) < tol:
            outline_faces.append(face)
        face_explorer.Next()
    if not outline_faces:
        raise RuntimeError(f"외곽선(Z={z_outline})을 찾지 못했습니다.")

    # baseline(min_y) 계산
    all_outline_verts = []
    for face in outline_faces:
        wire_explorer = TopExp_Explorer(face, TopAbs_WIRE)
        while wire_explorer.More():
            wire = topods.Wire(wire_explorer.Current())
            we = BRepTools_WireExplorer(wire)
            while we.More():
                p = BRep_Tool.Pnt(we.CurrentVertex())
                all_outline_verts.append((p.X(), p.Y()))
                we.Next()
            wire_explorer.Next()
    ys = [v[1] for v in all_outline_verts]
    baseline = min(ys) if ys else 0.0

    # 전체 bbox → cad_extent
    bbox = Bnd_Box()
    brepbndlib.Add(shape, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    cad_extent = (xmin, xmax, ymin, ymax)

    # OCC 렌더 → 버퍼
    base_w = width_hint or 1920
    phys_w = xmax - xmin
    phys_h = ymax - ymin
    aspect = phys_h / (phys_w + 1e-6)
    base_h = max(1, int(base_w * aspect))

    display, _, _, _ = init_display(size=(base_w, base_h), display_triedron=False)
    display.EraseAll()
    display.set_bg_gradient_color([255, 255, 255], [255, 255, 255])
    dark = Quantity_Color(0.03, 0.03, 0.03, Quantity_TOC_RGB)
    display.DisplayShape(shape, update=True, color=dark)
    display.View.SetProj(0, 0, -1)
    display.View.SetUp(0, 1, 0)
    display.View.FitAll(0.0)

    buf = io.BytesIO()
    if hasattr(display.View, "DumpToBuffer"):
        display.View.DumpToBuffer(buf, base_w, base_h)
    else:
        tmp = "_temp_occ_dump.png"
        display.View.Dump(tmp)
        with open(tmp, "rb") as f:
            buf.write(f.read())
        os.remove(tmp)
    buf.seek(0)
    pil = Image.open(buf).convert("RGBA")
    pw, ph = pil.size
    raw = io.BytesIO(); pil.save(raw, format="PNG"); raw.seek(0)
    image_b64 = base64.b64encode(raw.read()).decode("ascii")

    # CAD 수평선 추출
    horizontal_lines = []
    tolerance = 1e-3
    for face in target_faces:
        edge_explorer = TopExp_Explorer(face, TopAbs_EDGE)
        while edge_explorer.More():
            edge = topods.Edge(edge_explorer.Current())
            curve_adaptor = BRepAdaptor_Curve(edge)
            if curve_adaptor.GetType() == GeomAbs_Line:
                p1 = curve_adaptor.Value(curve_adaptor.FirstParameter())
                p2 = curve_adaptor.Value(curve_adaptor.LastParameter())
                if abs(p1.Y() - p2.Y()) < tolerance and abs(p1.X() - p2.X()) > tolerance:
                    horizontal_lines.append(((p1.X(), p1.Y()), (p2.X(), p2.Y())))
            edge_explorer.Next()

    return {
        "image_b64": image_b64,
        "cad_extent": cad_extent,           # (xmin, xmax, ymin, ymax)
        "pixel_size": (pw, ph),
        "baseline": baseline,
        "horizontal_lines": horizontal_lines,
    }

# OCC 이미지 좌표 변환( stp2constraint.py와 동일하게 X/Y 뒤집힘 반영 )
def px_to_cad_occ(x_px: float, y_px: float, cad_extent, pw: int, ph: int):
    xmin, xmax, ymin, ymax = cad_extent
    cx = xmax - (x_px / pw) * (xmax - xmin)
    cy = ymax - (y_px / ph) * (ymax - ymin)
    return cx, cy

# 새 유틸(아무 서비스 모듈에 배치 가능)
def compute_h1_p_from_px_box(px_box: tuple[float,float,float,float], cad_extent, pixel_size, baseline: float, horizontal_lines):
    x1, y1, x2, y2 = px_box
    pw, ph = pixel_size
    xmin, xmax, ymin, ymax = cad_extent

    min_px, max_px = min(x1, x2), max(x1, x2)
    min_py, max_py = min(y1, y2), max(y1, y2)

    cad_x_start = xmax - (min_px / pw) * (xmax - xmin)
    cad_x_end   = xmax - (max_px / pw) * (xmax - xmin)
    cad_y_start = ymax - (max_py / ph) * (ymax - ymin)
    cad_y_end   = ymax - (min_py / ph) * (ymax - ymin)
    ux = tuple(sorted((cad_x_start, cad_x_end)))
    uy = tuple(sorted((cad_y_start, cad_y_end)))
    cx = (ux[0] + ux[1]) / 2.0

    cands = []
    for (p1, p2) in horizontal_lines:
        line_y = p1[1]
        lx, rx = sorted((p1[0], p2[0]))
        if (uy[0] <= line_y <= uy[1]) and (lx <= ux[1] and rx >= ux[0]):
            cands.append(((p1, p2), line_y))
    if not cands:
        return None  # 후보 없음

    max_y = max(y for _, y in cands)
    top = [c for c in cands if c[1] == max_y]
    if len(top) == 1:
        matched_y = top[0][1]
        seg = top[0][0]
    else:
        # 중심과 가장 가까운 선분 선택
        def midx(seg): return (seg[0][0] + seg[1][0]) / 2.0
        seg = min((t[0] for t in top), key=lambda s: (cx - midx(s))**2)
        matched_y = max_y

    h1p_value = abs(baseline) + matched_y
    return {
        "value": h1p_value,
        "matched_y": matched_y,
        "components": {"base": baseline, "y": matched_y}
    }

def compute_rev_forming_from_px_box(px_box: tuple[float,float,float,float], cad_extent, pixel_size, horizontal_lines):
    x1, y1, x2, y2 = px_box
    pw, ph = pixel_size
    xmin, xmax, ymin, ymax = cad_extent

    min_px, max_px = min(x1, x2), max(x1, x2)
    min_py, max_py = min(y1, y2), max(y1, y2)

    ux0 = xmax - (min_px / pw) * (xmax - xmin)
    ux1 = xmax - (max_px / pw) * (xmax - xmin)
    uy0 = ymax - (max_py / ph) * (ymax - ymin)
    uy1 = ymax - (min_py / ph) * (ymax - ymin)
    ux = tuple(sorted((ux0, ux1)))
    uy = tuple(sorted((uy0, uy1)))

    # 박스 내 수평선 후보
    cands = []
    for (p1, p2) in horizontal_lines:
        line_y = p1[1]
        lx, rx = sorted((p1[0], p2[0]))
        if (uy[0] <= line_y <= uy[1]) and (lx <= ux[1] and rx >= ux[0]):
            cands.append(line_y)

    if not cands:
        return None

    rev = len(cands) // 4
    unique_ys = sorted(list(set(cands)), reverse=True)

    forming_widths_samples = []
    for i in range(0, len(unique_ys) - 1, 2):
        y_upper = unique_ys[i]
        y_lower = unique_ys[i + 1]
        w = y_upper - y_lower
        if w > 0:
            forming_widths_samples.append(w)

    forming_width = sum(forming_widths_samples) / len(forming_widths_samples) if forming_widths_samples else None

    return {
        "rev_forming": rev,
        "forming_widths_samples": forming_widths_samples,
        "forming_widths": forming_width,
    }