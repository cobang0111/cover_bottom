import os
import json
import tkinter as tk
from PIL import Image, ImageTk
import io
from collections import defaultdict
import pandas as pd

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import topods
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_WIRE, TopAbs_EDGE
from OCC.Core.BRep import BRep_Tool
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepTools import BRepTools_WireExplorer
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.GeomAbs import GeomAbs_Line

from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCC.Display.SimpleGui import init_display

import matplotlib
matplotlib.use('Agg')

# ========= 형상인자 규칙 정의 =========

# 수평 규칙의 선분 정의
HORIZONTAL_RULES_LINES = [
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
VERTICAL_RULES_RECTS = [
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

# h5 전용 오프셋
H5_Y_OFFSETS = {
    'h5_1': (28, 80), 'h5_6': (28, 80), 'h5_2': (22, 80),
    'h5_5': (22, 80), 'h5_3': (15, 45), 'h5_4': (15, 45),
}

# ========================= 유틸 =========================
def _is_h_up_or_down(rid: str) -> bool:
    return rid.endswith('_up') or rid.endswith('_down')

def _h_base(rid: str):
    if _is_h_up_or_down(rid):
        return rid.rsplit('_', 1)[0]
    return None

class InteractiveCadSelectorApp:
    
    def __init__(self, root, step_file_path, z_threshold, z_outline, win_width, win_height, min_area=1500.0):
        self.root = root
        self.root.title("대화형 형상인자 분석기")
        
        self.drawing_mode = 'h1p_fix'  
        self.fix_h1p = False          
        self.h1p_box_drawn = False
        
        self.analysis_results = {}
        self.forbidden_zone_coords = []
        self.min_area = float(min_area)
        self.h1_p_fixed_value = None
        self.h1_p_matched_y = None 
        print("STEP 파일 처리 및 이미지 생성 중...")
        self.shape = None
        self.target_faces = []
        self.horizontal_cad_lines = []
        
        self.cad_extent = None 
        self.pil_image, self.min_y_baseline = self._generate_drawing_data(
            step_file_path, z_threshold, z_outline, self.min_area
        )

        if not self.pil_image:
            self.root.destroy()
            return
        
        self._extract_horizontal_lines()
        
        self.horizontal_rect_rules, self.vertical_rect_rules = self._build_rule_rects()

        print("이미지/좌표 준비 완료. GUI 시작.")
        print("\n>> [1단계] 'h1_p'를 고정할 영역에 박스를 그리고 Enter를 누르세요.")
        print(">> 'h1p'를 고정하지 않으려면 박스를 그리지 말고 Enter를 누르세요.")

        original_width, original_height = self.pil_image.size
        aspect_ratio = original_width / original_height
        new_width = win_width
        new_height = int(new_width / aspect_ratio)
        if new_height > win_height:
            new_height = win_height
            new_width = int(new_height * aspect_ratio)
        self.pil_image = self.pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(self.pil_image)
        self.pixel_width, self.pixel_height = self.pil_image.size

        self.canvas = tk.Canvas(root, width=self.pixel_width, height=self.pixel_height, cursor="cross")
        self.canvas.image = self.tk_image
        self.canvas.pack(fill="both", expand=True)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

        self.h1p_rects = []
        self.rev_forming_rects = []
        self.required_rects = []
        self.forbidden_rects = []
        self.current_rect_id = None
        self.start_x, self.start_y = None, None

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.root.bind('<Return>', self.on_enter_press)
        self.root.bind('<Escape>', self.clear_all)

    def _build_rule_rects(self):
        h1_p_rule = {'id': 'h1_p', 'y_range': (255, 270), 'x_range': (-570, 570)}
        rev_forming_rule_def = {'id': 'rev_forming', 'x_range': (-535, 535), 'y_range': (200.58, 225.58)}

        y_groups = defaultdict(list)
        h5_rules_lines = []

        h1_p_y_center = (h1_p_rule['y_range'][0] + h1_p_rule['y_range'][1]) / 2
        rev_forming_y_center = (rev_forming_rule_def['y_range'][0] + rev_forming_rule_def['y_range'][1]) / 2

        y_groups[h1_p_y_center].append(h1_p_rule)
        y_groups[rev_forming_y_center].append(rev_forming_rule_def)

        for rule in HORIZONTAL_RULES_LINES:
            if rule['id'].startswith('h5'):
                h5_rules_lines.append(rule) 
            else:
                y_coord = rule['start_point'][1]
                y_groups[y_coord].append(rule)

        sorted_y = sorted(y_groups.keys(), reverse=True)
        y_ranges = {}
        for i, y_current in enumerate(sorted_y):
            if i == 0:
                y_next = sorted_y[i + 1]
                y_lower = (y_current + y_next) / 2
                y_upper = h1_p_rule['y_range'][1]
            else:
                y_previous = sorted_y[i - 1]
                y_upper = (y_current + y_previous) / 2

            if i == len(sorted_y) - 1:
                y_previous = sorted_y[i - 1]
                mid_boundary = (y_current + y_previous) / 2
                delta = mid_boundary - y_current
                y_lower = y_current - delta
            elif i != 0:
                y_next = sorted_y[i + 1]
                y_lower = (y_current + y_next) / 2

            y_ranges[y_current] = (y_lower, y_upper)

        horizontal_rect_rules = []
        for y_coord, rules in y_groups.items():
            for rule in rules:
                rid = rule['id']
                if 'x_range' in rule:
                    x_range = rule['x_range']
                else:
                    p1 = rule['start_point']; p2 = rule['end_point']
                    x_center = (p1[0] + p2[0]) / 2
                    x_range = (x_center - 20, x_center + 20)
                horizontal_rect_rules.append({
                    'id': rid,
                    'x_range': x_range,
                    'y_range': y_ranges[y_coord]
                })

        for rule in h5_rules_lines:
            rid = rule['id']
            p1 = rule['start_point']; p2 = rule['end_point']
            x_center = (p1[0] + p2[0]) / 2
            y_offset = H5_Y_OFFSETS[rid]
            horizontal_rect_rules.append({
                'id': rid,
                'x_range': (x_center - 20, x_center + 20),
                'y_range': (self.min_y_baseline + y_offset[0], self.min_y_baseline + y_offset[1])
            })

        vertical_rect_rules = VERTICAL_RULES_RECTS.copy()
        return horizontal_rect_rules, vertical_rect_rules

    # ---------- CAD 렌더 + 기준선 추출 ----------
    def _generate_drawing_data(self, step_file_path, z_threshold, z_outline, min_area):
        try:
            reader = STEPControl_Reader()
            reader.ReadFile(step_file_path)
            reader.TransferRoots()
            self.shape = reader.Shape()

            target_faces, outline_faces = [], []
            tol = 1e-6

            face_explorer = TopExp_Explorer(self.shape, TopAbs_FACE)
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
                
            self.target_faces = target_faces 

            if not outline_faces:
                print(f"오류: Z={z_outline}에서 외곽선(baseline) 면을 찾을 수 없습니다.")
                return None, None

            min_y_baseline = 0.0
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
                ys = [v[1] for v in all_outline_verts]
                min_y_baseline = min(ys)
            
            bbox = Bnd_Box()
            brepbndlib.Add(self.shape, bbox)
            xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
            self.cad_extent = (xmin, xmax, ymin, ymax) 
            print(f"계산된 전체 좌표 범위 (Extent): X [{xmin:.2f}, {xmax:.2f}], Y [{ymin:.2f}, {ymax:.2f}]")
            
            physical_width = xmax - xmin
            physical_height = ymax - ymin
            base_pixel_width = 1920
            aspect_ratio = physical_height / (physical_width + 1e-6)
            target_pixel_height = int(base_pixel_width * aspect_ratio)

            if target_pixel_height <= 0 or base_pixel_width <= 0:
                base_pixel_width, target_pixel_height = 1920, 1080
            else:
                print(f"모델 비율에 맞춘 새 캔버스 크기: ({base_pixel_width}, {target_pixel_height})")

            display, _, _, _ = init_display(size=(base_pixel_width, target_pixel_height), display_triedron=False)
            display.EraseAll()
            display.set_bg_gradient_color([255,255,255], [255,255,255]) 
            custom_dark_gray = Quantity_Color(0.03, 0.03, 0.03, Quantity_TOC_RGB)
            display.DisplayShape(self.shape, update=True, color=custom_dark_gray)
            display.View.SetProj(0, 0, -1)
            display.View.SetUp(0, 1, 0)   
            display.View.FitAll(0.0)
            
            buf = io.BytesIO()
            if hasattr(display.View, "DumpToBuffer"):
                display.View.DumpToBuffer(buf, base_pixel_width, target_pixel_height)
            else:
                temp_img_path = "_temp_occ_dump.png"
                display.View.Dump(temp_img_path)
                with open(temp_img_path, 'rb') as f_img:
                    buf.write(f_img.read())
                os.remove(temp_img_path)

            buf.seek(0)
            pil_image = Image.open(buf)
            
            return pil_image, min_y_baseline

        except Exception as e:
            print(f"CAD 데이터 처리 중 오류 발생: {e}")
            return None, None
        
    def _extract_horizontal_lines(self):
        print("CAD 모델에서 수평 선분 추출 중...")
        horizontal_lines = []
        tolerance = 1e-3 
        
        for face in self.target_faces:
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
        
        self.horizontal_cad_lines = horizontal_lines
        print(f"총 {len(self.horizontal_cad_lines)}개의 수평 선분 추출 완료.")
    
    # --- h1_p 계산 헬퍼 ---
    def _find_h1_p_in_box(self, user_rect_hotspot):
        rx_min, rx_max = user_rect_hotspot['x']
        ry_min, ry_max = user_rect_hotspot['y']
        rule_center_x = (rx_min + rx_max) / 2
        
        print(f"'h1_p' 탐색 영역: X({rx_min:.2f}, {rx_max:.2f}), Y({ry_min:.2f}, {ry_max:.2f})")

        candidate_lines = []
        for p1, p2 in self.horizontal_cad_lines:
            line_y = p1[1]
            line_x_min, line_x_max = sorted((p1[0], p2[0]))
            
            if (ry_min <= line_y <= ry_max) and (line_x_min <= rx_max and line_x_max >= rx_min):
                candidate_lines.append({'line': (p1, p2), 'y': line_y})

        if not candidate_lines:
            print("[실패] 'h1_p' 탐색 영역 내에서 후보 선분을 찾지 못했습니다.")
            return None

        max_y = max(c['y'] for c in candidate_lines)
        top_candidates = [c for c in candidate_lines if c['y'] == max_y]
        
        best_match = None
        if len(top_candidates) == 1:
            best_match = top_candidates[0]
        else:
            for cand in top_candidates:
                p1, p2 = cand['line']
                cand['dist'] = (rule_center_x - (p1[0] + p2[0])/2)**2
            best_match = min(top_candidates, key=lambda x: x['dist'])

        if best_match:
            matched_y = best_match['y']
            calculated_value = abs(self.min_y_baseline) + matched_y
            print(f"[성공] 'h1_p' 값을 {calculated_value:.2f}로 추출했습니다. (Base: {self.min_y_baseline:.2f}, Y: {matched_y:.2f})")
            
            return (calculated_value, matched_y)
        
        print("[실패] 'h1_p' 후보 선분 중 최종 매칭에 실패했습니다.")
        return None

    # --- 픽셀 -> CAD 좌표 변환 헬퍼 ---
    def _pixel_to_cad_rect(self, px_coords):
        if not self.cad_extent:
            print("[오류] _pixel_to_cad_rect: cad_extent가 없습니다.")
            return None

        xmin, xmax, ymin, ymax = self.cad_extent
        
        px = px_coords
        min_px, max_px = min(px[0], px[2]), max(px[0], px[2])
        min_py, max_py = min(px[1], px[3]), max(px[1], px[3])

        cad_x_start = xmax - (min_px / self.pixel_width) * (xmax - xmin)
        cad_x_end   = xmax - (max_px / self.pixel_width) * (xmax - xmin)
        cad_y_start = ymax - (max_py / self.pixel_height) * (ymax - ymin)
        cad_y_end   = ymax - (min_py / self.pixel_height) * (ymax - ymin)
        
        return {'x': sorted((cad_x_start, cad_x_end)), 'y': sorted((cad_y_start, cad_y_end))}

    # --- forming_width 계산 헬퍼 ---
    def _find_forming_width_in_box(self, user_rect_hotspot):
        rx_min, rx_max = user_rect_hotspot['x']
        ry_min, ry_max = user_rect_hotspot['y']
        
        candidate_lines = []
        for p1, p2 in self.horizontal_cad_lines:
            line_y = p1[1]
            line_x_min, line_x_max = sorted((p1[0], p2[0]))
            
            if (ry_min <= line_y <= ry_max) and (line_x_min <= rx_max and line_x_max >= rx_min):
                candidate_lines.append({'line': (p1, p2), 'y': line_y})

        if len(candidate_lines) < 2:
            print(f"[forming_width] X({rx_min:.1f},{rx_max:.1f}), Y({ry_min:.1f},{ry_max:.1f}) 영역에서 선을 2개 미만으로 찾음. (개수: {len(candidate_lines)})")
            return None

        ys = [c['y'] for c in candidate_lines]
        max_y = max(ys)
        min_y = min(ys)
        width = max_y - min_y
        
        print(f"[forming_width] 영역 Y({ry_min:.1f},{ry_max:.1f})에서 폭 {width:.2f} 계산 (MaxY: {max_y:.2f}, MinY: {min_y:.2f})")
        return width
    
    # ========================= GUI 인터랙션 =========================
    def on_press(self, event):
        if self.drawing_mode == 'done': return
        self.start_x, self.start_y = event.x, event.y
        
        if self.drawing_mode == 'h1p_fix':
            outline_color = 'green'
        elif self.drawing_mode == 'rev_forming':
            outline_color = 'orange' 
        elif self.drawing_mode == 'required':
            outline_color = 'blue'  
        else:
            outline_color = 'red'   
            
        self.current_rect_id = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y,
            outline=outline_color, 
            width=3
        )

    def on_drag(self, event):
        if self.current_rect_id:
            self.canvas.coords(self.current_rect_id, self.start_x, self.start_y, event.x, event.y)

    def on_release(self, event):
        if self.current_rect_id:
            rect_data = {'id': self.current_rect_id, 'px_coords': (self.start_x, self.start_y, event.x, event.y)}
            
            if self.drawing_mode == 'h1p_fix':
                self.h1p_rects.append(rect_data)
            elif self.drawing_mode == 'rev_forming':
                self.rev_forming_rects.append(rect_data) 
            elif self.drawing_mode == 'required':
                self.required_rects.append(rect_data)
            elif self.drawing_mode == 'forbidden':
                self.forbidden_rects.append(rect_data)
            self.current_rect_id = None

    def clear_all(self, event=None):
        CONSTRAINTS_FILE = 'constraints.json'
        if os.path.exists(CONSTRAINTS_FILE):
            try:
                os.remove(CONSTRAINTS_FILE)
                print(f"\n[초기화] Esc 키 입력: 기존 '{CONSTRAINTS_FILE}' 파일을 삭제했습니다.")
            except OSError as e:
                print(f"\n[오류] '{CONSTRAINTS_FILE}' 파일 삭제 중 오류 발생: {e}")

        for rect_data in self.h1p_rects + self.rev_forming_rects + self.required_rects + self.forbidden_rects:
            self.canvas.delete(rect_data['id'])
        
        self.h1p_rects, self.rev_forming_rects, self.required_rects, self.forbidden_rects, self.analysis_results, self.forbidden_zone_coords = [], [], [], [], {}, []
        self.drawing_mode = 'h1p_fix' 
        self.fix_h1p = False      
        self.h1p_box_drawn = False
        self.forming_variable_results = {}
        
        print(">> 모든 박스를 삭제하고 초기 상태로 돌아갑니다.")
        print("\n>> [1단계] 'h1_p'를 추출할 *핫스팟 영역*에 박스(녹색)를 그리고 Enter를 누르세요.")
        print(">> 'h1_p'를 고정하지 않으려면 *박스를 그리지 말고* Enter를 누르세요.")

    def on_enter_press(self, event):
        
        if self.drawing_mode == 'h1p_fix':
            if self.h1p_rects:
                self.h1p_box_drawn = True
                rect_data = self.h1p_rects[0]
                if len(self.h1p_rects) > 1:
                    print(f"[알림] 'h1_p' 박스가 {len(self.h1p_rects)}개 그려졌으나, 첫 번째 박스만 사용합니다.")
                
                if not self.cad_extent:
                    print("[오류] cad_extent(좌표 지도)가 없습니다. h1_p 추출 실패.")
                    user_rect_hotspot = None
                else:
                    user_rect_hotspot = self._pixel_to_cad_rect(rect_data['px_coords'])

                if user_rect_hotspot:
                    result = self._find_h1_p_in_box(user_rect_hotspot)
                    if result is not None:
                        found_value, matched_y = result 
                        self.h1_p_fixed_value = found_value
                        self.h1_p_matched_y = matched_y 
                        self.fix_h1p = True
                        print(f"\n>> 'h1_p' 값을 {self.h1_p_fixed_value:.2f}로 추출하여 고정합니다.")
                    else:
                        print(f"\n>> 'h1_p' 박스를 그렸으나, 해당 영역에서 'h1_p' 선을 추출하지 못했습니다. 고정하지 않습니다.")
                        self.fix_h1p = False
                        self.h1_p_matched_y = None
                else:
                    self.fix_h1p = False
                    self.h1_p_matched_y = None
            else:
                self.h1p_box_drawn = False
                print(f"\n>> 'h1_p' 박스가 없습니다. 'h1_p' 값을 고정하지 않습니다.")
                self.fix_h1p = False
                self.h1_p_matched_y = None
            
            self.drawing_mode = 'rev_forming'
            print("\n>> [2단계] 'rev_forming' 박스(주황색)를 그리고 Enter를 누르세요.")

        # ==========================================================
        # --- [수정된 로직] ---
        # ==========================================================
        elif self.drawing_mode == 'rev_forming':
            print("\n>> [2단계] 'rev_forming' 박스 분석 시작...")
            forming_variable_results = {}
            forming_widths_samples = []
            rev_value = None
            final_forming_width = None

            if not self.rev_forming_rects:
                print(f">> 'rev_forming' 박스가 없어 계산을 건너뜁니다.")
            elif not self.cad_extent or not self.horizontal_cad_lines:
                print("[오류] CAD 좌표/수평선 데이터가 없습니다. 계산 실패.")
            else:
                if len(self.rev_forming_rects) > 1:
                    print(f"[알림] 'rev_forming' 박스가 {len(self.rev_forming_rects)}개 그려졌으나, 첫 번째 박스 하나만 사용합니다.")
                
                main_rect_data = self.rev_forming_rects[0]
                user_rect_hotspot = self._pixel_to_cad_rect(main_rect_data['px_coords'])
                rx_min, rx_max = user_rect_hotspot['x']
                ry_min, ry_max = user_rect_hotspot['y']
                print(f">> 탐색 영역: X({rx_min:.1f},{rx_max:.1f}), Y({ry_min:.1f},{ry_max:.1f})")

                candidate_lines = []
                for p1, p2 in self.horizontal_cad_lines:
                    line_y = p1[1]
                    line_x_min, line_x_max = sorted((p1[0], p2[0]))
                    if (ry_min <= line_y <= ry_max) and (line_x_min <= rx_max and line_x_max >= rx_min):
                        candidate_lines.append({'line': (p1, p2), 'y': line_y})
                
                print(f">> 박스 내에서 {len(candidate_lines)}개의 수평선(후보)을 찾았습니다.")

                if not candidate_lines:
                    print(">> 수평선이 없어 'rev_forming' 값을 계산할 수 없습니다.")
                else:
                    # 3. 'rev_forming' 계산 
                    rev_value = len(candidate_lines) // 4
                    print(f">> rev_forming은 {rev_value}입니다. (수평선 {len(candidate_lines)}개 / 4)")
                    
                    # 4. 'forming_widths' 계산 
                    unique_ys = sorted(list(set(c['y'] for c in candidate_lines)), reverse=True)
                    
                    for i in range(0, len(unique_ys) - 1, 2):
                        y_upper = unique_ys[i]
                        y_lower = unique_ys[i+1]
                        width = y_upper - y_lower
                        if width > 0:
                            forming_widths_samples.append(width)
                    
                    print(f">> forming_widths_samples: {[round(w, 2) for w in forming_widths_samples]}")
                    
                    if forming_widths_samples:
                        final_forming_width = sum(forming_widths_samples) / len(forming_widths_samples)
                        print(f">> forming_widths (평균): {final_forming_width:.2f}")
                    else:
                        print(f">> 유효한 forming_width 샘플이 없어 평균을 계산하지 못했습니다.")
            
            # 5. 최종 결과를 self에 저장
            forming_variable_results['rev_forming'] = rev_value
            forming_variable_results['forming_widths_samples'] = forming_widths_samples
            forming_variable_results['forming_widths'] = final_forming_width
            self.forming_variable_results = forming_variable_results
            
            self.drawing_mode = 'required'
            print("\n>> [3단계] '포밍설계 필수 구역'(파란색)을 그리고 Enter를 누르세요.")

        elif self.drawing_mode == 'required':
            self._analyze_and_store(self.required_rects, "포밍설계 필수 영역", 'required')
            self.drawing_mode = 'forbidden'
            print("\n>> [4단계] '포밍설계 불가 구역'(빨간색)을 그리고 Enter를 누르세요. (초기화: Esc)")
            
        elif self.drawing_mode == 'forbidden':
            self._analyze_and_store(self.forbidden_rects, "포밍설계 불가 영역", 'forbidden')
            self.drawing_mode = 'done'
            print("\n>> 모든 분석이 완료되었습니다. 결과를 'constraints.json' 파일로 저장합니다.")
            self._save_constraints_to_json()
            print(">> (초기화: Esc, 프로그램 종료: 창 닫기)")

    def _get_conditions(self, found_vertical, mode, all_found_v_for_fixed_value, found_horizontal=None, wide_found_h=None):
        conditions = {
            'add_unconditional': [], 'fixed_zero': [], 'non_zero': [],
            'not_equal': [], 'must_equal': [], 'fixed_value': {}, 'upper_bound': {}
        }
        
        if mode == 'required':

            if wide_found_h:
                bases = defaultdict(set)
                for rid in wide_found_h:
                    if _is_h_up_or_down(rid):
                        base, suffix = rid.rsplit('_', 1)
                        bases[base].add(suffix)
                for base, suffixes in bases.items():
                    if len(suffixes) == 2:
                        conditions['not_equal'].append([f"{base}_up", f"{base}_down"])
                
                exception_pairs_for_fixed_zero = {
                    ('h2_1_down', 'h3_1_up'),
                    ('h2_2_down', 'h3_2_up'),
                    ('h2_3_down', 'h3_3_up'),
                    ('h2_4_down', 'h3_4_up'),
                    ('h2_5_down', 'h3_5_up'),
                }

                parsed_rules = {}
                for rid in wide_found_h:
                    if not rid.startswith('h') or not _is_h_up_or_down(rid): continue
                    try:
                        parts_str, suffix = rid.replace('h', '').rsplit('_', 1)
                        parts = parts_str.split('_')
                        if len(parts) == 2:
                            a, b = int(parts[0]), int(parts[1])
                            parsed_rules[rid] = {'a': a, 'b': b, 'suffix': suffix}
                    except (ValueError, IndexError): continue
                up_rules = {rid: data for rid, data in parsed_rules.items() if data['suffix'] == 'up'}
                down_rules_lookup = {(data['a'], data['b']): rid for rid, data in parsed_rules.items() if data['suffix'] == 'down'}
                
                for up_id, up_data in up_rules.items():
                    target_key = (up_data['a'] - 1, up_data['b'])
                    if target_key in down_rules_lookup:
                        down_id = down_rules_lookup[target_key]
                        current_pair = tuple(sorted([down_id, up_id]))

                        if current_pair in exception_pairs_for_fixed_zero:
                            print(f"[조건 생성] '가로형 필수 박스' (예외) 쌍 검출: {current_pair} -> 'fixed_zero'에 추가")
                            conditions['fixed_zero'].append(down_id)
                            conditions['fixed_zero'].append(up_id)
                        else:
                            print(f"[조건 생성] '가로형 필수 박스' (일반) 쌍 검출: {current_pair} -> 'must_equal'에 추가")
                            conditions['must_equal'].append(list(current_pair))
                            if up_data['a'] == 2 and up_data['suffix'] == 'up':
                                target_h2_down = f"h2_{up_data['b']}_down"
                                conditions['upper_bound'][target_h2_down] = 70.0
        
        elif mode == 'forbidden':
            conditions['fixed_zero'].extend(sorted(found_vertical))
            processed = set()
            for rid in found_horizontal:
                base = _h_base(rid)
                if base and base not in processed:
                    conditions['must_equal'].append([f"{base}_up", f"{base}_down"])
                    processed.add(base)
        
            v_rules_for_fixed_value = {'v5_3', 'v5_4', 'v6_3', 'v6_4', 'v1_3', 'v1_4'}
            for v_id in all_found_v_for_fixed_value:
                if v_id in v_rules_for_fixed_value:
                    param_p = f"{v_id}_p"
                    conditions['fixed_value'][param_p] = 567
                    print(f"[조건 추가] '{v_id}' (불가영역) 검출 -> '{param_p}'는 567로 고정")
                    
            p_alignment_pairs = [
                ('v5_3', 'v5_4'),
                ('v2_3', 'v2_4')
            ]

            for v_a, v_b in p_alignment_pairs:
                if v_a in all_found_v_for_fixed_value and v_b in all_found_v_for_fixed_value:
                    p_a = f"{v_a}_p"
                    p_b = f"{v_b}_p"
                    
                    conditions['must_equal'].append([p_a, p_b])
                    print(f"[조건 추가] 불가영역 내 쌍({v_a}, {v_b}) 동시 검출 -> 위치 변수 '{p_a}' == '{p_b}' 조건 생성")
        
        return conditions

    # ---------- 영역 분석 ----------
    def _analyze_and_store(self, rect_list, title, mode):
        if not rect_list:
            print(f"\n>> 분석할 {title} 영역이 없습니다.")
            self.analysis_results[mode] = {'add_unconditional': [], 'fixed_zero': [], 'non_zero': [], 'not_equal': [], 'must_equal': [], 'fixed_value': {}, 'upper_bound': {}}
            return
        
        wide_found_h, wide_found_v = set(), set()
        final_non_zero_from_tall_boxes = set()
        final_fixed_zero_from_tall_boxes_p = set()

        if not self.cad_extent:
            print("[오류] cad_extent(좌표 지도)가 없습니다. 분석 실패.")
            return
        
        for rect_data in rect_list:
            user_rect = self._pixel_to_cad_rect(rect_data['px_coords'])
            if not user_rect: continue

            # --- 픽셀 좌표 직접 비교 -> CAD 좌표 차이 비교 ---
            px = rect_data['px_coords']
            pixel_width = abs(px[0] - px[2])
            pixel_height = abs(px[1] - px[3])
            is_tall = (pixel_height >= pixel_width)
            
            found_h = {r['id'] for r in self.horizontal_rect_rules if (user_rect['x'][0] < r['x_range'][1] and user_rect['x'][1] > r['x_range'][0] and user_rect['y'][0] < r['y_range'][1] and user_rect['y'][1] > r['y_range'][0])}
            found_v = {r['id'] for r in self.vertical_rect_rules if (user_rect['x'][0] < sorted(r['x_range'])[1] and user_rect['x'][1] > sorted(r['x_range'])[0] and user_rect['y'][0] < sorted(r['y_range'])[1] and user_rect['y'][1] > sorted(r['y_range'])[0])}

            if mode == 'required' and is_tall:
                non_zero_candidates_for_this_box = set(found_v)
                
                for layer in range(1, 5):
                    for i in range(1, 6):
                        v_small = f"v{i}_{layer}"
                        v_large = f"v{i+1}_{layer}"
                        
                        if {v_small, v_large}.issubset(found_v):
                            print(f"[조건 생성] 단일 '세로형 박스'에서 쌍 검출: ({v_small}, {v_large}) -> '{v_large}'는 제외하고 '{v_small}'만 non-zero 조건 부여")
                            if v_large in non_zero_candidates_for_this_box:
                                non_zero_candidates_for_this_box.remove(v_large)
                
                final_non_zero_from_tall_boxes.update(non_zero_candidates_for_this_box)

                pairs_to_check_for_p_zero = [
                    ('v3_1', 'v4_1'),
                    ('v3_2', 'v4_2'),
                    ('v3_3', 'v4_3'),
                ]
                
                for v_small, v_large in pairs_to_check_for_p_zero:
                    if {v_small, v_large}.issubset(found_v):
                        param_small_p = f"{v_small}_p"
                        print(f"[조건 생성] '세로형 필수 박스'에서 쌍 검출: ({v_small}, {v_large}) -> '{param_small_p}'만 fixed_zero에 추가")
                        final_fixed_zero_from_tall_boxes_p.add(param_small_p)

            if not is_tall:
                wide_found_h.update(found_h)
                wide_found_v.update(found_v)
        
        all_found_v = wide_found_v | final_non_zero_from_tall_boxes 
        
        print("\n" + "="*50 + f"\n--- {title} 분석 결과 (종합) ---\n" +
              f"[세로형 검출->non_zero] v: {', '.join(sorted(final_non_zero_from_tall_boxes)) if final_non_zero_from_tall_boxes else '없음'}\n" +
              f"[세로형 검출->fixed_zero_p] v_p: {', '.join(sorted(final_fixed_zero_from_tall_boxes_p)) if final_fixed_zero_from_tall_boxes_p else '없음'}\n" +
              f"[가로형] h: {', '.join(sorted(wide_found_h)) if wide_found_h else '없음'}\n" +
              f"[가로형] v: {', '.join(sorted(wide_found_v)) if wide_found_v else '없음'}\n" + "="*50)
        
        if mode == 'required':
            conditions = self._get_conditions(found_vertical=set(), mode=mode, all_found_v_for_fixed_value=all_found_v, wide_found_h=wide_found_h)
            conditions['non_zero'].extend(list(final_non_zero_from_tall_boxes))
            conditions['fixed_zero'].extend(list(final_fixed_zero_from_tall_boxes_p))
            
        else: 
            tall_found_v = set() 
            rect_list_v_rules = set()
            for rect_data in rect_list:
                user_rect = self._pixel_to_cad_rect(rect_data['px_coords'])
                if not user_rect: continue
                found_v = {r['id'] for r in self.vertical_rect_rules if (user_rect['x'][0] < sorted(r['x_range'])[1] and user_rect['x'][1] > sorted(r['x_range'])[0] and user_rect['y'][0] < sorted(r['y_range'])[1] and user_rect['y'][1] > sorted(r['y_range'])[0])}
                rect_list_v_rules.update(found_v)
            
            all_h_rules = set() 
            for rect_data in rect_list:
                user_rect = self._pixel_to_cad_rect(rect_data['px_coords'])
                if not user_rect: continue
                found_h = {r['id'] for r in self.horizontal_rect_rules if (user_rect['x'][0] < r['x_range'][1] and user_rect['x'][1] > r['x_range'][0] and user_rect['y'][0] < r['y_range'][1] and user_rect['y'][1] > r['y_range'][0])}
                all_h_rules.update(found_h)

            conditions = self._get_conditions(found_horizontal=all_h_rules, found_vertical=rect_list_v_rules, mode=mode, all_found_v_for_fixed_value=rect_list_v_rules)
            
        self.analysis_results[mode] = conditions

    def _save_constraints_to_json(self):
        final_constraints = {
            'add_unconditional': [], 'fixed_zero': [], 'non_zero': [],
            'not_equal': [], 'must_equal': [], 'fixed_value': {}, 'forbidden_zones': [], 'upper_bound': {}
        }
        if 'required' in self.analysis_results:
            req_res = self.analysis_results['required']
            final_constraints['fixed_zero'].extend(req_res.get('fixed_zero', []))
            final_constraints['non_zero'].extend(req_res.get('non_zero', []))
            final_constraints['not_equal'].extend(req_res.get('not_equal', []))
            final_constraints['must_equal'].extend(req_res.get('must_equal', []))
            final_constraints['fixed_value'].update(req_res.get('fixed_value', {}))
            final_constraints['upper_bound'].update(req_res.get('upper_bound', {}))

        if 'forbidden' in self.analysis_results:
            for_res = self.analysis_results['forbidden']
            final_constraints['fixed_zero'].extend(for_res.get('fixed_zero', []))
            final_constraints['must_equal'].extend(for_res.get('must_equal', []))
            final_constraints['fixed_value'].update(for_res.get('fixed_value', {}))

        if self.fix_h1p and self.h1_p_fixed_value is not None:
            final_constraints['fixed_value']['h1_p'] = self.h1_p_fixed_value
            print(f"[조건 추가] CAD에서 추출한 'h1_p' 값을 {self.h1_p_fixed_value}로 고정합니다.")
            final_constraints['h1_p_components'] = {
                "base": self.min_y_baseline,
                "y": self.h1_p_matched_y
            }
            print(f"[조건 추가] 'h1_p_components' (base: {self.min_y_baseline:.2f}, y: {self.h1_p_matched_y:.2f})를 저장합니다.")
        elif not self.h1p_box_drawn:
            final_constraints['h1_p_components'] = {
                "base": self.min_y_baseline
            }
            print(f"[조건 추가] 'h1_p' 박스를 그리지 않았으므로 'base'({self.min_y_baseline:.2f}) 값만 저장합니다.")
        else:
            print("[알림] 'h1_p' 관련 값이 'fixed_value'에 추가되지 않았습니다. (추출 실패)")
        
        if hasattr(self, 'forming_variable_results') and self.forming_variable_results:
            final_constraints['forming_variable'] = self.forming_variable_results
            print(f"[조건 추가] 'forming_variable' {self.forming_variable_results}를 저장합니다.")

        def _pair_key(pair):
            a, b = sorted(pair)
            base_a = _h_base(a) if _is_h_up_or_down(a) else a
            base_b = _h_base(b) if _is_h_up_or_down(b) else b
            return base_a if base_a and base_a == base_b else (a, b)
        
        must_eq_keys = set(_pair_key(p) for p in final_constraints['must_equal'])
        final_constraints['not_equal'] = [p for p in final_constraints['not_equal'] if _pair_key(p) not in must_eq_keys]
        
        fixed_zero_set = set(final_constraints['fixed_zero'])
        final_constraints['non_zero'] = [v for v in final_constraints['non_zero'] if v not in fixed_zero_set]
        final_constraints['forbidden_zones'] = self.forbidden_zone_coords
        
        for key, value in final_constraints.items():
            if key in ['fixed_zero', 'non_zero']:
                final_constraints[key] = sorted(list(set(value)))
            elif key in ['not_equal', 'must_equal']:
                unique_pairs = list(set(tuple(sorted(p)) for p in value))
                final_constraints[key] = sorted([list(p) for p in unique_pairs])
        
        filepath = 'constraints.json'
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(final_constraints, f, ensure_ascii=False, indent=4)
        print(f"\n[성공] 제약 조건이 '{os.path.abspath(filepath)}' 파일에 저장되었습니다.")

# ========================= 실행부 =========================
if __name__ == '__main__':
    # STEP_FILE=r"F:\Final_Code\4_Optimization\Matching_Parameter\Drawing\delete.stp"
    # STEP_FILE=r"F:\Final_Code\4_Optimization\Matching_Parameter\Drawing\cross.stp"
    STEP_FILE=r"F:\Final_Code\4_Optimization\Matching_Parameter\Drawing\two_cross.stp"
    # STEP_FILE=r"F:\F:\Final_Code\4_Optimization\Matching_Parameter\Drawing\55_cover_bottom_simplified.stp"
    # STEP_FILE=r"F:\Final_Code\4_Optimization\Matching_Parameter\Drawing\55_cover_bottom.stp"
    Z_THRESHOLD = -43.2
    Z_OUTLINE   = -38.2
    MIN_AREA    = 1500.0

    CONSTRAINTS_FILE = 'constraints.json'
    if os.path.exists(CONSTRAINTS_FILE):
        try:
            os.remove(CONSTRAINTS_FILE)
            print(f"\n[초기화] 기존 '{CONSTRAINTS_FILE}' 파일을 삭제했습니다.")
        except OSError as e:
            print(f"\n[오류] '{CONSTRAINTS_FILE}' 파일 삭제 중 오류 발생: {e}")

    if not os.path.exists(STEP_FILE):
        print(f"오류: STEP 파일을 찾을 수 없습니다 - {STEP_FILE}")
    else:
        root = tk.Tk()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        win_width = int(screen_width * 0.6)
        win_height = int(screen_height * 0.6)
        win_x = int((screen_width - win_width) / 2)
        win_y = int((screen_height - win_height) / 2)
        root.geometry(f"{win_width}x{win_height}+{win_x}+{win_y}")
        app = InteractiveCadSelectorApp(root, STEP_FILE, Z_THRESHOLD, Z_OUTLINE, win_width, win_height, min_area=MIN_AREA)
        root.mainloop()