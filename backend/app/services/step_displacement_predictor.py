# ===================================================================
# STEP 파일 변위 예측 자동화 스크립트
#
# 기능:
# 1. STEP(.stp) 파일에서 95개 형상인자를 내부적으로 추출
# 2. PI-DeepONet 모델을 사용하여 변위(Displacement) 예측
# 3. 예측된 최대 변위 값을 터미널에 출력
# ===================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# --- Matplotlib 시각화 관련 라이브러리 ---
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import ticker
from collections import defaultdict

# --- 3D CAD(python-occ) 관련 라이브러리 ---
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import topods
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_WIRE
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.GeomAbs import GeomAbs_Line
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.BRepTools import BRepTools_WireExplorer
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib
from OCC.Core.IntCurvesFace import IntCurvesFace_ShapeIntersector
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Lin

def _pick_device():
    if os.environ.get("FORCE_CPU", "").lower() in ("1","true","yes"):
        return torch.device("cpu")
    if torch.cuda.is_available():
        try:
            torch.cuda.current_device()
            _ = torch.zeros(1, device="cuda")
            return torch.device("cuda")
        except Exception:
            pass
    return torch.device("cpu")

# ===================================================================
# 1. 경로 설정 (!! 사용자 환경에 맞게 이 부분을 수정해주세요 !!)
# ===================================================================

# --- 입력 STEP 파일 경로 ---
# 분석하고 싶은 STEP 파일의 전체 경로를 지정합니다.
STEP_FILE_PATH = r"F:\DeepONet_2\CAD\drawing\delete.stp"
# STEP_FILE_PATH = r"F:\DeepONet_2\CAD\drawing\cross.stp"
# STEP_FILE_PATH = r"F:\DeepONet_2\CAD\drawing\two_cross.stp"

# --- AI 모델 및 자산 경로 ---
# 모델 가중치(.pth), 스케일러 등이 저장된 폴더 경로입니다.
ASSETS_DIR = r'F:\DeepONet_2\7200_Grav\7200_DOE_ONE\grav_model_assets'

# --- 시각화용 좌표 데이터 경로 ---
# 변위 등고선 플롯을 그릴 때 사용할 X, Y 좌표 원본 CSV 파일 경로입니다.
ORIGINAL_DATA_DIR = r'F:\DeepONet_2\7200_Grav\7200_DOE_ONE\Expansion_4'
TRUNK_CSV_PATH = os.path.join(ORIGINAL_DATA_DIR, 'Weight_DOE900_Scaling_120x60_Trunk_4.csv')


# ===================================================================
# 2. STEP 파일 분석 및 형상인자 추출 함수
# ===================================================================

def _compute_uniform_thickness_quick(shape, eps_gap=1e-3):
    """
    쉘 모델(두께 일정) 가정 하에서, Z-방향 레이와 shape의 교차 z좌표들 사이의
    '최소 양의 간격'을 두께로 판단. 중앙에서 시도 후 필요 시 3x3 소그리드.
    실패 시 바운딩박스 높이로 폴백.
    """
    box = Bnd_Box()
    brepbndlib.Add(shape, box)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    xmid, ymid = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    zspan = zmax - zmin

    def ray_hit_gaps(x0, y0):
        line = gp_Lin(gp_Pnt(x0, y0, (zmin + zmax) * 0.5), gp_Dir(0, 0, 1))
        inter = IntCurvesFace_ShapeIntersector()
        inter.Load(shape, 1e-7)
        inter.Perform(line, zmin - zspan, zmax + zspan)
        if not inter.IsDone() or inter.NbPnt() < 2: return None
        zs = sorted([inter.Pnt(i).Z() for i in range(1, inter.NbPnt() + 1)])
        gaps = [g for g in [zs[i+1] - zs[i] for i in range(len(zs)-1)] if g > eps_gap]
        return min(gaps) if gaps else None

    t = ray_hit_gaps(xmid, ymid)
    if t is not None: return t
    
    dx, dy = 0.1 * (xmax - xmin), 0.1 * (ymax - ymin)
    samples = [(xmid, ymid), (xmid+dx, ymid), (xmid-dx, ymid), (xmid, ymid+dy), (xmid, ymid-dy),
               (xmid+dx, ymid+dy), (xmid-dx, ymid+dy), (xmid+dx, ymid-dy), (xmid-dx, ymid-dy)]
    
    thicknesses = [res for xx, yy in samples if (res := ray_hit_gaps(xx, yy)) is not None]
    return min(thicknesses) if thicknesses else abs(zmax - zmin)


def extract_shape_factors_from_step(step_file, z_threshold, z_outline, min_area, horizontal_rules_mixed, vertical_rules, all_rule_ids_in_order, fixed_h_defaults):
    """
    STEP 파일에서 형상인자를 추출하여 예측에 사용할 95개 값을 DataFrame으로 반환합니다.
    """
    print("--- [단계 1] STEP 파일 분석 및 형상인자 추출 시작 ---")
    
    # 1. STEP 파일 로드
    reader = STEPControl_Reader()
    if not os.path.exists(step_file):
        print(f"오류: STEP 파일을 찾을 수 없습니다 - {step_file}")
        return None
    reader.ReadFile(step_file)
    reader.TransferRoots()
    shape = reader.Shape()
    print(f"  -> '{os.path.basename(step_file)}' 파일 로딩 성공.")

    # 2. Face 추출
    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    target_faces, outline_faces = [], []
    while face_explorer.More():
        current_face = topods.Face(face_explorer.Current())
        props = GProp_GProps()
        brepgprop.SurfaceProperties(current_face, props)
        z_coord = props.CentreOfMass().Z()
        area = props.Mass()
        if area > min_area and z_coord <= z_threshold:
            target_faces.append(current_face)
        if abs(z_coord - z_outline) < 1e-6:
            outline_faces.append(current_face)
        face_explorer.Next()
    
    min_y_baseline = 0
    outline_verts = []
    for face in outline_faces:
        wire_explorer = TopExp_Explorer(face, TopAbs_WIRE)
        while wire_explorer.More():
            we = BRepTools_WireExplorer(topods.Wire(wire_explorer.Current()))
            while we.More():
                outline_verts.append(tuple(BRep_Tool.Pnt(we.CurrentVertex()).Coord()[:2]))
                we.Next()
            wire_explorer.Next()
    
    if outline_verts:
        y_coords = [v[1] for v in outline_verts]
        min_y_baseline = min(y_coords)
    
    # h5 규칙 처리
    h5_y_offsets = {'h5_1':(28,80),'h5_6':(28,80),'h5_2':(22,80),'h5_5':(22,80),'h5_3':(15,45),'h5_4':(15,45)}
    processed_horizontal_rules = []
    for rule in horizontal_rules_mixed:
        if rule.get('id', '').startswith('h5'):
            p1, p2 = rule['start_point'], rule['end_point']
            x_center = (p1[0] + p2[0]) / 2
            y_offset = h5_y_offsets[rule['id']]
            processed_horizontal_rules.append({'id': rule['id'], 'x_range': (x_center - 20, x_center + 20), 'y_range': (min_y_baseline + y_offset[0], min_y_baseline + y_offset[1])})
        else:
            processed_horizontal_rules.append(rule)

    # 3. 수평/수직 선분 추출
    horizontal_cad_lines, vertical_cad_lines = [], []
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
                    horizontal_cad_lines.append(((p1.X(), p1.Y()), (p2.X(), p2.Y())))
                elif abs(p1.X() - p2.X()) < tolerance and abs(p1.Y() - p2.Y()) > tolerance:
                    vertical_cad_lines.append(((p1.X(), p1.Y()), (p2.X(), p2.Y())))
            edge_explorer.Next()
    
    # 4. 매칭 로직 수행 및 값 저장
    matched_values = {}
    
    # 4.1. 수평 규칙 매칭
    for rule in processed_horizontal_rules:
        rx_min, rx_max = sorted(rule['x_range'])
        ry_min, ry_max = sorted(rule['y_range'])
        rule_center_x = (rx_min + rx_max) / 2
        rule_center_y = (ry_min + ry_max) / 2
        
        candidate_lines = []
        for p1, p2 in horizontal_cad_lines:
            line_y = p1[1]
            line_x_min, line_x_max = sorted((p1[0], p2[0]))
            if (ry_min <= line_y <= ry_max) and (line_x_min <= rx_max and line_x_max >= rx_min):
                candidate_lines.append({'line': (p1, p2), 'y': line_y, 'length': abs(p1[0] - p2[0])})
        
        if not candidate_lines: continue
        rule_id = rule['id']

        if rule_id == 'rev_forming1':
            long_lines = [c for c in candidate_lines if c['length'] >= 100]
            if long_lines:
                matched_values['rev_forming1'] = len(long_lines) / 4
                y_values = [line['y'] for line in long_lines]
                matched_values['forming_width'] = max(y_values) - min(y_values) if len(long_lines) > 1 else 0
            continue

        best_match = None
        if rule_id == 'h1_p':
            max_y = max(c['y'] for c in candidate_lines)
            top_candidates = [c for c in candidate_lines if c['y'] == max_y]
            if len(top_candidates) == 1:
                best_match = top_candidates[0]
            else:
                for cand in top_candidates: cand['dist'] = (rule_center_x - (cand['line'][0][0] + cand['line'][1][0])/2)**2
                best_match = min(top_candidates, key=lambda x: x['dist'])
        else:
            for cand in candidate_lines: cand['dist'] = (rule_center_x - (cand['line'][0][0] + cand['line'][1][0])/2)**2 + (rule_center_y - cand['y'])**2
            best_match = min(candidate_lines, key=lambda x: x['dist'])

        if best_match:
            p1, p2 = best_match['line']
            matched_y = p1[1]
            if rule_id == 'h1_p':
                matched_values[rule_id] = abs(min_y_baseline) + matched_y
            elif rule_id.startswith('h5'):
                matched_values[rule_id] = abs(min_y_baseline - matched_y)
            else:
                matched_values[rule_id] = matched_y

    # 4.2. 수직 규칙 매칭
    for rule in vertical_rules:
        rx_min, rx_max = sorted(rule['x_range'])
        ry_min, ry_max = sorted(rule['y_range'])
        candidate_lines = []
        for p1, p2 in vertical_cad_lines:
            line_x = p1[0]
            line_y_min, line_y_max = sorted((p1[1], p2[1]))
            if (rx_min <= line_x <= rx_max) and (line_y_min <= ry_max and line_y_max >= ry_min):
                candidate_lines.append({'line': (p1, p2), 'x': line_x})
        
        if not candidate_lines: continue
        rule_id, rule_id_prefix = rule['id'], rule['id'].split('_')[0]
        
        is_left_side = rule_id_prefix in ['v4', 'v5', 'v6']
        main_match = min(candidate_lines, key=lambda c: c['x']) if is_left_side else max(candidate_lines, key=lambda c: c['x'])
        matched_values[rule_id] = main_match['line'][0][0]

        if len(candidate_lines) > 1:
            p_match = max(candidate_lines, key=lambda c: c['x']) if is_left_side else min(candidate_lines, key=lambda c: c['x'])
            matched_values[rule_id + '_p'] = p_match['line'][0][0]

    # 4.3. 매칭 값 후처리
    for rule in vertical_rules:
        rid, p_rid = rule['id'], rule['id'] + '_p'
        if rid in matched_values and p_rid in matched_values and abs(matched_values[rid] - matched_values[p_rid]) < 1e-5:
            del matched_values[rid], matched_values[p_rid]

    for rid in list(matched_values.keys()):
        if rid.startswith('v') and not rid.endswith('_p'):
            p_rid = rid + '_p'
            if p_rid in matched_values:
                matched_values[rid] = abs(matched_values[rid] - matched_values[p_rid])

    # 4.4. 기타 인자 계산
    matched_values['thick'] = _compute_uniform_thickness_quick(shape)
    matched_values['bending height (T/L/R)'] = 3.8
    matched_values['bending height (B)'] = -5.0

    # 5. 최종 결과 데이터 생성
    final_values = {}
    for rule_id in all_rule_ids_in_order:
        value = matched_values.get(rule_id)
        if value is None:
            if rule_id in fixed_h_defaults:
                value = fixed_h_defaults[rule_id]
            elif rule_id.startswith('v') and rule_id.endswith('_p'):
                value = 567
            else:
                value = 0
        
        param94_name, param95_name = 'bending height (T/L/R)', 'bending height (B)'
        if (rule_id.startswith('h') or rule_id.endswith('_p')) and rule_id not in [param94_name, param95_name]:
            value = abs(value)
        
        final_values[rule_id] = value
        
    # 예측에 사용할 DataFrame 생성
    branch_input_df = pd.DataFrame([final_values], columns=all_rule_ids_in_order)
    
    print("--- [단계 1] 형상인자 추출 완료 ---")
    return branch_input_df


# ===================================================================
# 3. PI-DeepONet 모델 정의 및 변위 예측 함수
# ===================================================================

class MLP(nn.Module):
    def __init__(self, in_dim, h_dims, out_dim, dropout=0.1):
        super().__init__()
        layers = []
        prev = in_dim
        for h in h_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class DeepONet(nn.Module):
    def __init__(self, b_net, t_net):
        super().__init__()
        self.branch_net = b_net
        self.trunk_net  = t_net
    def forward(self, xb, xt):
        return torch.matmul(self.branch_net(xb), self.trunk_net(xt).transpose(0,1))

def predict_displacement(branch_input_df, assets_dir, trunk_csv_path, output_contour_path):
    """
    추출된 형상인자(DataFrame)를 입력받아 변위를 예측하고 결과를 출력합니다.
    """
    print("\n--- [단계 2] 변위 예측 시작 ---")
    
    # 1. 자산 로딩 및 모델 구조 복원
    try:
        assets_path = os.path.join(assets_dir, 'assets_for_ga.pth')
        model_path = os.path.join(assets_dir, 'best_grav_model.pth')
        
        assets = torch.load(assets_path, map_location='cpu', weights_only=False)
        model_state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
        print("  -> 모델 자산 및 가중치 로딩 성공.")

        branch_scaler = assets['branch_scaler']
        target_scaler = assets['target_scaler']
        trunk_tensor = assets['trunk_tensor']
        feature_weights = assets.get('feature_weights', {})
        model_params = assets['model_hyperparameters']

    except FileNotFoundError as e:
        print(f"오류: 필수 모델 파일({e.filename})을 찾을 수 없습니다.")
        print(f"'{assets_dir}' 폴더가 올바른지 확인해주세요.")
        return

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = _pick_device()   
    model = DeepONet(
        MLP(model_params['branch_input_dim'], model_params['branch_hidden_dims'], model_params['latent_dim']),
        MLP(model_params['trunk_input_dim'],  model_params['trunk_hidden_dims'],  model_params['latent_dim'])
    ).to(device)
    model.load_state_dict(model_state_dict)
    model.eval()
    trunk_tensor = trunk_tensor.to(device)
    print(f"  -> PI-DeepONet 모델이 {device} 장치에서 성공적으로 복원되었습니다.")

    # 2. 입력 데이터 전처리
    scaled_values = branch_scaler.transform(branch_input_df.values)
    scaled_df = pd.DataFrame(scaled_values, columns=branch_input_df.columns)
    for feat, w in feature_weights.items():
        if feat in scaled_df.columns:
            scaled_df[feat] *= w
    branch_input_tensor = torch.tensor(scaled_df.values, dtype=torch.float32).to(device)
    print("  -> 입력 형상인자 스케일링 및 가중치 적용 완료.")

    # 3. 모델 예측 및 후처리
    with torch.no_grad():
        scaled_pred = model(branch_input_tensor, trunk_tensor)
    physical_pred = target_scaler.inverse_transform(scaled_pred.cpu().numpy()).flatten()
    print("  -> 모델 예측 및 물리 단위 변환 완료.")

    # 4. 시각화 (선택 사항) 및 결과 출력
    try:
        trunk_df = pd.read_csv(trunk_csv_path)
        x_coords, y_coords = trunk_df["X"].values, trunk_df["Y"].values
    except FileNotFoundError:
        print(f"오류: 시각화용 좌표 파일 '{trunk_csv_path}'를 찾을 수 없습니다.")
        # 좌표 파일이 없어도 최대값 계산은 가능하므로 계속 진행
        x_coords, y_coords = None, None

    max_abs_val = np.abs(physical_pred).max()
    
    # --- 이미지 생성 및 저장 (필요 시 주석 해제) ---
    # fig, ax = plt.subplots(figsize=(12, 6))
    # colors_list = [(0,0,200/255),(21/255,121/255,1),(0,199/255,221/255),(40/255,1,185/255),(57/255,1,0),
    #                (170/255,1,0),(1,227/255,0),(1,113/255,0),(1,0,0)]
    # custom_cmap = ListedColormap(colors_list)
    # vmin, vmax = np.min(physical_pred), np.max(physical_pred)
    # boundaries = np.linspace(vmin, vmax, 10)
    # if x_coords is not None:
    #     idx_max_abs = np.argmax(np.abs(physical_pred))
    #     cp = ax.tricontourf(x_coords, y_coords, physical_pred, levels=boundaries, cmap=custom_cmap)
    #     ax.plot(x_coords[idx_max_abs], y_coords[idx_max_abs], 'o', markersize=12, markerfacecolor='none', 
    #             markeredgecolor='white', markeredgewidth=2, 
    #             label=f'Max Abs Disp: {abs(physical_pred[idx_max_abs]):.4f} mm at ({x_coords[idx_max_abs]:.1f}, {y_coords[idx_max_abs]:.1f})')
    #     ax.set_title("Predicted Displacement Contour from Shape Factors", fontsize=16)
    #     ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
    #     ax.set_aspect("equal"); ax.legend()
    #     divider = make_axes_locatable(ax)
    #     cax = divider.append_axes("right", size="5%", pad=0.1)
    #     fig.colorbar(cp, cax=cax, format=ticker.FormatStrFormatter('%.3f'), label="Displacement (mm)")
    #     plt.tight_layout()
    #     plt.savefig(output_contour_path, dpi=200)
    #     print(f"  -> 변위 예측 등고선 플롯 저장 완료: '{output_contour_path}'")
    # plt.close('all')
    
    print("\n==========================================================")
    print(f" 최종 결과: 예측된 최대 절대 변위 = {max_abs_val:.6f} mm")
    print("==========================================================")
    print("\n--- [단계 2] 변위 예측 완료 ---")


def compute_preopt_max_displacement(step_file_path: str) -> float:
    """
    로직/수치 변경 없이, 스크립트에 정의된 함수/상수를 그대로 사용해
    주어진 STEP 파일의 최대 절대 변위를 계산해 float으로 반환합니다.
    """
    # 스크립트 상단의 경로 상수들을 현재 저장소 기준으로 설정
    # (수치/로직이 아니라 '경로'만 프로젝트 안으로 맞춥니다)
    assets_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "PI-DeepONet", "grav_model_assets"))
    original_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "PI-DeepONet", "Expansion_4"))
    trunk_csv_path = os.path.join(original_data_dir, 'Weight_DOE900_Scaling_120x60_Trunk_4.csv')

    # --- 규칙 / 파라미터 정의는 붙여넣은 스크립트의 값들을 그대로 사용합니다 ---
    from collections import defaultdict

    output_order_ids = [
        'h1_p',
        'h1_1_down','h1_2_down','h1_3_down','h1_4_down','h1_5_down',
        'h2_1_up','h2_2_up','h2_3_up','h2_4_up','h2_5_up',
        'h2_1_down','h2_2_down','h2_3_down','h2_4_down','h2_5_down',
        'h3_1_up','h3_2_up','h3_3_up','h3_4_up','h3_5_up',
        'h3_1_down','h3_2_down','h3_3_down','h3_4_down','h3_5_down',
        'h4_1_up','h4_2_up','h4_3_up','h4_4_up','h4_5_up',
        'h4_1_down','h4_2_down','h4_3_down','h4_4_down','h4_5_down',
        'h5_1','h5_2','h5_3','h5_4','h5_5','h5_6',
        'v1_1','v1_1_p','v2_1','v2_1_p','v3_1','v3_1_p','v4_1','v4_1_p','v5_1','v5_1_p','v6_1','v6_1_p',
        'v1_2','v1_2_p','v2_2','v2_2_p','v3_2','v3_2_p','v4_2','v4_2_p','v5_2','v5_2_p','v6_2','v6_2_p',
        'v1_3','v1_3_p','v2_3','v2_3_p','v3_3','v3_3_p','v4_3','v4_3_p','v5_3','v5_3_p','v6_3','v6_3_p',
        'v1_4','v1_4_p','v2_4','v2_4_p','v3_4','v3_4_p','v4_4','v4_4_p','v5_4','v5_4_p','v6_4','v6_4_p',
        'rev_forming1','forming_width','thick',
        'bending height (T/L/R)','bending height (B)'
    ]

    HORIZONTAL_RULES_LINES = [{'id':'h1_1_down','start_point':(497.65,163.06),'end_point':(255.0,163.06)},{'id':'h1_2_down','start_point':(220.0,163.06),'end_point':(155.0,163.06)},{'id':'h1_3_down','start_point':(80.0,163.06),'end_point':(-80.0,163.06)},{'id':'h1_4_down','start_point':(-220.0,163.06),'end_point':(-155.0,163.06)},{'id':'h1_5_down','start_point':(-497.65,163.06),'end_point':(-255.0,163.06)},{'id':'h2_1_up','start_point':(497.65,75.0),'end_point':(255.0,75.0)},{'id':'h2_2_up','start_point':(220.0,75.0),'end_point':(155.0,75.0)},{'id':'h2_3_up','start_point':(80.0,75.0),'end_point':(-80.0,75.0)},{'id':'h2_4_up','start_point':(-220.0,75.0),'end_point':(-155.0,75.0)},{'id':'h2_5_up','start_point':(-497.65,75.0),'end_point':(-255.0,75.0)},{'id':'h2_1_down','start_point':(497.65,15.0),'end_point':(403.67,15.0)},{'id':'h2_2_down','start_point':(368.67,15.0),'end_point':(180.0,15.0)},{'id':'h2_3_down','start_point':(80.0,15.0),'end_point':(-80.0,15.0)},{'id':'h2_4_down','start_point':(-368.67,15.0),'end_point':(-180.0,15.0)},{'id':'h2_5_down','start_point':(-497.65,15.0),'end_point':(-403.67,15.0)},{'id':'h3_1_up','start_point':(497.65,-14.5),'end_point':(403.67,-14.5)},{'id':'h3_2_up','start_point':(368.67,-14.5),'end_point':(180.0,-14.5)},{'id':'h3_3_up','start_point':(80.0,-14.5),'end_point':(-80.0,-14.5)},{'id':'h3_4_up','start_point':(-368.67,-14.5),'end_point':(-180.0,-14.5)},{'id':'h3_5_up','start_point':(-497.65,-14.5),'end_point':(-403.67,-14.5)},{'id':'h3_1_down','start_point':(497.65,-70.5),'end_point':(367.5,-70.5)},{'id':'h3_2_down','start_point':(317.5,-70.5),'end_point':(180.0,-70.5)},{'id':'h3_3_down','start_point':(80.0,-70.5),'end_point':(-80.0,-70.5)},{'id':'h3_4_down','start_point':(-317.5,-70.5),'end_point':(-180.0,-70.5)},{'id':'h3_5_down','start_point':(-497.65,-70.5),'end_point':(-367.5,-70.5)},{'id':'h4_1_up','start_point':(497.65,-162.8),'end_point':(367.5,-162.8)},{'id':'h4_2_up','start_point':(317.5,-162.8),'end_point':(180.0,-162.8)},{'id':'h4_3_up','start_point':(80.0,-162.8),'end_point':(-80.0,-162.8)},{'id':'h4_4_up','start_point':(-317.5,-162.8),'end_point':(-180.0,-162.8)},{'id':'h4_5_up','start_point':(-497.65,-162.8),'end_point':(-367.5,-162.8)},{'id':'h4_1_down','start_point':(493.65,-220.96),'end_point':(427.5,-220.96)},{'id':'h4_2_down','start_point':(347.5,-220.96),'end_point':(200.5,-220.96)},{'id':'h4_3_down','start_point':(180,-251.96),'end_point':(-180,-251.96)},{'id':'h4_4_down','start_point':(-347.5,-220.96),'end_point':(-200.5,-220.96)},{'id':'h4_5_down','start_point':(-493.65,-220.96),'end_point':(-427.5,-220.96)},{'id':'h5_1','start_point':(567.95,-285.96),'end_point':(493.65,-285.96)},{'id':'h5_2','start_point':(427.5,-291.96),'end_point':(347.5,-291.96)},{'id':'h5_3','start_point':(160.0,-293.17),'end_point':(98.0,-293.17)},{'id':'h5_4','start_point':(-160.0,-293.17),'end_point':(-98.0,-293.17)},{'id':'h5_5','start_point':(-427.5,-291.96),'end_point':(-347.5,-291.96)},{'id':'h5_6','start_point':(-567.95,-285.96),'end_point':(-493.65,-285.96)}]

    VERTICAL_RULES_RECTS = [{'id':'v1_1','x_range':(497.65,567.95),'y_range':(75,163.06)},{'id':'v2_1','x_range':(220,255),'y_range':(75,163.06)},{'id':'v3_1','x_range':(80,155),'y_range':(75,163.06)},{'id':'v4_1','x_range':(-155,-80),'y_range':(75,163.06)},{'id':'v5_1','x_range':(-255,-220),'y_range':(75,163.06)},{'id':'v6_1','x_range':(-567.95,-497.65),'y_range':(75,163.06)},{'id':'v1_2','x_range':(497.65,567.95),'y_range':(-14.5,15)},{'id':'v2_2','x_range':(368.67,403.67),'y_range':(-14.5,15)},{'id':'v3_2','x_range':(80,180),'y_range':(-14.5,15)},{'id':'v4_2','x_range':(-180,-80),'y_range':(-14.5,15)},{'id':'v5_2','x_range':(-403.67,-368.67),'y_range':(-14.5,15)},{'id':'v6_2','x_range':(-567.95,-497.65),'y_range':(-14.5,15)},{'id':'v1_3','x_range':(497.65,567.95),'y_range':(-162.8,-70.5)},{'id':'v2_3','x_range':(317.5,367.5),'y_range':(-162.8,-70.5)},{'id':'v3_3','x_range':(80,180),'y_range':(-162.8,-70.5)},{'id':'v4_3','x_range':(-180,-80),'y_range':(-162.8,-70.5)},{'id':'v5_3','x_range':(-367.5,-317.5),'y_range':(-162.8,-70.5)},{'id':'v6_3','x_range':(-567.95,-497.65),'y_range':(-162.8,-70.5)},{'id':'v1_4','x_range':(493.65,567.95),'y_range':(-285.96,-220.96)},{'id':'v2_4','x_range':(347.5,427.5),'y_range':(-291.96,-220.96)},{'id':'v3_4','x_range':(98,160),'y_range':(-293.17,-251.96)},{'id':'v4_4','x_range':(-160,-98),'y_range':(-293.17,-251.96)},{'id':'v5_4','x_range':(-427.5,-347.5),'y_range':(-291.96,-220.96)},{'id':'v6_4','x_range':(-567.95,-493.65),'y_range':(-285.96,-220.96)}]

    # 규칙 전처리 (스크립트와 동일)
    h1_p_rule = {'id': 'h1_p', 'y_range': (255, 270), 'x_range': (-570, 570)}
    rev_forming1_rule_def = {'id': 'rev_forming1', 'x_range': (-535, 535), 'y_range': (200.58, 225.58)}
    y_groups = defaultdict(list)
    h5_rules_lines = [r for r in HORIZONTAL_RULES_LINES if r['id'].startswith('h5')]
    for y_center, rule in [(((h1_p_rule['y_range'][0] + h1_p_rule['y_range'][1]) / 2), h1_p_rule),
                           (((rev_forming1_rule_def['y_range'][0] + rev_forming1_rule_def['y_range'][1]) / 2), rev_forming1_rule_def)]:
        y_groups[y_center].append(rule)
    for rule in HORIZONTAL_RULES_LINES:
        if not rule['id'].startswith('h5'):
            y_groups[rule['start_point'][1]].append(rule)
    sorted_y = sorted(y_groups.keys(), reverse=True)
    y_ranges = {}
    for i, y_current in enumerate(sorted_y):
        y_upper = (y_current + sorted_y[i-1])/2 if i > 0 else h1_p_rule['y_range'][1]
        y_lower = (y_current + sorted_y[i+1])/2 if i < len(sorted_y)-1 else y_current - ((y_current + sorted_y[i-1])/2 - y_current)
        y_ranges[y_current] = (y_lower, y_upper)
    horizontal_rules_to_pass = []
    for y_coord, rules in y_groups.items():
        for rule in rules:
            if 'x_range' not in rule:
                p1, p2 = rule['start_point'], rule['end_point']
                x_center = (p1[0] + p2[0]) / 2
                rule['x_range'] = (x_center - 20, x_center + 20)
            rule['y_range'] = y_ranges[y_coord]
            horizontal_rules_to_pass.append(rule)
    horizontal_rules_to_pass.extend(h5_rules_lines)
    midpoint_y = (-162.8 - 251.96) / 2
    for rule in horizontal_rules_to_pass:
        if rule['id'] == 'h4_3_up': rule['y_range'] = (midpoint_y, rule['y_range'][1])
        elif rule['id'] == 'h4_3_down': rule['y_range'] = (rule['y_range'][0], midpoint_y)
    fixed_h_defaults = {f'h{j}_{i}_{s}': v for j, s, v in [(4,'up',260),(3,'up',120),(2,'down',120),(1,'down',240)] for i in range(1,6)}

    # STEP → 형상인자 추출
    branch_input_df = extract_shape_factors_from_step(
        step_file=step_file_path,
        z_threshold=-43.2,
        z_outline=-38.2,
        min_area=1500.0,
        horizontal_rules_mixed=horizontal_rules_to_pass,
        vertical_rules=VERTICAL_RULES_RECTS,
        all_rule_ids_in_order=output_order_ids,
        fixed_h_defaults=fixed_h_defaults
    )
    if branch_input_df is None:
        raise RuntimeError("STEP에서 형상인자 추출 실패")

    # 변위 예측: 스크립트의 predict_displacement 내부 로직과 동일 계산 (수치/로직 동일)
    assets_path = os.path.join(assets_dir, 'assets_for_ga.pth')
    model_path = os.path.join(assets_dir, 'best_grav_model.pth')
    assets = torch.load(assets_path, map_location='cpu', weights_only=False)
    model_state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
    branch_scaler = assets['branch_scaler']
    target_scaler = assets['target_scaler']
    trunk_tensor = assets['trunk_tensor']
    feature_weights = assets.get('feature_weights', {})
    model_params = assets['model_hyperparameters']
    device = _pick_device()
    model = DeepONet(
        MLP(model_params['branch_input_dim'], model_params['branch_hidden_dims'], model_params['latent_dim']),
        MLP(model_params['trunk_input_dim'],  model_params['trunk_hidden_dims'],  model_params['latent_dim'])
    ).to(device)
    model.load_state_dict(model_state_dict)
    model.eval()
    trunk_tensor = trunk_tensor.to(device)

    scaled_values = branch_scaler.transform(branch_input_df.values)
    scaled_df = pd.DataFrame(scaled_values, columns=branch_input_df.columns)
    for feat, w in feature_weights.items():
        if feat in scaled_df.columns:
            scaled_df[feat] *= w
    branch_input_tensor = torch.tensor(scaled_df.values, dtype=torch.float32).to(device)
    with torch.no_grad():
        scaled_pred = model(branch_input_tensor, trunk_tensor)
    physical_pred = target_scaler.inverse_transform(scaled_pred.cpu().numpy()).flatten()
    max_abs_val = float(np.abs(physical_pred).max())
    return max_abs_val

