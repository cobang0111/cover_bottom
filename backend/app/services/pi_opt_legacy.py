# ==================================================================
# FINAL INTEGRATED SCRIPT (All Syntax Errors Corrected)
# ==================================================================

import os
import types
import re
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import json
import copy

# --- OCC (CAD) 관련 라이브러리 ---
try:
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Core.BRepBndLib import brepbndlib
    from OCC.Core.Bnd import Bnd_Box
    from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
    from OCC.Display.SimpleGui import init_display
    OCC_AVAILABLE = True
except ImportError:
    OCC_AVAILABLE = False
    print("경고: python-occ 라이브러리를 찾을 수 없습니다. CAD 배경 생성 기능이 비활성화됩니다.")

# --- DEAP (유전 알고리즘) 관련 라이브러리 ---
try:
    from deap import base, creator, tools, algorithms
except ImportError:
    print("오류: deap 라이브러리가 설치되어 있지 않습니다. 'pip install deap'으로 설치해주세요.")
    exit()

# --- 시각화 관련 라이브러리 ---
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.path import Path
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Rectangle

# ==================================================================
# PART 0: 전역 설정
# ==================================================================
OUTPUT_DIR = "GA_Result_Visualization"

# 재현성을 위한 랜덤 시드 고정
SEED = 100
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

# 자산/데이터 로드(1회)
device = _pick_device()

LEGACY_STATE = {
    "initialized": False,
    "device": _pick_device(),
    "output_dir": None,
    "assets": {},
    "dataframes": {},
    "model": None,
    "param_names": [],
    "param_bounds": [],
    "indices_template": {},
}

def _state():
    if not LEGACY_STATE["initialized"]:
        raise RuntimeError("initialize_legacy_pipeline 호출 후 사용하세요.")
    return LEGACY_STATE

# 경로 설정(프로젝트 구조)
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "PI-DeepONet"))
ASSETS_DIR = os.path.join(BASE, "grav_model_assets")
ORIG_DIR   = os.path.join(BASE, "Expansion_4")
VIEW_CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "uploads", "views"))
RESULT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "uploads", "results"))
LEGACY_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "uploads", "legacy_results")

assets_path      = os.path.join(ASSETS_DIR, "assets_for_ga.pth")
model_path       = os.path.join(ASSETS_DIR, "best_grav_model.pth")
branch_csv_path  = os.path.join(ORIG_DIR,  "Weight_DOE900_Scaling_120x60_Branch_4.csv")
trunk_csv_path   = os.path.join(ORIG_DIR,  "Weight_DOE900_Scaling_120x60_Trunk_4.csv")
top_seeds_path   = os.path.join(ASSETS_DIR,"top_50_seed_ids.csv")

trunk_df = pd.read_csv(trunk_csv_path, index_col=0)

def _ensure_creator_types():
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMin)

def initialize_legacy_pipeline(
    *,
    assets_dir: str,
    original_data_dir: str,
    seed_csv_path: str,
    output_dir: str | None = None,
):
    if LEGACY_STATE["initialized"]:
        return

    device = LEGACY_STATE["device"]
    output_dir = output_dir or os.path.join(original_data_dir, "..", "..", "legacy_results")
    os.makedirs(output_dir, exist_ok=True)

    assets_path = os.path.join(assets_dir, "assets_for_ga.pth")
    model_path  = os.path.join(assets_dir, "best_grav_model.pth")
    branch_csv  = os.path.join(original_data_dir, "Weight_DOE900_Scaling_120x60_Branch_4.csv")
    trunk_csv   = os.path.join(original_data_dir, "Weight_DOE900_Scaling_120x60_Trunk_4.csv")

    map_location = torch.device("cpu")
    assets = torch.load(assets_path, map_location=map_location, weights_only=False)
    model_state = torch.load(model_path, map_location=map_location, weights_only=False)

    branch_df = pd.read_csv(branch_csv, index_col=0)
    trunk_df  = pd.read_csv(trunk_csv, index_col=0)
    seed_ids  = pd.read_csv(seed_csv_path)["Sample_ID"].tolist()

    model_params = assets["model_hyperparameters"]
    model = DeepONet(
        MLP(model_params["branch_input_dim"], model_params["branch_hidden_dims"], model_params["latent_dim"]),
        MLP(model_params["trunk_input_dim"],  model_params["trunk_hidden_dims"],  model_params["latent_dim"])
    ).to(device)
    model.load_state_dict(model_state)
    model.eval()

    param_names = branch_df.columns.tolist()
    param_mins  = branch_df.min().values
    param_maxs  = branch_df.max().values
    param_bounds = list(zip(param_mins, param_maxs))

    params_to_modify = [
        'v1_3','v2_3','v3_3','v4_3','v5_3','v6_3',
        'v1_4','v2_4','v3_4','v4_4','v5_4','v6_4',
    ]
    for p in params_to_modify:
        if p in param_names:
            i = param_names.index(p)
            lo, hi = param_bounds[i]
            param_bounds[i] = (lo, hi + 50.0)
            print(f" -> '{p}' (index {i}) bounds: ({lo:.2f}, {hi:.2f}) -> ({lo:.2f}, {hi+50.0:.2f})")

    LEGACY_STATE.update({
        "initialized": True,
        "output_dir": output_dir,
        "assets": {
            "branch_scaler": assets["branch_scaler"],
            "target_scaler": assets["target_scaler"],
            "trunk_tensor": assets["trunk_tensor"].to(device),
            "feature_weights": assets["feature_weights"],
            "seed_ids": seed_ids,
        },
        "dataframes": {"branch": branch_df, "trunk": trunk_df},
        "model": model,
        "param_names": param_names,
        "param_bounds": param_bounds,
        "indices_template": _build_indices_template(param_names, param_bounds),
    })


def _apply_dynamic_constraints(indices_map, dynamic_constraints, param_names):
    get_indices = lambda n: [param_names.index(n)] if isinstance(n,str) else [param_names.index(x) for x in n]
    
    for grp in dynamic_constraints.get('add_unconditional', []) or []:
        indices_map['unconditional'].append(get_indices(grp))
    for a,b in dynamic_constraints.get('must_equal', []) or []:
        ia = get_indices(a)[0]; ib = get_indices(b)[0]
        indices_map['must_equal_pairs'].append((ia, ib))
    for a,b in dynamic_constraints.get('not_equal', []) or []:
        ia = get_indices(a)[0]; ib = get_indices(b)[0]
        indices_map['not_equal_pairs'].append((ia, ib))
    indices_map['fixed_zero'].extend(get_indices(dynamic_constraints.get('fixed_zero', []) or []))
    indices_map['non_zero'].extend(get_indices(dynamic_constraints.get('non_zero', []) or []))
    for name, val in (dynamic_constraints.get('fixed_value', {}) or {}).items():
        idx = get_indices(name)[0]
        indices_map['fixed_value'][idx] = float(val)
    
    if 'upper_bound' in dynamic_constraints:
        for name, limit in dynamic_constraints['upper_bound'].items():
                try:
                    idx = get_indices([name])[0]
                    indices_map['upper_bound'][idx] = limit
                except ValueError:
                    print(f"[경고] 'upper_bound'의 파라미터 '{name}'를 찾을 수 없습니다.")
                    pass


# ==================================================================
# PART 1: CAD 배경 생성 함수
# ==================================================================
def create_cad_background_and_get_bounds(step_path, save_path):
    if not OCC_AVAILABLE: return None, None
    if not os.path.exists(step_path):
        print(f"STEP 파일을 찾을 수 없습니다: {step_path}")
        return None, None
        
    reader = STEPControl_Reader()
    if reader.ReadFile(step_path) != 1:
        print("STEP 파일 읽기 실패.")
        return None, None
    reader.TransferRoots()
    shape = reader.OneShape()
    print("STEP 파일 로드 성공!")

    bbox = Bnd_Box()
    brepbndlib.Add(shape, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    cad_extent = (xmin, xmax, ymin, ymax)
    print(f"계산된 좌표 범위 (Extent): X [{xmin:.2f}, {xmax:.2f}], Y [{ymin:.2f}, {ymax:.2f}]")

    # --- [수정된 부분] ---
    # 1. CAD 모델의 실제 물리적 가로(X) / 세로(Y) 폭을 계산합니다.
    physical_width = xmax - xmin
    physical_height = ymax - ymin
    
    # 2. 기준이 될 픽셀 폭을 정합니다 (예: 1920)
    base_pixel_width = 1920
    
    # 3. 물리적 비율에 맞춰 픽셀 높이를 계산합니다.
    # (혹시 physical_width가 0이 되어도 오류가 나지 않도록 1e-6을 더함)
    aspect_ratio = physical_height / (physical_width + 1e-6)
    target_pixel_height = int(base_pixel_width * aspect_ratio)

    # 4. 계산된 크기가 유효한지 확인 (너무 작거나 0이 되는 것 방지)
    if target_pixel_height <= 0 or base_pixel_width <= 0:
        print(f"[경고] 바운딩 박스 크기 계산 오류. (1920, 1080) 기본값 사용.")
        base_pixel_width, target_pixel_height = 1920, 1080
    else:
        print(f"모델 비율에 맞춘 새 캔버스 크기: ({base_pixel_width}, {target_pixel_height})")

    # 5. 고정된 (1920, 1080) 대신, 계산된 캔버스 크기를 사용합니다.
    display, _, _, _ = init_display(size=(base_pixel_width, target_pixel_height), display_triedron=False)
    # --- [수정 완료] ---

    display.EraseAll()
    display.set_bg_gradient_color([255,255,255], [255,255,255]) 

    custom_dark_gray = Quantity_Color(0.03, 0.03, 0.03, Quantity_TOC_RGB)
    display.DisplayShape(shape, update=True, color=custom_dark_gray)

    display.View.SetProj(0, 0, -1)
    display.View.SetUp(0, 1, 0)   
    display.View.FitAll(0.0)  # <-- 'display.FitAll()'을 'display.View.FitAll(0.0)'로 수정
    display.View.Dump(save_path)
    print(f"CAD 배경 이미지 저장 완료: {save_path}")

    return save_path, cad_extent

# ==================================================================
# PART 2: PI-DeepONet 및 유전 알고리즘 핵심 로직
# ==================================================================
class MLP(nn.Module):
    def __init__(self, in_dim, h_dims, out_dim, dropout=0.1):
        super().__init__()
        layers, prev = [], in_dim
        for h in h_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class DeepONet(nn.Module):
    def __init__(self, b_net, t_net):
        super().__init__()
        self.branch_net, self.trunk_net  = b_net, t_net
    def forward(self, xb, xt): return torch.matmul(self.branch_net(xb), self.trunk_net(xt).transpose(0,1))

def evaluate_design(individual, indices_map):
    state = _state()
    branch_df = state["dataframes"]["branch"]
    branch_scaler = state["assets"]["branch_scaler"]
    target_scaler = state["assets"]["target_scaler"]
    trunk_tensor = state["assets"]["trunk_tensor"]
    feature_weights = state["assets"]["feature_weights"]
    model = state["model"]
    param_names = state["param_names"]

    TOL, PENALTY = 1e-6, 10.0
    for idx in indices_map.get('non_zero', []):
        if abs(individual[idx]) < TOL: return (PENALTY,)
    for i, j in indices_map.get('not_equal_pairs', []):
        if abs(individual[i] - individual[j]) < TOL: return (PENALTY,)
    physical_df = pd.DataFrame([individual], columns=param_names)
    scaled_values = branch_scaler.transform(physical_df.values)
    scaled_df = pd.DataFrame(scaled_values, columns=param_names)
    for feat, w in feature_weights.items():
        if feat in scaled_df.columns: scaled_df[feat] *= w
    branch_input = torch.tensor(scaled_df.values, dtype=torch.float32).to(device)
    with torch.no_grad():
        scaled_pred = model(branch_input, trunk_tensor)
    physical_pred = target_scaler.inverse_transform(scaled_pred.cpu().numpy())
    return (np.max(np.abs(physical_pred)),)

def predict_full_field(individual):
    state = _state()
    branch_df = state["dataframes"]["branch"]
    branch_scaler = state["assets"]["branch_scaler"]
    target_scaler = state["assets"]["target_scaler"]
    trunk_tensor = state["assets"]["trunk_tensor"]
    feature_weights = state["assets"]["feature_weights"]
    model = state["model"]
    param_names = state["param_names"]

    physical_df = pd.DataFrame([individual], columns=param_names)
    scaled_values = branch_scaler.transform(physical_df.values)
    scaled_df = pd.DataFrame(scaled_values, columns=param_names)
    for feat, w in feature_weights.items():
        if feat in scaled_df.columns: scaled_df[feat] *= w
    branch_input = torch.tensor(scaled_df.values, dtype=torch.float32).to(device)
    with torch.no_grad():
        scaled_pred = model(branch_input, trunk_tensor)
    return target_scaler.inverse_transform(scaled_pred.cpu().numpy()).flatten()

def mutate_adaptive_sigma(individual, bounds, mu, sigma_ratio, indpb):
    for i in range(len(individual)):
        if random.random() < indpb:
            min_val, max_val = bounds[i]
            range_width = max_val - min_val
            sigma_i = range_width * sigma_ratio if range_width > 0 else 0.1
            individual[i] += random.gauss(mu, sigma_i)
    return individual,

def enforce_constraints(individual, indices_map):
    for group in indices_map.get('unconditional', []):
        master_value = individual[group[0]]
        for i in group[1:]: individual[i] = master_value
        
    for condition in indices_map.get('conditional', []):
        if abs(individual[condition['if'][0]] - individual[condition['if'][1]]) < 1e-6:
            for action in condition['then']: individual[action[1]] = individual[action[0]]
            
    for i, j in indices_map.get('must_equal_pairs', []): individual[j] = individual[i]
    
    for idx in indices_map.get('fixed_zero', []): individual[idx] = 0.0
    
    for idx, limit in indices_map.get('upper_bound', {}).items():
        if individual[idx] > limit:
            individual[idx] = limit
            
    for idx, val in indices_map.get('fixed_value', {}).items():
        individual[idx] = val
        
    for down_idx, up_idx in indices_map.get('greater_or_equal_pairs', []):
        if individual[down_idx] < individual[up_idx]:
            individual[down_idx] = individual[up_idx]

    for i in range(90):
        if i in indices_map.get('fixed_value', {}):
            continue
        if individual[i] < 10:
            individual[i] = 0.0
    # ==========================================================

    param_names_list = LEGACY_STATE["param_names"]
    if 'rev_forming1' in param_names_list:
        rev_idx = param_names_list.index('rev_forming1')
        current_val, valid_options = individual[rev_idx], [1.0, 3.0, 5.0]
        individual[rev_idx] = min(valid_options, key=lambda x: abs(x - current_val))
    return individual

# ==================================================================
# PART 3: 시각화 및 검증 함수
# ==================================================================
def visualize_optimal_result(individual, trunk_dataframe, output_dir=None):
    state = _state()
    output_dir = output_dir or state["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n--- Generating Contour Plot for the Optimal Result ---")
    predicted_displacement = predict_full_field(individual)
    x_coords, y_coords = trunk_dataframe["X"].values, trunk_dataframe["Y"].values
    
    # ==========================================================
    # [추가] 7200개 좌표 변위 값을 CSV 파일로 저장하는 로직
    # ==========================================================
    # 1. X, Y, Displacement 데이터를 DataFrame으로 만듭니다.
    displacement_df = pd.DataFrame({
        'X_Coordinate': x_coords,
        'Y_Coordinate': y_coords,
        'Predicted_Displacement': predicted_displacement
    })

    # 2. 저장할 파일 경로를 지정합니다.
    csv_save_path = os.path.join(output_dir, "optimal_displacement_data_grav.csv")

    # 3. CSV 파일로 저장합니다.
    displacement_df.to_csv(csv_save_path, index=False, encoding='utf-8-sig')
    print(f"7200개 좌표 변위 값 저장 완료: {csv_save_path}")
    # ==========================================================
    
    colors_list = [(0/255,0/255,200/255),(21/255,121/255,255/255),(0/255,199/255,221/255),(40/255,255/255,185/255),(57/255,255/255,0/255),(170/255,255/255,0/255),(255/255,227/255,0/255),(255/255,113/255,0/255),(255/255,0/255,0/255)]
    custom_cmap, vmin, vmax = ListedColormap(colors_list), np.min(predicted_displacement), np.max(predicted_displacement)
    boundaries = np.linspace(vmin, vmax, 10)
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    cp = ax.tricontourf(x_coords, y_coords, predicted_displacement, levels=boundaries, cmap=custom_cmap, norm=BoundaryNorm(boundaries, ncolors=len(colors_list), clip=True))
    idx_max = np.argmax(np.abs(predicted_displacement))
    max_disp_val = predicted_displacement[idx_max]
    ax.plot(x_coords[idx_max], y_coords[idx_max], 'o', markersize=12, markerfacecolor='none', markeredgecolor='white', markeredgewidth=2, label=f'Max Abs Disp: {abs(max_disp_val):.4f} at ({x_coords[idx_max]:.1f}, {y_coords[idx_max]:.1f})')
    ax.legend()
    ax.set_title("Predicted Displacement Contour for Optimal Design (Gravity)", fontsize=16)
    ax.set_xlabel("X (mm)"), ax.set_ylabel("Y (mm)"), ax.set_aspect("equal")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(cp, cax=cax, format=ticker.FormatStrFormatter('%.3f'), label="Displacement (mm)")
    plt.tight_layout()
    save_path = os.path.join(output_dir, "optimal_design_contour_grav.png")
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Contour plot saved to: {save_path}")

def visualize_normalized_scatter(optimal_params, param_bounds, param_names, output_dir=None):
    state = _state()
    output_dir = output_dir or state["output_dir"]

    print("\n--- Generating Normalized Optimal Parameter Scatter Plots ---")
    optimal_params_array, param_bounds_array = np.array(optimal_params), np.array(param_bounds)
    min_bounds, max_bounds = param_bounds_array[:, 0], param_bounds_array[:, 1]
    range_widths = max_bounds - min_bounds
    range_widths[range_widths == 0] = 1.0 
    normalized_values = (optimal_params_array - min_bounds) / range_widths * 100.0
    param_names_array = np.array(param_names)
    path1 = os.path.join(output_dir, "optimal_normalized_scatter_grav_1-45.png")
    path2 = os.path.join(output_dir, "optimal_normalized_scatter_grav_46-95.png")
    fig1, ax1 = plt.subplots(figsize=(10, 15))
    ax1.scatter(normalized_values[:45], param_names_array[:45], color='dodgerblue', alpha=0.7, label='Optimal Value')
    ax1.set_title('Normalized Optimal Parameters (1-45)', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Position within Range (%) [0=Min, 100=Max]'), ax1.set_ylabel('Parameter Name'), ax1.set_xlim(-5, 105)
    ax1.axvline(x=0,c='r',ls='--',lw=1,label='Min/Max Bounds'), ax1.axvline(x=100,c='r',ls='--',lw=1), ax1.axvline(x=50,c='g',ls=':',lw=1,label='Center')
    ax1.grid(axis='x',ls='--',alpha=0.6), ax1.legend(), ax1.invert_yaxis(), plt.tight_layout()
    plt.savefig(path1, dpi=150)
    plt.close(fig1)
    fig2, ax2 = plt.subplots(figsize=(10, 15))
    ax2.scatter(normalized_values[45:], param_names_array[45:], color='dodgerblue', alpha=0.7, label='Optimal Value')
    ax2.set_title('Normalized Optimal Parameters (46-95)', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Position within Range (%) [0=Min, 100=Max]'), ax2.set_ylabel('Parameter Name'), ax2.set_xlim(-5, 105)
    ax2.axvline(x=0,c='r',ls='--',lw=1,label='Min/Max Bounds'), ax2.axvline(x=100,c='r',ls='--',lw=1), ax2.axvline(x=50,c='g',ls=':',lw=1,label='Center')
    ax2.grid(axis='x',ls='--',alpha=0.6), ax2.legend(), ax2.invert_yaxis(), plt.tight_layout()
    plt.savefig(path2, dpi=150)
    plt.close(fig2)
    print(f"Normalized parameter scatter plots saved to '{output_dir}' folder.")

# --- 이 함수는 visualize_overlay_final 내부에서 사용됩니다 ---
def evaluate_expression_final(expr_str, p_obj, local_vars):
    """(수정) 파라미터 평가를 위한 도우미 함수"""
    expr_str = str(expr_str)
    for name, value in local_vars.items():
        expr_str = expr_str.replace(name, str(value))
    
    param_names = re.findall(r'[a-zA-Z0-9_ ]+', expr_str)
    for name in sorted(param_names, key=len, reverse=True):
        lookup_name = name.strip().replace(' ', '_')
        if hasattr(p_obj, lookup_name):
            value = getattr(p_obj, lookup_name)
            expr_str = expr_str.replace(name, f"({value})")
    try:
        return pd.eval(expr_str)
    except Exception:
        return None

# --- 기존의 모든 시각화 함수를 이 함수 하나로 교체합니다 ---
def visualize_overlay_final(shape_parameters, background_image_path, cad_extent, dynamic_constraints, output_dir=None):
    
    state = _state()
    output_dir = output_dir or state["output_dir"]
    
    print("\n--- Generating Final Overlay Image (NumPy Mask Method) ---")
    if not os.path.exists(background_image_path) or cad_extent is None:
        print("경고: 배경 이미지 또는 좌표가 없어 오버레이를 생성할 수 없습니다.")
        return

    # 1. 배경 CAD 이미지를 NumPy 배열로 로드
    background_img = plt.imread(background_image_path)
    img_height, img_width, _ = background_img.shape

    # 2. 파라미터 준비 (기존과 동일)
    param_dict = shape_parameters.to_dict()
    clean_param_dict = {k.replace(' ', '_'): v for k, v in param_dict.items()}
    p = types.SimpleNamespace(**clean_param_dict)
    
    # --- [수정] 2. y_min_cad와 y_max_cad를 JSON에서 로드 ---
    # 2-1. x_min_cad, x_max_cad는 고정값 유지
    x_min_cad, x_max_cad = -579, 579

    # 2-2. constraints.json에서 'h1_p_components'를 가져옴
    #      (존재하지 않을 경우 빈 딕셔너리 {} 반환)
    h1p_components = dynamic_constraints.get('h1_p_components', {})

    if 'base' in h1p_components:
        y_min_cad = h1p_components['base']
        print(f"[알림] constraints.json의 'base' 값({y_min_cad:.2f})을 y_min_cad로 설정합니다.")
    else:
        y_min_cad = -314  # 'base'가 없으면 기존 고정값
        print(f"[경고] constraints.json에 'base' 값이 없습니다. y_min_cad를 기본값 {y_min_cad}로 설정합니다.")

    # 2-4. y_max_cad (y) 로드
    if 'y' in h1p_components:
        y_max_cad = h1p_components['y']
        print(f"[알림] constraints.json의 'y' 값({y_max_cad:.2f})을 y_max_cad로 설정합니다.")
    else:
        # --- [수정된 폴백 로직] ---
        # 'y'가 없으면, 고정값 248 대신 GA 결과를 사용
        print(f"[알림] constraints.json에 'y' 값이 없습니다. 최적화된 h1_p 값을 사용합니다.")
        try:
            # 'p' 객체 (GA 최적화 결과)에서 'h1_p' 값을 직접 가져옴
            optimal_h1_p_value = p.h1_p 
            
            # y_max_cad = h1_p + base
            y_max_cad = optimal_h1_p_value + y_min_cad 
            
            print(f"[알림] 최적 h1_p({optimal_h1_p_value:.2f}) + base({y_min_cad:.2f}) = y_max_cad({y_max_cad:.2f})로 설정합니다.")
        
        except AttributeError:
            # 'h1_p' 파라미터가 GA 결과에 없는 비상 상황
            y_max_cad = y_min_cad + 562 # 임시 고정값 (과거 h1_p 기본값)
            print(f"[오류] 'h1_p' 파라미터를 최적화 결과에서 찾을 수 없습니다. y_max_cad를 임시값 {y_max_cad:.2f}로 설정합니다.")
        except Exception as e:
            y_max_cad = y_min_cad + 562
            print(f"[오류] y_max_cad 계산 중 오류 발생: {e}. y_max_cad를 임시값 {y_max_cad:.2f}로 설정합니다.")
        # --- [수정 완료] ---
    
    # evaluate_expression_final 함수가 사용할 'y_min' 등의 변수를 정의합니다.
    local_vars = {'x_min': x_min_cad, 'x_max': x_max_cad, 'y_min': y_min_cad}

    # 3. 초록색 오버레이 레이어 및 '마스터 마스크' 생성 (기존과 동일)
    overlay_image = np.zeros((img_height, img_width, 4), dtype=np.float32)
    cutout_mask = np.full((img_height, img_width), False, dtype=bool)

    # --- [수정된 로직 1] ---
    # 4. '지도(extent)'에서 좌표를 풀고, '변환 비율(p)'을 미리 계산합니다.
    xmin, xmax, ymin, ymax = cad_extent
    
    # 4-1. X축, Y축의 '변환 비율 p' (px/mm)를 계산합니다.
    #      (p_x = 1920 / (610.91 - (-610.91)))
    p_x = img_width / (xmax - xmin + 1e-6)
    p_y = img_height / (ymax - ymin + 1e-6)
    
    # 5. CAD 좌표(mm)를 이미지 픽셀(px)로 변환하는 '새로운' 함수 정의
    def cad_to_pixel(x, y):
        px_raw = (x - xmin) * p_x
        px = px_raw
        py_raw = (y - ymin) * p_y
        py = img_height - py_raw
        return int(round(px)), int(round(py))

    # [수정] 3-3. 전체 설계 영역만 초록색으로 칠합니다.
    r, g, b = 84, 240, 84
    green_rgba = (r/255, g/255, b/255, 0.6)
    px_main_left, py_main_top = cad_to_pixel(x_min_cad, y_max_cad)
    px_main_right, py_main_bottom = cad_to_pixel(x_max_cad, y_min_cad)
    overlay_image[py_main_top:py_main_bottom, px_main_left:px_main_right] = green_rgba

    # 5. 모든 cutout 정의를 순회하며 마스크에 'True'로 칠하기
    cutout_definitions = [
        ['-v1_1_p', '-v2_1_p-v2_1', 'h2_1_up', 'h1_1_down' ],
        ['-v2_1_p', '-v3_1_p-v3_1', 'h2_2_up', 'h1_2_down'],
        ['-v3_1_p', 'v4_1_p', 'h2_3_up', 'h1_3_down' ],
        ['v4_1_p+v4_1', 'v5_1_p','h2_4_up', 'h1_4_down' ],
        ['v5_1_p+v5_1', 'v6_1_p','h2_5_up', 'h1_5_down' ],
        ['-v1_2_p', '-v2_2_p-v2_2', '-h3_1_up', 'h2_1_down' ],
        ['-v2_2_p', '-v3_2_p-v3_2', '-h3_2_up', 'h2_2_down'],
        ['-v3_2_p', 'v4_2_p', '-h3_3_up', 'h2_3_down' ],
        ['v4_2_p+v4_2', 'v5_2_p','-h3_4_up', 'h2_4_down' ],
        ['v5_2_p+v5_2', 'v6_2_p','-h3_5_up', 'h2_5_down' ],
        ['-v1_3_p', '-v2_3_p-v2_3', '-h4_1_up', '-h3_1_down' ],
        ['-v2_3_p', '-v3_3_p-v3_3', '-h4_2_up', '-h3_2_down'],
        ['-v3_3_p', 'v4_3_p', '-h4_3_up', '-h3_3_down' ],
        ['v4_3_p+v4_3', 'v5_3_p', '-h4_4_up', '-h3_4_down' ],
        ['v5_3_p+v5_3', 'v6_3_p','-h4_5_up', '-h3_5_down' ],
        ['-v1_4_p', '-v2_4_p-v2_4', 'y_min', '-h4_1_down' ],
        ['-v2_4_p', '-v3_4_p-v3_4-20', 'y_min', '-h4_2_down'],
        ['-v3_4_p', 'v4_4_p', 'y_min', '-h4_3_down' ],
        ['v4_4_p+v4_4+20', 'v5_4_p', 'y_min', '-h4_4_down' ],
        ['v5_4_p+v5_4', 'v6_4_p','y_min', '-h4_5_down' ],
        ['-v1_4_p-v1_4', '-v1_4_p', 'y_min', 'y_min+h5_1'],
        ['-v2_4-v2_4_p', '-v2_4_p',  'y_min', 'y_min+h5_2'],
        ['-v3_4-v3_4_p', '-v3_4_p',  'y_min', 'y_min+h5_3'],
        ['v4_4_p', 'v4_4_p+v4_4',  'y_min', 'y_min+h5_4'],
        ['v5_4_p', 'v5_4_p+v5_4',  'y_min', 'y_min+h5_5'],
        ['v6_4_p', 'v6_4_p+v6_4',  'y_min', 'y_min+h5_6'],
        ['-v3_4_p-v3_4-20', '-v3_4_p-v3_4',  'y_min', '-h4_2_down-20'],
        ['v4_4_p+v4_4', 'v4_4_p+v4_4+20',  'y_min', '-h4_4_down-20'],
        ['x_min', '-v1_4-v1_4_p', 'y_min', 'y_min+h5_1'],
        ['x_min', 'x_min+12', 'y_min+h5_1', '-h4_1_up'],
        ['x_min', 'x_min+12', '-h4_1_up', '-h3_1_down'],
        ['x_min', 'x_min+12', '-h3_1_down', '-h3_1_up'],
        ['x_min', '-v1_2-v1_2_p', '-h3_1_up', 'h2_1_down'],
        ['x_min', 'x_min+12', 'h2_1_down', 'h2_1_up'],
        ['x_min', 'x_min+12', 'h2_1_up', 'h1_1_down'],
        ['x_min', 'x_min+12', 'h1_1_down', y_max_cad],
        ['v6_4+v6_4_p', x_max_cad, 'y_min', 'y_min+h5_6'],
        [f'{x_max_cad}-12', f'{x_max_cad}', 'y_min+h5_6', '-h4_5_up'],
        [f'{x_max_cad}-12', f'{x_max_cad}', '-h4_5_up', '-h3_5_down'],
        [f'{x_max_cad}-12', f'{x_max_cad}', '-h3_5_down', '-h3_5_up'],
        ['v6_2+v6_2_p', x_max_cad, '-h3_5_up', 'h2_5_down'],
        [f'{x_max_cad}-12', f'{x_max_cad}', 'h2_5_down', 'h2_5_up'],
        [f'{x_max_cad}-12', f'{x_max_cad}', 'h2_5_up', 'h1_5_down'],
        [f'{x_max_cad}-12', f'{x_max_cad}', 'h1_5_down', y_max_cad],
    ]
    try:
        rev = int(getattr(p, 'rev_forming1', 0))
        # --- [수정] y_bottom_expr 수식에서 y_min_cad 사용 ---
        y_bottom_expr = 'h1_p + y_min - 13 - forming_width' # 314 대신 y_min 사용
        y_top_expr = 'h1_p + y_min - 13'                   # 314 대신 y_min 사용
        # --- [수정 완료] ---
        if rev == 1: cutout_definitions.extend([['-515.25', '515.25', y_bottom_expr, y_top_expr]])
        elif rev == 3: cutout_definitions.extend([['-515.25', '-186.45', y_bottom_expr, y_top_expr], ['-107.25', '107.25', y_bottom_expr, y_top_expr], ['186.45', '515.25', y_bottom_expr, y_top_expr]])
        elif rev == 5: cutout_definitions.extend([['-515.25', '-369.75', y_bottom_expr, y_top_expr], ['-331.25', '-145.75', y_bottom_expr, y_top_expr], ['-107.25', '107.25', y_bottom_expr, y_top_expr], ['145.75', '331.25', y_bottom_expr, y_top_expr], ['369.75', '515.25', y_bottom_expr, y_top_expr]])
    except Exception: pass
    
    for definition in cutout_definitions:
        x_left = evaluate_expression_final(definition[0], p, local_vars)
        x_right = evaluate_expression_final(definition[1], p, local_vars)
        y_bottom = evaluate_expression_final(definition[2], p, local_vars)
        y_top = evaluate_expression_final(definition[3], p, local_vars)

        if all(v is not None for v in [x_left, x_right, y_bottom, y_top]):
            if (x_right - x_left > 0) and (y_top - y_bottom > 0):
                px_left, py_top = cad_to_pixel(x_left, y_top)
                px_right, py_bottom = cad_to_pixel(x_right, y_bottom)
                cutout_mask[py_top:py_bottom, px_left:px_right] = True

    # 6. 완성된 마스크를 사용해 오버레이 이미지에 구멍 뚫기(투명하게 만들기)
    overlay_image[cutout_mask, 3] = 0.0

    # 7. 최종 결과 시각화 및 저장
    fig, ax = plt.subplots(figsize=(12, (12 / img_width) * img_height))
    ax.imshow(background_img, extent=cad_extent)
    ax.imshow(overlay_image, extent=cad_extent)
    ax.set_aspect('equal', 'box')
    ax.axis('off')
    plt.tight_layout(pad=0)
    save_path = os.path.join(output_dir, "Optimal_Design_Overlay.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)
    print(f"Final overlay image saved to: {save_path}")


def verify_dynamic_constraints(best_individual, dynamic_constraints, param_names):
    print("\n--- 최종 동적 제약 조건 검증 ---")
    if not dynamic_constraints or not any(dynamic_constraints.values()):
        print(" -> 적용된 동적 제약 조건이 없습니다.")
        return
    def get_idx_val(name):
        idx = param_names.index(name)
        return idx, best_individual[idx]
    if dynamic_constraints.get('must_equal'):
        print("[같아야 하는 쌍 (must_equal)]")
        for a, b in dynamic_constraints['must_equal']:
            _, va = get_idx_val(a); _, vb = get_idx_val(b)
            print(f" - {a} == {b} | ({va:.4f}, {vb:.4f}) -> {'[만족]' if abs(va - vb) < 1e-6 else '[불만족]'}")
    if dynamic_constraints.get('not_equal'):
        print("\n[같으면 안 되는 쌍 (not_equal)]")
        for a, b in dynamic_constraints['not_equal']:
            _, va = get_idx_val(a); _, vb = get_idx_val(b)
            print(f" - {a} != {b} | ({va:.4f}, {vb:.4f}) -> {'[만족]' if abs(va - vb) >= 1e-6 else '[불만족]'}")
    if dynamic_constraints.get('fixed_zero'):
        print("\n[0 고정 조건]")
        for name in dynamic_constraints['fixed_zero']:
            _, v = get_idx_val(name)
            print(f" - {name} == 0 | {v:.4f} -> {'[만족]' if abs(v) < 1e-6 else '[불만족]'}")
    if dynamic_constraints.get('non_zero'):
        print("\n[0 방지 조건]")
        for name in dynamic_constraints['non_zero']:
            _, v = get_idx_val(name)
            print(f" - {name} != 0 | {v:.4f} -> {'[만족]' if abs(v) >= 1e-6 else '[불만족]'}")



def _build_indices_template(param_names, param_bounds):
    def get_indices(names):
        if isinstance(names, str):
            names = [names]
        return [param_names.index(n) for n in names]

    template = {
        "unconditional": [
            get_indices(["h1_1_down","h1_2_down","h1_3_down","h1_4_down","h1_5_down"]),
            get_indices(["h2_1_down","h2_2_down","h2_3_down","h2_4_down","h2_5_down"]),
            get_indices(["h3_1_down","h3_2_down","h3_3_down","h3_4_down","h3_5_down"]),
        ],
        "conditional": [ {'if':get_indices(['h2_1_up','h2_1_down']),'then':[(get_indices('v1_1_p')[0],get_indices('v1_2_p')[0]),(get_indices('v2_1_p')[0],get_indices('v2_2_p')[0]),(get_indices('v2_1')[0],get_indices('v2_2')[0])]}, {'if':get_indices(['h3_1_up','h3_1_down']),'then':[(get_indices('v1_2_p')[0],get_indices('v1_3_p')[0]),(get_indices('v2_2_p')[0],get_indices('v2_3_p')[0]),(get_indices('v2_2')[0],get_indices('v2_3')[0])]}, {'if':get_indices(['h4_1_up','h4_1_down']),'then':[(get_indices('v1_3_p')[0],get_indices('v1_4_p')[0]),(get_indices('v2_3_p')[0],get_indices('v2_4_p')[0]),(get_indices('v2_3')[0],get_indices('v2_4')[0])]}, {'if':get_indices(['h2_2_up','h2_2_down']),'then':[(get_indices('v2_1_p')[0],get_indices('v2_2_p')[0]),(get_indices('v3_1_p')[0],get_indices('v3_2_p')[0]),(get_indices('v3_1')[0],get_indices('v3_2')[0])]}, {'if':get_indices(['h3_2_up','h3_2_down']),'then':[(get_indices('v2_2_p')[0],get_indices('v2_3_p')[0]),(get_indices('v3_2_p')[0],get_indices('v3_3_p')[0]),(get_indices('v3_2')[0],get_indices('v3_3')[0])]}, {'if':get_indices(['h4_2_up','h4_2_down']),'then':[(get_indices('v2_3_p')[0],get_indices('v2_4_p')[0])]}, {'if':get_indices(['h2_3_up','h2_3_down']),'then':[(get_indices('v3_1_p')[0],get_indices('v3_2_p')[0]),(get_indices('v4_1_p')[0],get_indices('v4_2_p')[0])]}, {'if':get_indices(['h3_3_up','h3_3_down']),'then':[(get_indices('v3_2_p')[0],get_indices('v3_3_p')[0]),(get_indices('v4_2_p')[0],get_indices('v4_3_p')[0])]}, {'if':get_indices(['h4_3_up','h4_3_down']),'then':[(get_indices('v3_3_p')[0],get_indices('v3_4_p')[0]),(get_indices('v4_3_p')[0],get_indices('v4_4_p')[0])]}, {'if':get_indices(['h2_4_up','h2_4_down']),'then':[(get_indices('v5_1_p')[0],get_indices('v5_2_p')[0]),(get_indices('v4_1_p')[0],get_indices('v4_2_p')[0]),(get_indices('v4_1')[0],get_indices('v4_2')[0])]}, {'if':get_indices(['h3_4_up','h3_4_down']),'then':[(get_indices('v5_2_p')[0],get_indices('v5_3_p')[0]),(get_indices('v4_2_p')[0],get_indices('v4_3_p')[0]),(get_indices('v4_2')[0],get_indices('v4_3')[0])]}, {'if':get_indices(['h4_4_up','h4_4_down']),'then':[(get_indices('v5_3_p')[0],get_indices('v5_4_p')[0])]}, {'if':get_indices(['h2_5_up','h2_5_down']),'then':[(get_indices('v5_1_p')[0],get_indices('v5_2_p')[0]),(get_indices('v5_1')[0],get_indices('v5_2')[0]),(get_indices('v6_1_p')[0],get_indices('v6_2_p')[0])]}, {'if':get_indices(['h3_5_up','h3_5_down']),'then':[(get_indices('v5_2_p')[0],get_indices('v5_3_p')[0]),(get_indices('v5_2')[0],get_indices('v5_3')[0]),(get_indices('v6_2_p')[0],get_indices('v6_3_p')[0])]}, {'if':get_indices(['h4_5_up','h4_5_down']),'then':[(get_indices('v6_3_p')[0],get_indices('v6_4_p')[0]),(get_indices('v5_3_p')[0],get_indices('v5_4_p')[0]),(get_indices('v5_3')[0],get_indices('v5_4')[0])]}
        ],
        "fixed_zero": [],
        "non_zero": [],
        "fixed_value": {},
        "must_equal_pairs": [],
        "not_equal_pairs": [],
        "upper_bound": {}
    }

    try:
        template["greater_or_equal_pairs"] = [
            (param_names.index("h4_1_down"), param_names.index("h4_1_up")),
            (param_names.index("h4_2_down"), param_names.index("h4_2_up")),
            (param_names.index("h4_3_down"), param_names.index("h4_3_up")),
            (param_names.index("h4_4_down"), param_names.index("h4_4_up")),
            (param_names.index("h4_5_down"), param_names.index("h4_5_up")),
        ]
    except ValueError as e:
        print(f"[경고] greater_or_equal_pairs 구성 실패: {e}")

    return template


def run_legacy_ga(
    dynamic_constraints: dict,
    *,
    ga_opts: dict | None = None,
    cad_context: dict | None = None,
    return_base64: bool = False,
):
    if not LEGACY_STATE["initialized"]:
        raise RuntimeError("initialize_legacy_pipeline 호출 후 사용하세요.")

    _ensure_creator_types()
    state = LEGACY_STATE
    device = state["device"]
    branch_df = state["dataframes"]["branch"]
    trunk_df  = state["dataframes"]["trunk"]
    param_names = state["param_names"]
    param_bounds = copy.deepcopy(state["param_bounds"])
    output_dir = state["output_dir"]

    # GA 옵션 기본값
    opts = {
        "max_generations": 100,
        "p_crossover": 0.9,
        "p_mutation": 0.2,
        "num_random_individuals": 300,
        "early_stop_patience": 10,
        "mut_mu": 0.0,
        "mut_sigma_ratio": 0.2,
        "mut_indpb": 0.1,
    }
    if ga_opts:
        opts.update({k: v for k, v in ga_opts.items() if v is not None})

    indices_map = copy.deepcopy(state["indices_template"])
    _apply_dynamic_constraints(indices_map, dynamic_constraints, param_names)

    toolbox = base.Toolbox()
    IND_SIZE = len(param_bounds)
    for i, (lo, hi) in enumerate(param_bounds):
        toolbox.register(f"attr_{i}", random.uniform, lo, hi)
    attrs = tuple(getattr(toolbox, f"attr_{i}") for i in range(IND_SIZE))
    toolbox.register("individual", tools.initCycle, creator.Individual, attrs)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mutate_adaptive_sigma,
                     bounds=param_bounds, mu=opts["mut_mu"],
                     sigma_ratio=opts["mut_sigma_ratio"], indpb=opts["mut_indpb"])
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate_design, indices_map=indices_map)

    seed_ids = state["assets"]["seed_ids"]
    pop = [creator.Individual(branch_df.loc[sid].values.tolist()) for sid in seed_ids]
    pop += toolbox.population(n=opts["num_random_individuals"])
    for ind in pop:
        enforce_constraints(ind, indices_map)
    random.shuffle(pop)

    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    best_fitness = float("inf")
    patience = 0
    for g in range(1, opts["max_generations"] + 1):
        offspring = list(map(toolbox.clone, toolbox.select(pop, len(pop))))
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < opts["p_crossover"]:
                toolbox.mate(child1, child2)
                enforce_constraints(child1, indices_map)
                enforce_constraints(child2, indices_map)
                del child1.fitness.values, child2.fitness.values
        for mutant in offspring:
            if random.random() < opts["p_mutation"]:
                toolbox.mutate(mutant)
                enforce_constraints(mutant, indices_map)
                del mutant.fitness.values
        
        for ind in offspring:
            for i in range(IND_SIZE):
                # [수정] fixed_value가 적용된 인자는 범위 검사를 건너뜁니다.
                if i in indices_map['fixed_value']:
                    continue
                
                min_val, max_val = param_bounds[i]
                if ind[i] < min_val: ind[i] = min_val
                elif ind[i] > max_val: ind[i] = max_val

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        for ind, fit in zip(invalid_ind, map(toolbox.evaluate, invalid_ind)): ind.fitness.values = fit
        
        pop[:] = offspring
        hof.update(pop)
        record = stats.compile(pop)

        print(f"GEN: {g}, Min_Fitness: {record['min']:.6f}, Avg_Fitness: {record['avg']:.6f}")
        if record['min'] < best_fitness:
            best_fitness = record['min']
            patience = 0
            print(f"  -> New best fitness found: {best_fitness:.6f}")
        else:
            patience += 1
        if patience >= opts["early_stop_patience"]:
            print(f"\nEarly stopping triggered after generations.")
            break

    best_ind = hof[0]
    enforce_constraints(best_ind, indices_map)

    disp = predict_full_field(best_ind)
    max_abs_disp = float(np.max(np.abs(disp)))

    viz_payload = {}

    visualize_optimal_result(best_ind, trunk_df, output_dir=output_dir)
    visualize_normalized_scatter(best_ind, param_bounds, param_names, output_dir=output_dir)
    if cad_context and cad_context.get("background_path"):
        visualize_overlay_final(
            pd.Series(best_ind, index=param_names),
            cad_context["background_path"],
            cad_context["cad_extent"],
            dynamic_constraints,
            output_dir=output_dir,
        )

    verify_dynamic_constraints(best_ind, dynamic_constraints, param_names)
    return {
        "min_fitness": float(best_ind.fitness.values[0]),
        "max_displacement": max_abs_disp,
        "best_individual": best_ind,
        "visuals": viz_payload,
    }

