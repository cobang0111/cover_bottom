import os, io, random, json, types, re
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from deap import base, creator, tools
from app.services import pi_opt_legacy
import base64

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

pi_opt_legacy.initialize_legacy_pipeline(
      assets_dir=ASSETS_DIR,
      original_data_dir=ORIG_DIR,
      seed_csv_path=top_seeds_path,
      output_dir=LEGACY_OUTPUT_DIR,
)

# 재현성
SEED=100
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

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
assets = torch.load(assets_path, weights_only=False)
model_state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
original_branch_df = pd.read_csv(branch_csv_path, index_col=0)
trunk_df = pd.read_csv(trunk_csv_path, index_col=0)
seed_ids = pd.read_csv(top_seeds_path)['Sample_ID'].tolist()

branch_scaler     = assets['branch_scaler']
target_scaler     = assets['target_scaler']
trunk_tensor      = assets['trunk_tensor'].to(device)
feature_weights   = assets['feature_weights']
model_params      = assets['model_hyperparameters']

model = pi_opt_legacy.DeepONet(
    pi_opt_legacy.MLP(model_params['branch_input_dim'], model_params['branch_hidden_dims'], model_params['latent_dim']),
    pi_opt_legacy.MLP(model_params['trunk_input_dim'],  model_params['trunk_hidden_dims'],  model_params['latent_dim'])
).to(device)
model.load_state_dict(model_state_dict)
model.eval()

# 파라미터 정보
param_mins = original_branch_df.min().values
param_maxs = original_branch_df.max().values
PARAM_BOUNDS = list(zip(param_mins, param_maxs))
param_names = original_branch_df.columns.tolist()

# 외부 스크립트식 범위 확장(+50) 적용
params_to_modify = ['v1_3','v2_3','v3_3','v4_3','v5_3','v6_3','v1_4','v2_4','v3_4','v4_4','v5_4','v6_4']
for p in params_to_modify:
    if p in param_names:
        i = param_names.index(p)
        lo, hi = PARAM_BOUNDS[i]
        PARAM_BOUNDS[i] = (lo, hi + 50.0)
        print(f" -> '{p}' (인덱스 {i}) 범위 변경: ({lo:.2f}, {hi:.2f}) -> ({lo:.2f}, {hi+50.0:.2f})")


def shape_mask(trunk_df, bp):
    x = trunk_df['X'].values
    y = trunk_df['Y'].values
    X_LEFT, X_RIGHT = -567.7, 567.7
    mask_gray = np.zeros_like(x, dtype=bool)
    X_MIN_BOUNDARY = -579.34
    X_MAX_BOUNDARY = 579.34
    Y_MIN_BOUNDARY = -301.03
    Y_MAX_BOUNDARY = 301.03
    outside_mask = (x < X_MIN_BOUNDARY) | (x > X_MAX_BOUNDARY) | (y < Y_MIN_BOUNDARY) | (y > Y_MAX_BOUNDARY)
    mask_gray[outside_mask] = True

    y1_lo, y1_hi = bp['h2_1_up'], bp['h1_1_down']
    y2_lo, y2_hi = bp['h2_2_up'], bp['h1_2_down']
    y3_lo, y3_hi = bp['h2_3_up'], bp['h1_3_down']
    y4_lo, y4_hi = bp['h2_4_up'], bp['h1_4_down']
    y5_lo, y5_hi = bp['h2_5_up'], bp['h1_5_down']
    y6_lo,  y6_hi  = -bp['h3_1_up'], bp['h2_1_down']
    y7_lo,  y7_hi  = -bp['h3_2_up'], bp['h2_2_down']
    y8_lo,  y8_hi  = -bp['h3_3_up'], bp['h2_3_down']
    y9_lo,  y9_hi  = -bp['h3_4_up'], bp['h2_4_down']
    y10_lo, y10_hi = -bp['h3_5_up'], bp['h2_5_down']
    y11_lo, y11_hi = -bp['h4_1_up'],  -bp['h3_1_down']
    y12_lo, y12_hi = -bp['h4_2_up'],  -bp['h3_2_down']
    y13_lo, y13_hi = -bp['h4_3_up'],  -bp['h3_3_down']
    y14_lo, y14_hi = -bp['h4_4_up'],  -bp['h3_4_down']
    y15_lo, y15_hi = -bp['h4_5_up'],  -bp['h3_5_down']
    y_bot_lo, y_bot_hi = -314, 0

    mask_gray |= ((x >= -bp['v1_1_p']) & (x <= -(bp['v2_1_p']+bp['v2_1'])) & (y>=y1_lo)&(y<=y1_hi))
    mask_gray |= ((x >= X_LEFT)           & (x <= -(bp['v1_1_p']+bp['v1_1'])) & (y>=y1_lo)&(y<=y1_hi))
    mask_gray |= ((x >= -bp['v2_1_p']) & (x <= -(bp['v3_1_p']+bp['v3_1'])) & (y>=y2_lo)&(y<=y2_hi))
    mask_gray |= ((x >= -bp['v3_1_p']) & (x <=  bp['v4_1_p'])              & (y>=y3_lo)&(y<=y3_hi))
    mask_gray |= ((x >=  bp['v4_1_p']+bp['v4_1']) & (x <= bp['v5_1_p'])    & (y>=y4_lo)&(y<=y4_hi))
    mask_gray |= ((x >=  bp['v5_1_p']+bp['v5_1']) & (x <= bp['v6_1_p'])    & (y>=y5_lo)&(y<=y5_hi))
    mask_gray |= ((x >=  bp['v6_1_p']+bp['v6_1']) & (x <= X_RIGHT)           & (y>=y5_lo)&(y<=y5_hi))
    mask_gray |= ((x >= -bp['v1_2_p'])                   & (x <= -(bp['v2_2_p']+bp['v2_2'])) & (y>=y6_lo)&(y<=y6_hi))
    mask_gray |= ((x >= X_LEFT)                       & (x <= -(bp['v1_2_p']+bp['v1_2'])) & (y>=y6_lo)&(y<=y6_hi))
    mask_gray |= ((x >= -bp['v2_2_p'])                   & (x <= -(bp['v3_2_p']+bp['v3_2'])) & (y>=y7_lo)&(y<=y7_hi))
    mask_gray |= ((x >= -bp['v3_2_p'])                   & (x <=  bp['v4_2_p'])              & (y>=y8_lo)&(y<=y8_hi))
    mask_gray |= ((x >=  bp['v4_2_p']+bp['v4_2'])       & (x <= bp['v5_2_p'])               & (y>=y9_lo)&(y<=y9_hi))
    mask_gray |= ((x >=  bp['v5_2_p']+bp['v5_2'])       & (x <= bp['v6_2_p'])               & (y>=y10_lo)&(y<=y10_hi))
    mask_gray |= ((x >=  bp['v6_2_p']+bp['v6_2'])       & (x <= X_RIGHT)                    & (y>=y10_lo)&(y<=y10_hi))
    mask_gray |= ((x >= -bp['v1_3_p'])                   & (x <= -(bp['v2_3_p']+bp['v2_3'])) & (y>=y11_lo)&(y<=y11_hi))
    mask_gray |= ((x >= X_LEFT)                       & (x <= -(bp['v1_3_p']+bp['v1_3'])) & (y>=y11_lo)&(y<=y11_hi))
    mask_gray |= ((x >= -bp['v2_3_p'])                   & (x <= -(bp['v3_3_p']+bp['v3_3'])) & (y>=y12_lo)&(y<=y12_hi))
    mask_gray |= ((x >= -bp['v3_3_p'])                   & (x <=  bp['v4_3_p'])              & (y>=y13_lo)&(y<=y13_hi))
    mask_gray |= ((x >=  bp['v4_3_p']+bp['v4_3'])       & (x <= bp['v5_3_p'])               & (y>=y14_lo)&(y<=y14_hi))
    mask_gray |= ((x >=  bp['v5_3_p']+bp['v5_3'])       & (x <= bp['v6_3_p'])               & (y>=y15_lo)&(y<=y15_hi))
    mask_gray |= ((x >=  bp['v6_3_p']+bp['v6_3'])       & (x <= X_RIGHT)                    & (y>=y15_lo)&(y<=y15_hi))

    if bp['v1_4'] == 80:
        mask_gray |= ((x >= X_LEFT) & (x <= -(bp['v1_4_p'] + bp['v1_4'])) & (y >= y_bot_lo) & (y <= y_bot_lo + bp['h5_1']))
    else:
        mask_gray |= ((x >= X_LEFT) & (x <= -(bp['v1_4_p'] + bp['v1_4'])) & (y >= y_bot_lo) & (y <= -bp['h4_1_down']))
    mask_gray |= ((x >= -(bp['v1_4_p']+bp['v1_4']))    & (x <= -bp['v1_4_p'])              & (y>=y_bot_lo)&(y<= y_bot_lo + bp['h5_1']))
    mask_gray |= ((x >= -bp['v1_4_p'])                   & (x <= -(bp['v2_4_p']+bp['v2_4'])) & (y>=y_bot_lo)&(y<= -bp['h4_1_down']))
    mask_gray |= ((x >= -bp['v2_4_p'])                   & (x <= -200.25)                    & (y>=y_bot_lo)&(y<= -bp['h4_2_down']))
    mask_gray |= ((x >= -200.25)                       & (x <= -(bp['v3_4_p']+bp['v3_4'])) & (y>=y_bot_lo)&(y<= -bp['h4_3_down']))
    mask_gray |= ((x >= -(bp['v3_4_p']+bp['v3_4']))    & (x <= -bp['v3_4_p'])              & (y>=y_bot_lo)&(y<= y_bot_lo + bp['h5_3']))
    mask_gray |= ((x >= -bp['v3_4_p'])                   & (x <= bp['v4_4_p'])               & (y>=y_bot_lo)&(y<= -bp['h4_3_down']))
    mask_gray |= ((x >= bp['v4_4_p'])                    & (x <= bp['v4_4_p']+bp['v4_4'])   & (y>=y_bot_lo)&(y<= y_bot_lo + bp['h5_4']))
    mask_gray |= ((x >= bp['v4_4_p']+bp['v4_4'])        & (x <= 200.25)                     & (y>=y_bot_lo)&(y<= -bp['h4_3_down']))
    mask_gray |= ((x >= 200.25)                        & (x <= bp['v5_4_p'])               & (y>=y_bot_lo)&(y<= -bp['h4_4_down']))
    mask_gray |= ((x >= bp['v5_4_p'])                    & (x <= bp['v5_4_p']+bp['v5_4'])   & (y>=y_bot_lo)&(y<= y_bot_lo + bp['h5_5']))
    mask_gray |= ((x >= bp['v5_4_p']+bp['v5_4'])        & (x <= bp['v6_4_p'])               & (y>=y_bot_lo)&(y<= -bp['h4_5_down']))
    mask_gray |= ((x >= bp['v6_4_p'])                    & (x <= bp['v6_4_p']+bp['v6_4'])   & (y>=y_bot_lo)&(y<= y_bot_lo + bp['h5_6']))
    if bp['v6_4'] == 80:
        mask_gray |= ((x >= bp['v6_4_p'] + bp['v6_4']) & (x <= X_RIGHT) & (y >= y_bot_lo) & (y <= y_bot_lo + bp['h5_6']))
    else:
        mask_gray |= ((x >= bp['v6_4_p'] + bp['v6_4']) & (x <= X_RIGHT) & (y >= y_bot_lo) & (y <= -bp['h4_5_down']))

    rev = int(bp['rev_forming1'])
    fw  = bp['forming width']
    y_hi = bp['h1_p'] - 314.066 - 22.202
    y_lo = y_hi - fw
    if rev == 1:
        mask_gray |= ((x >= -515.25) & (x <= 515.25) & (y >= y_lo) & (y <= y_hi))
    elif rev == 3:
        for x0, x1 in [(-515.25,-186.45),(-107.25,107.25),(186.45,515.25)]:
            mask_gray |= ((x >= x0) & (x <= x1) & (y >= y_lo) & (y <= y_hi))
    elif rev == 5:
        for x0, x1 in [(-515.25,-369.75),(-331.25,-145.75),(-107.25,107.25),(145.75,331.25),(369.75,515.25)]:
            mask_gray |= ((x >= x0) & (x <= x1) & (y >= y_lo) & (y <= y_hi))
    return np.where(mask_gray, 0, 1)


def render_shape_mask(best_individual):
    best_series = pd.Series(best_individual, index=original_branch_df.columns)
    mask = shape_mask(trunk_df, best_series)
    x = trunk_df['X'].values; y = trunk_df['Y'].values
    fig, ax = plt.subplots(figsize=(12,8))
    cmap_mask = ListedColormap(['lightgray', 'gold'])
    ax.tricontourf(x, y, mask.astype(float), levels=[-0.5,0.5,1.5], cmap=cmap_mask)
    ax.set_title("Shape Mask of Optimal Design (Twist)")
    ax.set_aspect('equal'); ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
    buf = io.BytesIO(); plt.tight_layout(); fig.savefig(buf, format='png', dpi=180); plt.close(fig)
    buf.seek(0)
    import base64
    return base64.b64encode(buf.read()).decode('ascii')


def _build_cad_context(view_id: str | None):
    if not view_id:
        return None

    # OCC/ORIG/기존 메타 모두 탐색
    meta_candidates = [
        os.path.join(VIEW_CACHE_DIR, f"{view_id}_occ.json"),
        os.path.join(VIEW_CACHE_DIR, f"{view_id}_orig.json"),
        os.path.join(VIEW_CACHE_DIR, f"{view_id}.json"),
    ]
    meta_path = next((p for p in meta_candidates if os.path.exists(p)), None)
    if not meta_path:
        return None

    meta = json.load(open(meta_path, "r", encoding="utf-8"))

    # 1) OCC 메타: cad_extent 직접 사용
    if "cad_extent" in meta:
        xmin, xmax, ymin, ymax = meta["cad_extent"]
        cad_extent = (xmin, xmax, ymin, ymax)
    # 2) Matplotlib 메타: cad_xlim/cad_ylim으로부터 extent 구성
    elif "cad_xlim" in meta and "cad_ylim" in meta:
        x0, x1 = meta["cad_xlim"]
        y0, y1 = meta["cad_ylim"]
        cad_extent = (min(x0, x1), max(x0, x1), min(y0, y1), max(y0, y1))
    else:
        return None

    # 배경 이미지 후보(OCC/ORIG/일반)
    bg_candidates = [
        os.path.join(VIEW_CACHE_DIR, f"{view_id}_orig.png"),
        os.path.join(VIEW_CACHE_DIR, f"{view_id}.png"),
    ]
    bg_path = next((p for p in bg_candidates if os.path.exists(p)), None)

    return {
        "background_path": bg_path,
        "cad_extent": cad_extent,
    }

def run_ga_and_mask(dynamic_constraints, ga_opts=None, view_id: str | None = None):
    cad_context = _build_cad_context(view_id)

    result = pi_opt_legacy.run_legacy_ga(
        dynamic_constraints=dynamic_constraints or {},
        ga_opts=ga_opts or {},
        cad_context=cad_context,
        return_base64=False,
    )

    visuals = result.get("visuals", {}) or {}
    overlay = visuals.get("overlay_base64") or visuals.get("mask_base64")
    contour = visuals.get("contour_base64")

    # Fallback: 레거시 PNG를 읽어 base64로 반환
    if not overlay:
        fallback_overlay = os.path.join(LEGACY_OUTPUT_DIR, "Optimal_Design_Overlay.png")
        if os.path.exists(fallback_overlay):
            with open(fallback_overlay, "rb") as f:
                overlay = base64.b64encode(f.read()).decode("ascii")
    if not contour:
        fallback_contour = os.path.join(LEGACY_OUTPUT_DIR, "optimal_design_contour_grav.png")
        if os.path.exists(fallback_contour):
            with open(fallback_contour, "rb") as f:
                contour = base64.b64encode(f.read()).decode("ascii")

    return {
        "min_fitness": result.get("min_fitness"),
        "mask_base64": overlay,
        "contour_base64": contour,
        "max_displacement": result.get("max_displacement"),
    }