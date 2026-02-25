import os
import uuid, base64, json
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from app.routers import cover_bottom
#from app.services.models import ViewParams, ViewResponse, AnalyzeRequest, AnalyzeResponse, OptimizeRequest, OptimizeResponse
#from app.services.cad_service import render_view, create_cad_background_and_get_bounds
from app.services.constraint_analyzer import analyze_from_rects
from app.services.pi_opt_service import run_ga_and_mask
from app.services.step_displacement_predictor import compute_preopt_max_displacement

from app.services.cad_service import render_view, create_cad_background_and_get_bounds, render_occ_and_extract, compute_h1_p_from_px_box, compute_rev_forming_from_px_box
from app.services.models import ViewParams, ViewResponse, AnalyzeRequest, AnalyzeResponse, OptimizeRequest, OptimizeResponse, H1pRequest, H1pResponse, AnalyzeOccRequest

DEFAULT_STEP = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "55_cover_bottom.stp"))
UPLOAD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "uploads"))
VIEW_DIR = os.path.join(UPLOAD_DIR, "views")
os.makedirs(VIEW_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(cover_bottom.router, prefix="/api")

@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".stp", ".step")):
        raise HTTPException(status_code=400, detail="Only .stp/.step allowed")
    fname = f"{uuid.uuid4().hex}_{os.path.basename(file.filename)}"
    dst = os.path.join(UPLOAD_DIR, fname)
    with open(dst, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"step_file": dst}

@app.post("/api/view-occ-upload", response_model=ViewResponse)
async def api_view_occ_upload(
    file: UploadFile = File(...),
    z_threshold: float = Form(-45.2),
    z_outline: float = Form(-38.2),
    width: int | None = Form(None),
):
    if not file.filename.lower().endswith((".stp", ".step")):
        raise HTTPException(status_code=400, detail="Only .stp/.step allowed")
    fname = f"{uuid.uuid4().hex}_{os.path.basename(file.filename)}"
    dst = os.path.join(UPLOAD_DIR, fname)
    with open(dst, "wb") as f:
        shutil.copyfileobj(file.file, f)

    meta = render_occ_and_extract(dst, z_threshold, z_outline, width)
    view_id = uuid.uuid4().hex

    with open(os.path.join(VIEW_DIR, f"{view_id}.png"), "wb") as f:
        f.write(base64.b64decode(meta["image_b64"]))
    with open(os.path.join(VIEW_DIR, f"{view_id}_occ.json"), "w", encoding="utf-8") as f:
        json.dump({**meta, "step_file": dst}, f)

    xmin, xmax, ymin, ymax = meta["cad_extent"]
    cad_xlim = (xmax, xmin)  # X축 반전 효과 유지(_px_to_cad 호환)
    cad_ylim = (meta["baseline"], ymax)  # baseline을 하한으로
    pw, ph = meta["pixel_size"]

    return ViewResponse(
        image_base64=meta["image_b64"],
        cad_xlim=cad_xlim,
        cad_ylim=cad_ylim,
        pixel_width=pw,
        pixel_height=ph,
        view_id=view_id
    )


@app.post("/api/view-occ", response_model=ViewResponse)
def api_view_occ(params: ViewParams):
    step_file = params.step_file or DEFAULT_STEP
    meta = render_occ_and_extract(step_file, params.z_threshold, params.z_outline, params.width)
    view_id = uuid.uuid4().hex

    with open(os.path.join(VIEW_DIR, f"{view_id}.png"), "wb") as f:
        f.write(base64.b64decode(meta["image_base64"] if "image_base64" in meta else meta["image_b64"]))
    with open(os.path.join(VIEW_DIR, f"{view_id}_occ.json"), "w", encoding="utf-8") as f:
        json.dump({**meta, "step_file": step_file}, f)

    xmin, xmax, ymin, ymax = meta["cad_extent"]
    cad_xlim = (xmax, xmin)
    cad_ylim = (meta["baseline"], ymax)
    pw, ph = meta["pixel_size"]

    return ViewResponse(
        image_base64=meta["image_b64"],
        cad_xlim=cad_xlim,
        cad_ylim=cad_ylim,
        pixel_width=pw,
        pixel_height=ph,
        view_id=view_id
    )


@app.post("/api/h1p-occ", response_model=H1pResponse)
def api_h1p_occ(req: H1pRequest):
    meta_path = os.path.join(VIEW_DIR, f"{req.view_id}_occ.json")
    if not os.path.exists(meta_path):
        raise HTTPException(status_code=404, detail="occ meta not found")
    meta = json.load(open(meta_path, "r", encoding="utf-8"))

    r = req.box
    res = compute_h1_p_from_px_box(
        (r.x1, r.y1, r.x2, r.y2),
        meta["cad_extent"],
        tuple(meta["pixel_size"]),
        meta["baseline"],
        meta["horizontal_lines"]
    )
    if not res:
        raise HTTPException(status_code=404, detail="h1_p not found in box")

    return H1pResponse(value=res["value"], matched_y=res["matched_y"], base=meta["baseline"])


@app.post("/api/analyze-occ", response_model=AnalyzeResponse)
def api_analyze_occ(body: AnalyzeOccRequest):
    meta_path = os.path.join(VIEW_DIR, f"{body.view_id}_occ.json")
    if not os.path.exists(meta_path):
        raise HTTPException(status_code=404, detail="occ meta not found")
    meta = json.load(open(meta_path, "r", encoding="utf-8"))

    xmin, xmax, ymin, ymax = meta["cad_extent"]
    cad_xlim = (xmax, xmin)  # _px_to_cad와 일치하도록 X 반전
    cad_ylim = (meta["baseline"], ymax)
    pw, ph = meta["pixel_size"]

    # 기존 분석기를 그대로 재사용
    res = analyze_from_rects(body.required_rects, body.forbidden_rects, cad_xlim, cad_ylim, pw, ph)

    # 같은 호출에서 h1_p까지 계산해 고정값에 병합(옵션)
    if body.h1p_box:
        b = body.h1p_box
        h1 = compute_h1_p_from_px_box(
            (b.x1, b.y1, b.x2, b.y2),
            meta["cad_extent"],
            tuple(meta["pixel_size"]),
            meta["baseline"],
            meta["horizontal_lines"]
        )
        if h1:
            res.setdefault("fixed_value", {})
            res["fixed_value"]["h1_p"] = h1["value"]
            res["h1_p_components"] = {"base": meta["baseline"], "y": h1["matched_y"]}

    # h1_p 폴백: 박스가 없거나 실패해도 base만 넣어줌
    if not body.h1p_box:
        res.setdefault("h1_p_components", {"base": meta["baseline"]})
    elif "h1_p_components" not in res:
        res["h1_p_components"] = {"base": meta["baseline"]}

    # rev_forming/forming_width 계산(선택 입력)
    if body.rev_forming_box:
        r = body.rev_forming_box
        rv = compute_rev_forming_from_px_box(
            (r.x1, r.y1, r.x2, r.y2),
            meta["cad_extent"],
            tuple(meta["pixel_size"]),
            meta["horizontal_lines"],
        )
        if rv:
            res["forming_variable"] = rv

    return AnalyzeResponse(**res)



@app.post("/api/view-upload", response_model=ViewResponse)
async def api_view_upload(
    file: UploadFile = File(...),
    z_threshold: float = Form(-45.2),
    z_outline: float = Form(-38.2),
    width: int | None = Form(None),
):
    if not file.filename.lower().endswith((".stp", ".step")):
        raise HTTPException(status_code=400, detail="Only .stp/.step allowed")
    fname = f"{uuid.uuid4().hex}_{os.path.basename(file.filename)}"
    dst = os.path.join(UPLOAD_DIR, fname)
    with open(dst, "wb") as f:
        shutil.copyfileobj(file.file, f)

    b64, cad_xlim, cad_ylim, w, h = render_view(dst, z_threshold, z_outline, width)

    view_id = uuid.uuid4().hex
    with open(os.path.join(VIEW_DIR, f"{view_id}.png"), "wb") as f:
        f.write(base64.b64decode(b64))
    with open(os.path.join(VIEW_DIR, f"{view_id}.json"), "w", encoding="utf-8") as f:
        json.dump({"cad_xlim": cad_xlim, "cad_ylim": cad_ylim, "w": w, "h": h, "step_file": dst}, f)
        
    # 원본 저장(OCC)
    try:
        orig_png = os.path.join(VIEW_DIR, f"{view_id}_orig.png")
        _, (xmin, xmax, ymin, ymax) = create_cad_background_and_get_bounds(dst, orig_png)
        with open(os.path.join(VIEW_DIR, f"{view_id}_orig.json"), "w", encoding="utf-8") as f:
            json.dump({"cad_xlim": (xmin, xmax), "cad_ylim": (ymin, ymax), "w": w, "h": h}, f)
    except Exception as e:
        print(f"[orig] 배경 생성 실패: {e}")

    return ViewResponse(
        image_base64=b64,
        cad_xlim=cad_xlim,
        cad_ylim=cad_ylim,
        pixel_width=w,
        pixel_height=h,
        view_id=view_id
    )

@app.post("/api/view", response_model=ViewResponse)
def api_view(params: ViewParams):
    step_file = params.step_file or DEFAULT_STEP
    b64, cad_xlim, cad_ylim, w, h = render_view(step_file, params.z_threshold, params.z_outline, params.width)
    
    view_id = uuid.uuid4().hex
    with open(os.path.join(VIEW_DIR, f"{view_id}.png"), "wb") as f:
        f.write(base64.b64decode(b64))
    with open(os.path.join(VIEW_DIR, f"{view_id}.json"), "w", encoding="utf-8") as f:
        json.dump({"cad_xlim": cad_xlim, "cad_ylim": cad_ylim, "w": w, "h": h}, f)

    return ViewResponse(
        image_base64=b64,
        cad_xlim=cad_xlim,
        cad_ylim=cad_ylim,
        pixel_width=w,
        pixel_height=h,
        view_id=view_id
    )

@app.post("/api/analyze", response_model=AnalyzeResponse)
def api_analyze(body: AnalyzeRequest):
    res = analyze_from_rects(body.required_rects, body.forbidden_rects, body.cad_xlim, body.cad_ylim, body.pixel_width, body.pixel_height)
    print("[ANALYZE]", {
            "add_unconditional": res.get("add_unconditional"),
            "fixed_zero": res.get("fixed_zero"),
            "non_zero": res.get("non_zero"),
            "not_equal": res.get("not_equal"),
            "must_equal": res.get("must_equal"),
            "upper_bound": res.get("upper_bound"),
            "fixed_value": res.get("fixed_value"),
            "forbidden_zones": res.get("forbidden_zones"),
        })
    return AnalyzeResponse(**res)

@app.post("/api/optimize-mask", response_model=OptimizeResponse)
def api_optimize_mask(req: OptimizeRequest):
    print(f"[OPT] view_id={req.view_id}")
    h1p_components = req.h1_p_components
    if (not h1p_components) and req.view_id and req.fixed_value and ("h1_p" in req.fixed_value):
        meta_candidates = [
            os.path.join(VIEW_DIR, f"{req.view_id}_occ.json"),
            os.path.join(VIEW_DIR, f"{req.view_id}_orig.json"),
            os.path.join(VIEW_DIR, f"{req.view_id}.json"),
        ]
        meta_path = next((p for p in meta_candidates if os.path.exists(p)), None)
        if meta_path:
            meta = json.load(open(meta_path, "r", encoding="utf-8"))
            baseline = meta.get("baseline")
            if baseline is not None:
                y_val = req.fixed_value["h1_p"] - abs(baseline)
                h1p_components = {"base": baseline, "y": y_val}

    dyn_constraints = {
        "add_unconditional": req.add_unconditional,
        "fixed_zero": req.fixed_zero,
        "non_zero": req.non_zero,
        "not_equal": req.not_equal,
        "fixed_value": req.fixed_value,
        "upper_bound": req.upper_bound,
        "must_equal": req.must_equal,
        "forbidden_zones": req.forbidden_zones,     
    }
    if h1p_components:
        dyn_constraints["h1_p_components"] = h1p_components  

    print("[OPT-CONSTRAINTS]\n" + json.dumps(dyn_constraints, ensure_ascii=False, indent=2))

    res = run_ga_and_mask(
        dynamic_constraints={
            "add_unconditional": req.add_unconditional,
            "fixed_zero": req.fixed_zero,
            "non_zero": req.non_zero,
            "not_equal": req.not_equal,
            "fixed_value": req.fixed_value,
            "upper_bound": req.upper_bound,
            "must_equal": req.must_equal,
        },
        ga_opts={
            "max_generations": req.max_generations,
            "p_crossover": req.p_crossover,
            "p_mutation": req.p_mutation,
            "num_random_individuals": req.num_random_individuals,
            "early_stop_patience": req.early_stop_patience,
        },
        view_id=req.view_id
    )

    if not res or not isinstance(res.get("mask_base64"), str) or not res["mask_base64"]:
        detail = res.get("error") if isinstance(res, dict) else "mask not generated"
        raise HTTPException(status_code=500, detail=f"optimize failed: {detail}")


    return OptimizeResponse(
        mask_base64=res["mask_base64"], 
        min_fitness=res["min_fitness"], 
        contour_base64=res.get("contour_base64"), 
        overlay_base64=res.get("overlay_base64") or (res["mask_base64"] if req.view_id else None),
        combined_base64=res.get("combined_base64"),
        max_displacement=res.get("max_displacement"),
    )

@app.post("/api/preopt-max-disp")
def api_preopt_max_disp(payload: dict = Body(...)):
    view_id = payload.get("view_id")
    if not view_id:
        raise HTTPException(status_code=400, detail="view_id is required")

    # OCC 우선, 일반 백업
    meta_candidates = [
        os.path.join(VIEW_DIR, f"{view_id}_occ.json"),
        os.path.join(VIEW_DIR, f"{view_id}.json"),
    ]

    #meta_path = os.path.join(VIEW_DIR, f"{view_id}.json")
    meta_path = next((p for p in meta_candidates if os.path.exists(p)), None)

    if not os.path.exists(meta_path):
        raise HTTPException(status_code=404, detail="view meta not found")
    meta = json.load(open(meta_path, "r", encoding="utf-8"))
    step_file = meta.get("step_file")
    if not step_file or not os.path.exists(step_file):
        raise HTTPException(status_code=404, detail="step file not found")
    try:
        val = compute_preopt_max_displacement(step_file)
        return {"max_displacement": val}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"preopt failed: {e}")