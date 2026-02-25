# backend/app/routers/cover_bottom.py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.services.simulation import enqueue_simulation, get_simulation_result

router = APIRouter(prefix="/cover-bottom", tags=["cover-bottom"])

@router.post("/simulations")
async def create_simulation(
    file: UploadFile = File(...),
    material: str = Form(...),
    pcb_length: float = Form(...),
    led_housing_gap: float = Form(...),
    led_spacing: float = Form(...),
    led_count: int = Form(...),
    led_watt: float = Form(...),
    led_length_x: float = Form(...),
    led_height_y: float = Form(...),
    led_depth_z: float = Form(...),
    remark: str = Form("")
):
    content = await file.read()
    job_id = enqueue_simulation(content, {
        "material": material,
        "pcb_length": pcb_length,
        "led_housing_gap": led_housing_gap,
        "led_spacing": led_spacing,
        "led_count": led_count,
        "led_watt": led_watt,
        "led_length_x": led_length_x,
        "led_height_y": led_height_y,
        "led_depth_z": led_depth_z,
        "remark": remark,
        "file_name": file.filename
    })
    return {"job_id": job_id}

@router.get("/simulations/{job_id}")
async def read_simulation(job_id: str):
    try:
        data = get_simulation_result(job_id)
        return data
    except Exception:
        raise HTTPException(status_code=404, detail="결과를 찾을 수 없습니다.")