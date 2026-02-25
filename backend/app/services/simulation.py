import time
from typing import Dict, Any

def enqueue_simulation(_: bytes, __: Dict[str, Any]) -> str:
    # 실제로는 작업 큐에 등록하고 job_id 반환 (예: Redis/RQ/Celery)
    return str(int(time.time() * 1000))

def get_simulation_result(_: str) -> Dict[str, Any]:
    # 데모 목적의 더미 결과
    return {
        "metadata": {
            "file_name": "Cad_File.stp",
            "status": "completed",
            "user_name": "사용자",
            "user_group": "TV CAE팀",
            "create_date": "2025-10-09T01:23:45Z",
            "analysis_start_date": "2025-10-09T01:24:11Z",
            "complete_date": "2025-10-09T01:34:51Z",
            "remark": "샘플",
        },
        "parameters": {
            "material": "Aluminium",
            "pcb_length": 945,
            "led_housing_gap": 6.5,
            "led_spacing": 1.74,
            "led_count": 216,
            "led_watt": 0.25,
            "led_length_x": 7,
            "led_height_y": 0.8,
            "led_depth_z": 2.0,
        },
        "results": {
            "contour_image": "/cover_bottom/contour.png",
            "hotspots": [
                {"no": 1, "x": 522, "y": 146, "z": 200, "temperature": 101.4, "spec": 100, "judge": "NG"},
                {"no": 2, "x": -516, "y": 146, "z": 200, "temperature": 96.7, "spec": 100, "judge": "OK"},
            ],
        },
    }