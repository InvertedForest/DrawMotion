import asyncio
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from demo.drawmotion_studio.runner import DrawMotionRunner


APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"
DEFAULT_CKPT = "logs/human_ml3d/last.ckpt"

app = FastAPI(title="DrawMotion Studio")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

runner = None
generate_lock = asyncio.Lock()


def validate_number(value, name, min_value, max_value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HTTPException(status_code=400, detail=f"{name} must be a number")
    if value < min_value or value > max_value:
        raise HTTPException(status_code=400, detail=f"{name} must be in [{min_value}, {max_value}]")


def validate_integer(value, name, min_value, max_value):
    if isinstance(value, bool) or not isinstance(value, int):
        raise HTTPException(status_code=400, detail=f"{name} must be an integer")
    if value < min_value or value > max_value:
        raise HTTPException(status_code=400, detail=f"{name} must be in [{min_value}, {max_value}]")


def validate_generate_payload(payload):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="payload must be an object")
    text = payload.get("text")
    if not isinstance(text, str) or len(text.strip()) == 0:
        raise HTTPException(status_code=400, detail="text must be a non-empty string")
    trajectory = payload.get("trajectory")
    if not isinstance(trajectory, list) or len(trajectory) < 2:
        raise HTTPException(status_code=400, detail="trajectory must contain at least two points")
    for point in trajectory:
        if not isinstance(point, dict):
            raise HTTPException(status_code=400, detail="trajectory points must be objects")
        validate_number(point.get("x"), "trajectory.x", -10000, 10000)
        validate_number(point.get("y"), "trajectory.y", -10000, 10000)
    validate_integer(payload.get("length"), "length", 2, 196)
    validate_number(payload.get("density"), "density", 0, 1)
    validate_number(payload.get("trajectory_scale"), "trajectory_scale", 1, 1000)
    validate_integer(payload.get("ifg_repeat"), "ifg_repeat", 0, 100)
    validate_number(payload.get("ifg_scale"), "ifg_scale", 0, 200)
    clean_payload = dict(payload)
    clean_payload["text"] = text.strip()
    clean_payload["stickmen"] = []
    return clean_payload


def get_runner():
    global runner
    if runner is None:
        ckpt = os.environ.get("DRAWMOTION_CKPT", DEFAULT_CKPT)
        gpu = os.environ.get("DRAWMOTION_GPU", "0")
        sample_index = os.environ.get("DRAWMOTION_SAMPLE_INDEX", "0")
        runner = DrawMotionRunner(ckpt_path=ckpt, gpu=gpu, sample_index=sample_index)
    return runner


@app.get("/", response_class=HTMLResponse)
async def index():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/status")
async def status():
    if runner is None:
        return {"loaded": False, "checkpoint": DEFAULT_CKPT}
    return {
        "loaded": True,
        "dataset": runner.dataset_name,
        "joints_num": runner.joints_num,
        "device": str(runner.device),
        "checkpoint": str(runner.ckpt_path),
    }


@app.post("/api/generate")
async def generate(request: Request):
    payload = await request.json()
    payload = validate_generate_payload(payload)
    async with generate_lock:
        return await asyncio.to_thread(get_runner().generate, payload)
