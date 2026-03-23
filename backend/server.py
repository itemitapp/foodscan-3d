#!/usr/bin/env python3
"""
FoodScan 3D — MapAnything Backend Server v2
Metric 3D food reconstruction using Facebook's MapAnything.
"""
import os
import sys
import uuid
import shutil
import tempfile
import asyncio
from pathlib import Path
from typing import List

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

# Add MapAnything to Python path
BACKEND_DIR = Path(__file__).parent
sys.path.insert(0, str(BACKEND_DIR / "map-anything"))

from volume_calculator import compute_food_volumes, mesh_to_glb

app = FastAPI(title="FoodScan 3D API", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GLB_DIR = BACKEND_DIR / "output"
GLB_DIR.mkdir(exist_ok=True)
app.mount("/glb", StaticFiles(directory=str(GLB_DIR)), name="glb")

# ── Device detection ──────────────────────────────────────────────────────────
import torch
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
print(f"🖥  Device: {DEVICE}")

# ── Lazy model loading ────────────────────────────────────────────────────────
_model = None
_model_lock = asyncio.Lock()

async def get_model():
    global _model
    async with _model_lock:
        if _model is None:
            from mapanything.models import MapAnything
            print("⬇️  Loading facebook/map-anything-apache (~5 GB on first run)...")
            _model = MapAnything.from_pretrained("facebook/map-anything-apache").to(DEVICE)
            _model.eval()
            print("✅ Model ready!")
    return _model

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok", "device": DEVICE, "version": "2.0"}


@app.post("/api/reconstruct")
async def reconstruct(
    images: List[UploadFile] = File(...),
    plate_diameter_cm: float = 26.0,
):
    if not images:
        raise HTTPException(400, "No images provided")

    job_id = str(uuid.uuid4())
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"foodscan_{job_id}_"))

    try:
        # Save uploaded images
        img_dir = tmp_dir / "images"
        img_dir.mkdir()
        for i, f in enumerate(images):
            suffix = Path(f.filename or "img.jpg").suffix or ".jpg"
            (img_dir / f"img_{i:03d}{suffix}").write_bytes(await f.read())

        # Run MapAnything in thread pool (non-blocking)
        model = await get_model()
        result = await asyncio.get_event_loop().run_in_executor(
            None, _infer, model, img_dir
        )

        # Compute volumes
        volumes_result = compute_food_volumes(
            pointcloud=result["pointcloud"],
            colors=result["colors"],
            plate_diameter_cm=plate_diameter_cm,
        )

        # Export GLB
        glb_path = GLB_DIR / f"{job_id}.glb"
        mesh_to_glb(result["pointcloud"], result["colors"], glb_path)

        return JSONResponse({
            "job_id": job_id,
            "glb_url": f"/glb/{job_id}.glb",
            "total_volume_ml": volumes_result["total_volume_ml"],
            "segments": volumes_result["segments"],
            "device_used": DEVICE,
            "num_images": len(images),
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(500, str(e))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _infer(model, img_dir: Path) -> dict:
    """Synchronous MapAnything inference — runs in thread pool."""
    import torch
    from mapanything.utils.image import load_images
    from mapanything.utils.geometry import depthmap_to_world_frame

    views = load_images(str(img_dir))
    print(f"  Loaded {len(views)} view(s). Running inference...")

    with torch.no_grad():
        outputs = model.infer(
            views,
            memory_efficient_inference=True,
            minibatch_size=1,
            use_amp=False,   # AMP = bf16, not supported on MPS
            apply_mask=True,
            mask_edges=True,
        )

    # Accumulate all views into a single point cloud
    all_pts = []
    all_colors = []

    for pred in outputs:
        depth   = pred["depth_z"][0].squeeze(-1)          # (H, W)
        K       = pred["intrinsics"][0]                    # (3, 3)
        pose    = pred["camera_poses"][0]                  # (4, 4)
        img_np  = pred["img_no_norm"][0].cpu().numpy()    # (H, W, 3) 0-255
        mask_np = pred["mask"][0].squeeze(-1).cpu().numpy().astype(bool)  # (H, W)

        pts3d, valid = depthmap_to_world_frame(depth, K, pose)
        combined_mask = mask_np & valid.cpu().numpy()

        pts   = pts3d.cpu().numpy()[combined_mask]           # (N, 3)
        colors = img_np[combined_mask].astype(np.float32) / 255.0  # (N, 3)

        all_pts.append(pts)
        all_colors.append(colors)

    pointcloud = np.concatenate(all_pts, axis=0)
    colors     = np.concatenate(all_colors, axis=0)
    print(f"  Point cloud: {len(pointcloud):,} points")
    return {"pointcloud": pointcloud, "colors": colors}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
