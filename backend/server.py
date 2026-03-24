#!/usr/bin/env python3
"""
FoodScan 3D — MapAnything Backend Server v3
Metric 3D food reconstruction using Facebook's MapAnything.
In production mode this server also hosts the built Vite frontend
so only one process is needed: python3 server.py → http://localhost:8000
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
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

# Add MapAnything to Python path
BACKEND_DIR = Path(__file__).parent
ROOT_DIR    = BACKEND_DIR.parent          # repo root
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

# ── Model loaded at startup (not lazily) ──────────────────────────────────────
_model = None

@app.on_event("startup")
async def load_model():
    global _model
    import asyncio
    loop = asyncio.get_event_loop()
    print("⬇️  Loading facebook/map-anything-apache from cache...")
    import time; t0 = time.time()
    _model = await loop.run_in_executor(None, _load_model_sync)
    print(f"✅ Model ready in {time.time()-t0:.1f}s! Accepting requests.")

def _load_model_sync():
    from mapanything.models import MapAnything
    m = MapAnything.from_pretrained("facebook/map-anything-apache").to(DEVICE)
    m.eval()
    return m

async def get_model():
    return _model


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok", "device": DEVICE, "version": "2.0"}


@app.post("/api/reconstruct")
async def reconstruct(
    images: List[UploadFile] = File(...),
    plate_diameter_cm: float = Form(26.0),
    segments:           int   = Form(0),     # 0 = auto
    min_food_height_mm: float = Form(3.0),   # minimum height above plate (mm)
    memory_efficient:   bool  = Form(True),  # slower but uses less RAM
    minibatch_size:     int   = Form(1),     # 1-4, higher = faster but more VRAM
    mask_edges:         bool  = Form(True),  # mask noisy edge depth predictions
):
    if not images:
        raise HTTPException(400, "No images provided")

    job_id = str(uuid.uuid4())
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"foodscan_{job_id}_"))

    cfg = dict(
        memory_efficient=memory_efficient,
        minibatch_size=max(1, min(4, minibatch_size)),
        mask_edges=mask_edges,
    )
    vol_cfg = dict(
        plate_diameter_cm=plate_diameter_cm,
        segments=segments,
        min_food_height_mm=max(0.5, min_food_height_mm),
    )

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
            None, _infer, model, img_dir, cfg
        )

        # Compute volumes
        volumes_result = compute_food_volumes(
            pointcloud=result["pointcloud"],
            colors=result["colors"],
            **vol_cfg,
        )

        # Export GLB
        glb_path = GLB_DIR / f"{job_id}.glb"
        mesh_to_glb(result["pointcloud"], result["colors"], glb_path)

        return JSONResponse({
            "job_id":           job_id,
            "glb_url":          f"/glb/{job_id}.glb",
            "total_volume_ml":  float(volumes_result["total_volume_ml"]),
            "segments":         [
                {**s, "volume_ml": float(s["volume_ml"]), "percentage": float(s.get("percentage", 0))}
                for s in volumes_result["segments"]
            ],
            "device_used":      DEVICE,
            "num_images":       len(images),
            "config":           {k: (bool(v) if isinstance(v, (bool,)) else float(v) if hasattr(v, 'item') else v) for k, v in {**cfg, **vol_cfg}.items()},
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(500, str(e))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _infer(model, img_dir: Path, cfg: dict = None) -> dict:
    """Synchronous MapAnything inference — runs in thread pool."""
    import torch
    from mapanything.utils.image import load_images

    cfg = cfg or {}
    views = load_images(str(img_dir))
    print(f"  Loaded {len(views)} view(s). Running inference (cfg={cfg})...")

    with torch.no_grad():
        outputs = model.infer(
            views,
            memory_efficient_inference=cfg.get("memory_efficient", True),
            minibatch_size=cfg.get("minibatch_size", 1),
            use_amp=False,   # AMP = bf16, not supported on MPS
            apply_mask=True,
            mask_edges=cfg.get("mask_edges", True),
        )

    # Accumulate all views into a single point cloud
    all_pts    = []
    all_colors = []

    for pred in outputs:
        # Use pts3d directly — already in world-frame metric coords
        pts3d_np = pred["pts3d"][0].cpu().numpy()       # (H, W, 3) in meters
        img_np   = pred["img_no_norm"][0].cpu().numpy() # (H, W, 3) range 0-1
        mask_np  = pred["mask"][0].squeeze(-1).cpu().numpy().astype(bool)   # (H, W)
        conf_np  = pred["conf"][0].cpu().numpy()        # (H, W) confidence 1-18

        # Quality filter: must be masked AND high-confidence
        conf_threshold = max(1.5, float(np.percentile(conf_np[mask_np], 25))) if mask_np.sum() > 0 else 1.5
        good_mask = mask_np & (conf_np > conf_threshold)

        pts    = pts3d_np[good_mask]                    # (N, 3)
        colors = img_np[good_mask].astype(np.float32)  # (N, 3) already 0-1

        print(f"  View: {pts3d_np.shape[:2]}, mask={mask_np.sum():,}, "
              f"conf>{conf_threshold:.1f}={good_mask.sum():,} pts")
        print(f"  XYZ range: X=[{pts[:,0].min():.2f},{pts[:,0].max():.2f}] "
              f"Y=[{pts[:,1].min():.2f},{pts[:,1].max():.2f}] "
              f"Z=[{pts[:,2].min():.2f},{pts[:,2].max():.2f}]")

        all_pts.append(pts)
        all_colors.append(colors)

    pointcloud = np.concatenate(all_pts, axis=0)
    colors     = np.concatenate(all_colors, axis=0)
    print(f"  Total point cloud: {len(pointcloud):,} points")
    return {"pointcloud": pointcloud, "colors": colors}


# ── Frontend static serving (production build) ───────────────────────────────
# Serve Vite's built frontend at the root if dist/ exists.
# In dev mode this is handled by `npm run dev` on port 5173.
DIST_DIR = ROOT_DIR / "dist"
if DIST_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(DIST_DIR / "assets")), name="assets")

    @app.get("/")
    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str = ""):
        """Catch-all: serve index.html for any non-API route (SPA routing)."""
        # Don't catch API or /glb routes
        if full_path.startswith(("api/", "glb/", "assets/")):
            raise HTTPException(404)
        return FileResponse(str(DIST_DIR / "index.html"))


if __name__ == "__main__":
    import webbrowser, uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"\n🚀 FoodScan 3D running at http://localhost:{port}")
    print(f"   Frontend: {'built (dist/)' if DIST_DIR.exists() else 'not built — run npm run build'}")
    if DIST_DIR.exists():
        webbrowser.open(f"http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
