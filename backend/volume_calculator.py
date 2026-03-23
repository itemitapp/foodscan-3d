#!/usr/bin/env python3
"""
Volume calculator for FoodScan 3D.
Takes a metric 3D point cloud from MapAnything and returns
per-region volumes in millilitres.
"""
import numpy as np
from pathlib import Path
from typing import Dict, Any


# ──────────────────────────────────────────────────────────────────────────────
# Main volume computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_food_volumes(
    pointcloud: np.ndarray,  # (N, 3) metric coords in meters
    colors: np.ndarray,      # (N, 3) RGB 0-1
    plate_diameter_cm: float = 26.0,
) -> Dict[str, Any]:
    """
    1. Fit a plate plane using RANSAC
    2. Project points above the plate
    3. Cluster food regions by color/proximity
    4. Compute volume per cluster
    """
    if pointcloud is None or len(pointcloud) == 0:
        return {"total_volume_ml": 0.0, "segments": []}

    # Step 1: Fit the plate plane via RANSAC
    plate_normal, plate_d = fit_plane_ransac(pointcloud)

    # Step 2: Compute height above plate for each point
    heights = pointcloud @ plate_normal - plate_d  # signed distance to plane

    # Filter to points above the plate (food only)
    above_mask = heights > 0.002  # 2mm minimum height (ignore noise)
    food_pts = pointcloud[above_mask]
    food_heights = heights[above_mask]
    food_colors = colors[above_mask]

    if len(food_pts) < 50:
        # Not enough food points — return a single rough estimate
        volume_m3 = estimate_volume_convex(food_pts, food_heights)
        volume_ml = volume_m3 * 1e6  # m³ to ml
        return {
            "total_volume_ml": round(volume_ml, 1),
            "segments": [{"id": 0, "label": "Food", "volume_ml": round(volume_ml, 1), "color": "#ff9900"}],
        }

    # Step 3: Cluster food by color (simple k-means on RGB)
    try:
        from sklearn.cluster import KMeans, DBSCAN
        k = min(6, max(1, len(food_pts) // 200))
        km = KMeans(n_clusters=k, n_init=3, random_state=42)
        labels = km.fit_predict(np.hstack([food_colors, food_pts[:, :2] * 2]))  # color + XY position
    except ImportError:
        # No sklearn — use single segment
        labels = np.zeros(len(food_pts), dtype=int)
        k = 1

    # Step 4: Compute volume per segment
    SEGMENT_COLORS = [
        "#FF6B6B", "#FF9F43", "#FECA57", "#48DBFB",
        "#1DD1A1", "#54A0FF", "#5F27CD", "#C8D6E5"
    ]
    FOOD_LABELS = ["Protein", "Carbs", "Vegetables", "Sauce", "Fruit", "Grain", "Dairy", "Other"]

    segments = []
    total_ml = 0.0

    for seg_id in range(k):
        mask = labels == seg_id
        seg_pts = food_pts[mask]
        seg_heights = food_heights[mask]

        if len(seg_pts) < 10:
            continue

        vol_m3 = estimate_volume_convex(seg_pts, seg_heights)
        vol_ml = vol_m3 * 1e6

        # Representative color (mean of this cluster's pixel colors)
        mean_rgb = food_colors[mask].mean(axis=0)
        hex_color = "#{:02x}{:02x}{:02x}".format(
            int(mean_rgb[0] * 255),
            int(mean_rgb[1] * 255),
            int(mean_rgb[2] * 255),
        )

        total_ml += vol_ml
        segments.append({
            "id": seg_id,
            "label": FOOD_LABELS[seg_id % len(FOOD_LABELS)],
            "volume_ml": round(vol_ml, 1),
            "color": hex_color,
            "point_count": int(mask.sum()),
        })

    # Sort largest first
    segments.sort(key=lambda s: s["volume_ml"], reverse=True)

    # Re-number
    for i, s in enumerate(segments):
        s["percentage"] = round(s["volume_ml"] / total_ml * 100 if total_ml > 0 else 0, 1)

    return {
        "total_volume_ml": round(total_ml, 1),
        "segments": segments,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────────────────────────────────────

def fit_plane_ransac(pts: np.ndarray, iterations: int = 100, threshold: float = 0.005) -> tuple:
    """
    Fit the dominant flat surface (plate/table) in the point cloud.
    Returns (normal, d) such that pts @ normal = d for the plane.
    """
    best_normal = np.array([0.0, 0.0, 1.0])
    best_d = 0.0
    best_inliers = 0

    rng = np.random.default_rng(42)

    for _ in range(iterations):
        idx = rng.choice(len(pts), 3, replace=False)
        p0, p1, p2 = pts[idx]
        v1, v2 = p1 - p0, p2 - p0
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-8:
            continue
        normal /= norm
        d = float(pts.mean(axis=0) @ normal)

        dists = np.abs(pts @ normal - d)
        inliers = int((dists < threshold).sum())

        if inliers > best_inliers:
            best_inliers = inliers
            best_normal = normal
            best_d = d

    # Ensure normal points upward
    if best_normal[2] < 0:
        best_normal = -best_normal
        best_d = -best_d

    return best_normal, best_d


def estimate_volume_convex(pts: np.ndarray, heights: np.ndarray) -> float:
    """
    Estimate volume (m³) of a food mound above the plate.
    Uses a 2D triangulated base × height integration approach.
    """
    if len(pts) < 4:
        return 0.0

    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(pts)
        return float(hull.volume)
    except Exception:
        # Fallback: cylinder approximation
        xy = pts[:, :2]
        r = np.linalg.norm(xy - xy.mean(0), axis=1).max()
        h = float(heights.mean())
        return np.pi * r**2 * h / 3  # cone approximation


# ──────────────────────────────────────────────────────────────────────────────
# GLB export for Three.js
# ──────────────────────────────────────────────────────────────────────────────

def mesh_to_glb(pointcloud: np.ndarray, colors: np.ndarray, output_path: Path):
    """
    Convert a point cloud to a GLB file that Three.js can load.
    Uses trimesh to create a simple point cloud mesh.
    """
    try:
        import trimesh

        # Create a point cloud mesh (trimesh PointCloud)
        pc = trimesh.PointCloud(vertices=pointcloud, colors=(colors * 255).astype(np.uint8))

        scene = trimesh.Scene()
        scene.add_geometry(pc)
        glb_bytes = scene.export(file_type="glb")
        output_path.write_bytes(glb_bytes)
        print(f"Saved GLB: {output_path} ({len(glb_bytes) / 1024:.1f} KB)")

    except Exception as e:
        print(f"GLB export failed: {e}. Saving empty file.")
        output_path.write_bytes(b"")
