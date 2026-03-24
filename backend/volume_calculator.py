#!/usr/bin/env python3
"""
Volume calculator for FoodScan 3D — v4 (Z-histogram depth approach).

Key insight from MapAnything coordinate system:
  - Camera is at origin, food/plate scene is in +Z direction
  - Z is the depth axis (closer = smaller Z)
  - X is horizontal (left-right), Y is vertical (up-down)
  - Food SITS ON TOP of the plate = food has SMALLER Z than plate
  - The plate appears as the dominant flat surface (Z histogram peak)

Algorithm:
  1. Find plate depth via Z-histogram peak of center region
  2. Food = points closer to camera than the plate (Z < plate_z - threshold)
  3. Clip food to XY extent matching the plate diameter
  4. Integrate height using Delaunay DEM on the XY plane
  5. Cluster by colour+XY for per-region volumes
"""
import numpy as np
from pathlib import Path
from typing import Dict, Any


def compute_food_volumes(
    pointcloud: np.ndarray,   # (N, 3) metric world coords — Z = depth from camera
    colors: np.ndarray,       # (N, 3) RGB 0-1
    plate_diameter_cm: float  = 26.0,
    segments: int             = 0,
    min_food_height_mm: float = 3.0,
) -> Dict[str, Any]:
    if pointcloud is None or len(pointcloud) < 50:
        return {"total_volume_ml": 0.0, "segments": []}

    plate_radius_m = (plate_diameter_cm / 2.0) / 100.0
    min_height_m   = max(0.0005, min_food_height_mm / 1000.0)

    # ── Step 1: Find plate depth using Z histogram ─────────────────────────
    # The plate is the dominant flat surface — it will be the tallest peak
    # in the depth histogram. Use the central 50% of XY to avoid edges.
    x, y, z = pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2]
    xc, yc  = float(np.median(x)), float(np.median(y))
    # Use points within 60% of the XY range from center (plate region)
    xr = (x.max() - x.min()) * 0.30
    yr = (y.max() - y.min()) * 0.30
    center_mask = (np.abs(x - xc) < xr) & (np.abs(y - yc) < yr)
    z_center = z[center_mask]

    if len(z_center) > 100:
        # Histogram: 80 bins over Z range, find the FIRST (nearest) prominent peak
        # which corresponds to the plate/table surface closest to camera.
        hist, bins = np.histogram(z_center, bins=80)
        # Smooth histogram to find peaks
        from scipy.ndimage import uniform_filter1d
        smooth = uniform_filter1d(hist.astype(float), size=3)
        # Find the dominant peak closest to the camera (smallest Z = smallest bin idx with high count)
        # Use a threshold of 10% of max count
        threshold = smooth.max() * 0.10
        candidates = np.where(smooth > threshold)[0]
        if len(candidates) > 0:
            # Use the FIRST peak cluster (nearest to camera)
            peak_idx = int(candidates[0])
            plate_z  = float((bins[peak_idx] + bins[peak_idx + 1]) / 2)
        else:
            peak_idx = int(np.argmax(hist))
            plate_z  = float((bins[peak_idx] + bins[peak_idx + 1]) / 2)
    else:
        plate_z = float(np.percentile(z, 60))

    print(f"  Plate depth (Z): {plate_z:.4f} m")
    print(f"  Center region: {center_mask.sum():,}/{len(pointcloud):,} pts")

    # ── Step 2: Select food points ─────────────────────────────────────────
    # Food is CLOSER to camera (smaller Z) than the plate by at least min_height
    # AND within the plate's XY extent (plate radius around XY centroid)

    # XY centroid from plate-depth inliers
    plate_inliers = center_mask & (np.abs(z - plate_z) < 0.020)
    if plate_inliers.sum() > 20:
        food_cx = float(x[plate_inliers].mean())
        food_cy = float(y[plate_inliers].mean())
    else:
        food_cx, food_cy = xc, yc

    dist_xy = np.sqrt((x - food_cx)**2 + (y - food_cy)**2)

    food_z_threshold = plate_z - min_height_m  # closer than this = food
    above_mask = (z < food_z_threshold) & (dist_xy < plate_radius_m)

    food_x      = x[above_mask]
    food_y      = y[above_mask]
    food_z_vals = z[above_mask]
    food_h      = plate_z - food_z_vals   # positive height above plate
    food_colors = colors[above_mask]

    print(f"  Plate centroid XY: ({food_cx:.3f}, {food_cy:.3f})")
    print(f"  Food threshold: Z < {food_z_threshold:.4f} m, radius < {plate_radius_m*100:.0f} cm")
    print(f"  Food points: {above_mask.sum():,}")
    if len(food_h) > 0:
        print(f"  Food height range: {food_h.min()*10:.1f}mm – {food_h.max()*100:.1f}cm")

    if len(food_x) < 30:
        print("  WARNING: Too few food pts — trying wider radius")
        # Try 2× radius and see if that helps
        food_mask2 = (z < food_z_threshold) & (dist_xy < plate_radius_m * 2)
        if food_mask2.sum() > 30:
            above_mask = food_mask2
            food_x      = x[above_mask]
            food_y      = y[above_mask]
            food_z_vals = z[above_mask]
            food_h      = plate_z - food_z_vals
            food_colors = colors[above_mask]
            print(f"  Extended radius: {above_mask.sum():,} food pts")
        else:
            return {"total_volume_ml": 0.0, "segments": [
                {"id": 0, "label": "No food detected", "volume_ml": 0.0,
                 "color": "#888888", "percentage": 100.0}
            ]}

    # ── Step 3: Cluster food by colour+XY ─────────────────────────────────
    food_xy = np.column_stack([food_x, food_y])
    try:
        from sklearn.cluster import KMeans
        if segments and 1 <= segments <= 8:
            k = segments
        else:
            k = min(6, max(1, len(food_x) // 500))
        features = np.hstack([
            food_colors * 2.0,
            food_x[:, None] * 3.0,
            food_y[:, None] * 3.0,
        ])
        km     = KMeans(n_clusters=k, n_init=3, random_state=42)
        labels = km.fit_predict(features)
    except Exception:
        labels = np.zeros(len(food_x), dtype=int)
        k = 1

    # ── Step 4: Per-segment volume via Delaunay DEM on XY ─────────────────
    total_ml     = 0.0
    segments_out = []
    fc_xy        = food_xy.mean(axis=0)
    max_dist_xy  = np.linalg.norm(food_xy - fc_xy, axis=1).max() or 0.001

    for seg_id in range(k):
        mask_s  = labels == seg_id
        seg_xy  = food_xy[mask_s]
        seg_h   = food_h[mask_s]

        if len(seg_xy) < 10:
            continue

        vol_m3  = volume_from_dem(seg_xy, seg_h)
        vol_ml  = vol_m3 * 1e6

        seg_ctr  = seg_xy.mean(axis=0)
        rel_dist = np.linalg.norm(seg_ctr - fc_xy) / max_dist_xy

        if rel_dist < 0.25:
            label = "Main Portion"
        elif rel_dist < 0.55:
            label = f"Food Item {seg_id + 1}"
        else:
            label = f"Side Item {seg_id}"

        mean_rgb  = food_colors[mask_s].mean(axis=0)
        hex_color = "#{:02x}{:02x}{:02x}".format(
            int(np.clip(mean_rgb[0], 0, 1) * 255),
            int(np.clip(mean_rgb[1], 0, 1) * 255),
            int(np.clip(mean_rgb[2], 0, 1) * 255),
        )

        total_ml += vol_ml
        segments_out.append({
            "id":          seg_id,
            "label":       label,
            "volume_ml":   round(vol_ml, 1),
            "color":       hex_color,
            "point_count": int(mask_s.sum()),
        })

    segments_out.sort(key=lambda s: s["volume_ml"], reverse=True)
    for s in segments_out:
        s["percentage"] = round(s["volume_ml"] / total_ml * 100 if total_ml > 0 else 0, 1)

    print(f"  Total volume: {total_ml:.1f} ml, segments: {len(segments_out)}")
    return {"total_volume_ml": round(total_ml, 1), "segments": segments_out}


# ──────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────────────────────────────────────

def volume_from_dem(xy: np.ndarray, h: np.ndarray, max_edge_m: float = 0.03) -> float:
    """
    Integrate height field above the plate using Delaunay triangulation.
    Filters out triangles with any edge longer than max_edge_m (default=3cm),
    which removes large interpolated triangles over empty plate regions while
    keeping small triangles from densely sampled food areas.
    """
    if len(xy) < 4:
        return 0.0
    try:
        from scipy.spatial import Delaunay
        tri   = Delaunay(xy)
        total = 0.0
        
        for simplex in tri.simplices:
            p0, p1, p2 = xy[simplex]
            h0, h1, h2 = h[simplex]
            # Skip triangles with any edge longer than max_edge_m
            if (np.linalg.norm(p1-p0) > max_edge_m or
                np.linalg.norm(p2-p1) > max_edge_m or
                np.linalg.norm(p0-p2) > max_edge_m):
                continue
            area  = 0.5 * abs((p1[0]-p0[0])*(p2[1]-p0[1]) - (p2[0]-p0[0])*(p1[1]-p0[1]))
            avg_h = (h0 + h1 + h2) / 3.0
            if avg_h > 0:
                total += area * avg_h
        return total
    except Exception:
        r = np.linalg.norm(xy - xy.mean(0), axis=1).max()
        return np.pi * r**2 * float(h.mean()) / 2


# ──────────────────────────────────────────────────────────────────────────────
# GLB export — visible colored surface mesh
# ──────────────────────────────────────────────────────────────────────────────

def mesh_to_glb(pointcloud: np.ndarray, colors: np.ndarray, output_path: Path):
    """
    Export a colored surface mesh GLB file for Three.js.
    Uses Delaunay on the XY plane (since camera looks along Z).
    Filters out large background triangles.
    """
    try:
        import trimesh
        from scipy.spatial import Delaunay

        if len(pointcloud) < 4:
            raise ValueError("Not enough points")

        # Subsample for performance (center-weighted)
        n = min(len(pointcloud), 20000)
        if len(pointcloud) > n:
            cx, cy = float(pointcloud[:, 0].mean()), float(pointcloud[:, 1].mean())
            dist   = np.sqrt((pointcloud[:, 0] - cx)**2 + (pointcloud[:, 1] - cy)**2)
            w      = 1.0 / (dist + 0.01)
            w     /= w.sum()
            idx    = np.random.choice(len(pointcloud), n, replace=False, p=w)
        else:
            idx = np.arange(len(pointcloud))
        pts = pointcloud[idx]
        clr = colors[idx]

        # Delaunay on XY (camera looks along Z so XY = image plane)
        xy    = pts[:, :2]
        tri   = Delaunay(xy)
        faces = tri.simplices

        # Filter huge triangles (depth discontinuities / background)
        edge_len = np.array([
            max(np.linalg.norm(pts[f[0]] - pts[f[1]]),
                np.linalg.norm(pts[f[1]] - pts[f[2]]),
                np.linalg.norm(pts[f[2]] - pts[f[0]]))
            for f in faces
        ])
        threshold = np.median(edge_len) * 3.0
        faces     = faces[edge_len < threshold]

        vertex_colors = (np.clip(clr, 0, 1) * 255).astype(np.uint8)
        mesh = trimesh.Trimesh(
            vertices=pts,
            faces=faces,
            vertex_colors=vertex_colors,
            process=False,
        )
        scene     = trimesh.Scene()
        scene.add_geometry(mesh)
        glb_bytes = scene.export(file_type="glb")
        output_path.write_bytes(glb_bytes)
        print(f"  GLB saved: {output_path} ({len(glb_bytes)/1024:.1f}KB), "
              f"{len(faces)} faces, {len(pts)} verts")

    except Exception as e:
        print(f"  Surface mesh failed ({e}), falling back to PointCloud GLB")
        try:
            import trimesh
            n   = min(len(pointcloud), 50000)
            idx = np.random.choice(len(pointcloud), n, replace=False)
            pc  = trimesh.PointCloud(
                vertices=pointcloud[idx],
                colors=(np.clip(colors[idx], 0, 1) * 255).astype(np.uint8),
            )
            scene = trimesh.Scene()
            scene.add_geometry(pc)
            output_path.write_bytes(scene.export(file_type="glb"))
        except Exception as e2:
            print(f"  GLB export failed: {e2}")
            output_path.write_bytes(b"")
