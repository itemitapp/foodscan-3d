#!/usr/bin/env python3
"""
Volume calculator for FoodScan 3D.
Takes a metric 3D point cloud from MapAnything and returns
per-region volumes in millilitres.

Physics-based approach:
  1. RANSAC-fit the plate plane
  2. Clip to plate-radius so background points don't poison the estimate
  3. Integrate food height above the plate using a Delaunay triangration (like a DEM)
  4. Cluster by colour+XY for per-region volumes
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
    3. Clip to the known plate diameter
    4. Cluster food regions by color/proximity
    5. Compute volume per cluster using triangulated height integration
    """
    if pointcloud is None or len(pointcloud) == 0:
        return {"total_volume_ml": 0.0, "segments": []}

    plate_radius_m = (plate_diameter_cm / 2.0) / 100.0  # cm → m

    # Step 1: Fit the plate plane via RANSAC
    plate_normal, plate_d = fit_plane_ransac(pointcloud)

    # Step 2: Project all points onto the plate plane coordinate system
    # height = signed distance above the plate
    heights = pointcloud @ plate_normal - plate_d

    # Step 3: Clip to plate region
    # Project points onto the plate plane (in-plane XY)
    up = plate_normal
    # Build two orthogonal in-plane vectors
    arbitrary = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    right = np.cross(up, arbitrary); right /= np.linalg.norm(right)
    fwd   = np.cross(right, up);    fwd   /= np.linalg.norm(fwd)

    # In-plane coordinates of each point
    proj_x = pointcloud @ right
    proj_y = pointcloud @ fwd

    # Centre of the plate (mean of plate-plane inliers)
    plate_mask = np.abs(heights) < 0.008  # ≤8mm: plate inliers
    if plate_mask.sum() > 10:
        cx = proj_x[plate_mask].mean()
        cy = proj_y[plate_mask].mean()
    else:
        cx = proj_x.mean()
        cy = proj_y.mean()

    dist_from_center = np.sqrt((proj_x - cx)**2 + (proj_y - cy)**2)

    # Food points: above the plate AND within plate radius
    above_mask = (heights > 0.003) & (dist_from_center < plate_radius_m)
    food_pts    = pointcloud[above_mask]
    food_x      = proj_x[above_mask]
    food_y      = proj_y[above_mask]
    food_h      = heights[above_mask]
    food_colors = colors[above_mask]

    print(f"  Plate radius {plate_radius_m*100:.0f}cm: {above_mask.sum()} food pts "
          f"(of {len(pointcloud)} total)")

    if len(food_pts) < 30:
        # Very few food points — volume based on bounding cylinder
        rx = (food_x.max() - food_x.min()) / 2 if len(food_pts) > 1 else 0.05
        ry = (food_y.max() - food_y.min()) / 2 if len(food_pts) > 1 else 0.05
        h  = float(food_h.mean()) if len(food_pts) > 0 else 0.03
        vol_ml = np.pi * rx * ry * h / 2 * 1e6
        return {
            "total_volume_ml": round(vol_ml, 1),
            "segments": [{"id": 0, "label": "Food", "volume_ml": round(vol_ml, 1),
                          "color": "#ff9900", "percentage": 100.0}],
        }

    # Step 4: Cluster food by colour+position (KMeans)
    try:
        from sklearn.cluster import KMeans
        k = min(6, max(1, len(food_pts) // 500))
        features = np.hstack([
            food_colors * 2.0,           # colour weight
            food_x[:, None] * 3.0,       # position weight
            food_y[:, None] * 3.0,
        ])
        km = KMeans(n_clusters=k, n_init=3, random_state=42)
        labels = km.fit_predict(features)
    except Exception:
        labels = np.zeros(len(food_pts), dtype=int)
        k = 1

    # Step 5: Per-segment volume using triangulated height integration (DEM)
    segments = []
    total_ml  = 0.0
    food_xy   = np.column_stack([food_x, food_y])

    for seg_id in range(k):
        mask       = labels == seg_id
        seg_xy     = food_xy[mask]
        seg_h      = food_h[mask]

        if len(seg_xy) < 10:
            continue

        vol_m3 = volume_from_dem(seg_xy, seg_h)
        vol_ml = vol_m3 * 1e6  # m³ → mL

        # Spatial label
        seg_center = seg_xy.mean(axis=0)
        dist_norm  = np.linalg.norm(seg_center - food_xy.mean(axis=0)) / (plate_radius_m or 0.1)
        if dist_norm < 0.25:
            label = "Main Portion"
        elif dist_norm < 0.55:
            label = f"Component {seg_id + 1}"
        else:
            label = f"Side Item {seg_id}"

        mean_rgb  = food_colors[mask].mean(axis=0)
        hex_color = "#{:02x}{:02x}{:02x}".format(
            int(np.clip(mean_rgb[0], 0, 1) * 255),
            int(np.clip(mean_rgb[1], 0, 1) * 255),
            int(np.clip(mean_rgb[2], 0, 1) * 255),
        )

        total_ml += vol_ml
        segments.append({
            "id":          seg_id,
            "label":       label,
            "volume_ml":   round(vol_ml, 1),
            "color":       hex_color,
            "point_count": int(mask.sum()),
        })

    # Sort largest first
    segments.sort(key=lambda s: s["volume_ml"], reverse=True)
    for s in segments:
        s["percentage"] = round(s["volume_ml"] / total_ml * 100 if total_ml > 0 else 0, 1)

    return {
        "total_volume_ml": round(total_ml, 1),
        "segments":        segments,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────────────────────────────────────

def fit_plane_ransac(pts: np.ndarray, iterations: int = 150, threshold: float = 0.008) -> tuple:
    """Fit the dominant flat surface (plate/table). Returns (normal, d)."""
    best_normal   = np.array([0.0, 0.0, 1.0])
    best_d        = float(np.median(pts[:, 2]))
    best_inliers  = 0
    rng           = np.random.default_rng(42)

    for _ in range(iterations):
        idx = rng.choice(len(pts), 3, replace=False)
        p0, p1, p2 = pts[idx]
        v1, v2     = p1 - p0, p2 - p0
        normal     = np.cross(v1, v2)
        norm       = np.linalg.norm(normal)
        if norm < 1e-8:
            continue
        normal /= norm
        # Use centroid of sampled points, not full mean (faster)
        d  = float(pts[idx].mean(axis=0) @ normal)
        ds = np.abs(pts @ normal - d)
        n  = int((ds < threshold).sum())
        if n > best_inliers:
            best_inliers = n
            best_normal  = normal
            best_d       = d

    if best_normal[2] < 0:
        best_normal = -best_normal
        best_d      = -best_d

    return best_normal, best_d


def volume_from_dem(xy: np.ndarray, h: np.ndarray) -> float:
    """
    Estimate food volume by triangulating the 2D footprint and integrating heights.
    This is the correct way to integrate a height field above a flat plate.
    Returns volume in m³.
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
            # Area of the triangle in XY
            area = 0.5 * abs((p1[0]-p0[0])*(p2[1]-p0[1]) - (p2[0]-p0[0])*(p1[1]-p0[1]))
            # Average height of the three vertices
            avg_h = (h0 + h1 + h2) / 3.0
            if avg_h > 0:
                total += area * avg_h
        return total
    except Exception:
        # Cylinder fallback
        r = np.linalg.norm(xy - xy.mean(0), axis=1).max()
        return np.pi * r**2 * float(h.mean()) / 2


# ──────────────────────────────────────────────────────────────────────────────
# GLB export for Three.js — export as a voxelised surface mesh so it's visible
# ──────────────────────────────────────────────────────────────────────────────

def mesh_to_glb(pointcloud: np.ndarray, colors: np.ndarray, output_path: Path):
    """
    Convert a metric point cloud to a GLB file for Three.js.
    Builds a Delaunay surface mesh from the food points so the model is visible.
    Falls back to a raw PointCloud if scipy is unavailable.
    """
    try:
        import trimesh
        from scipy.spatial import Delaunay

        if len(pointcloud) < 4:
            raise ValueError("Not enough points")

        # Subsample for performance (max 30k points for mesh)
        if len(pointcloud) > 30000:
            idx         = np.random.choice(len(pointcloud), 30000, replace=False)
            pts         = pointcloud[idx]
            clr         = colors[idx]
        else:
            pts, clr = pointcloud, colors

        # Build Delaunay triangulation on XZ plane (top-down view)
        xy   = pts[:, [0, 2]]   # use X and Z for top-down triangulation
        tri  = Delaunay(xy)

        # Filter out huge triangles (background noise)
        verts    = pts
        faces    = tri.simplices
        edge_len = np.array([
            np.max([
                np.linalg.norm(verts[f[0]] - verts[f[1]]),
                np.linalg.norm(verts[f[1]] - verts[f[2]]),
                np.linalg.norm(verts[f[2]] - verts[f[0]]),
            ])
            for f in faces
        ])
        median_e = np.median(edge_len)
        faces    = faces[edge_len < median_e * 4]  # remove outlier triangles

        vertex_colors = (np.clip(clr, 0, 1) * 255).astype(np.uint8)
        # Assign face colors (mean of vertex colors)
        face_colors   = vertex_colors[faces].mean(axis=1).astype(np.uint8)

        mesh = trimesh.Trimesh(
            vertices=verts,
            faces=faces,
            vertex_colors=vertex_colors,
            process=False,
        )
        scene     = trimesh.Scene()
        scene.add_geometry(mesh)
        glb_bytes = scene.export(file_type="glb")
        output_path.write_bytes(glb_bytes)
        print(f"Saved surface mesh GLB: {output_path} ({len(glb_bytes) / 1024:.1f} KB), "
              f"{len(faces)} faces")

    except Exception as e:
        print(f"Surface mesh failed ({e}), falling back to PointCloud GLB")
        try:
            import trimesh
            sub = min(len(pointcloud), 50000)
            idx = np.random.choice(len(pointcloud), sub, replace=False)
            pc  = trimesh.PointCloud(
                vertices=pointcloud[idx],
                colors=(np.clip(colors[idx], 0, 1) * 255).astype(np.uint8),
            )
            scene = trimesh.Scene()
            scene.add_geometry(pc)
            output_path.write_bytes(scene.export(file_type="glb"))
        except Exception as e2:
            print(f"GLB export failed entirely: {e2}")
            output_path.write_bytes(b"")
