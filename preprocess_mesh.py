"""
Preprocesses 2yr.json (2D stormwater flood depth GeoJSON) into:
  - flood_mesh.bin : compact binary WebGL + spatial-picking buffer
  - flood_meta.json: summary statistics, depth histogram, and hotspot camera presets
"""

import json
import math
import os
import struct
import time
import numpy as np
import shapely
from shapely.geometry import Polygon as ShapelyPolygon

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_GEOJSON = os.path.join(BASE_DIR, "2yr.json")
OUTPUT_BIN = os.path.join(BASE_DIR, "flood_mesh.bin")
OUTPUT_META = os.path.join(BASE_DIR, "flood_meta.json")


def lonlat_to_merc_norm(lon_deg: float, lat_deg: float):
    mx = (lon_deg + 180.0) / 360.0
    lat_rad = math.radians(max(-85.05112878, min(85.05112878, lat_deg)))
    my = (1.0 - math.log(math.tan(lat_rad) + (1.0 / math.cos(lat_rad))) / math.pi) * 0.5
    return mx, my


def merc_norm_to_lonlat(mx: float, my: float):
    lon_deg = mx * 360.0 - 180.0
    n = math.pi - 2.0 * math.pi * my
    lat_deg = math.degrees(math.atan(0.5 * (math.exp(n) - math.exp(-n))))
    return lon_deg, lat_deg


def earclip_ring(pts):
    """Simple ear-clipping triangulator for a 2D simple polygon ring [(x, y), ...] (unclosed)."""
    n = len(pts)
    if n < 3:
        return []
    if n == 3:
        return [(0, 1, 2)]

    # Signed area to determine winding
    area2 = 0.0
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        area2 += (x1 * y2 - x2 * y1)
    idxs = list(range(n))
    if area2 < 0:
        idxs.reverse()

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    def point_in_tri(p, a, b, c):
        c1 = cross(a, b, p)
        c2 = cross(b, c, p)
        c3 = cross(c, a, p)
        return (c1 >= -1e-15) and (c2 >= -1e-15) and (c3 >= -1e-15)

    tris = []
    guard = 0
    max_guard = n * n + 20
    while len(idxs) > 3 and guard < max_guard:
        guard += 1
        m = len(idxs)
        ear_found = False
        for i in range(m):
            i_prev = idxs[(i - 1) % m]
            i_curr = idxs[i]
            i_next = idxs[(i + 1) % m]
            a, b, c = pts[i_prev], pts[i_curr], pts[i_next]
            if cross(a, b, c) <= 1e-16:
                continue
            any_inside = False
            for j in range(m):
                ij = idxs[j]
                if ij in (i_prev, i_curr, i_next):
                    continue
                if point_in_tri(pts[ij], a, b, c):
                    any_inside = True
                    break
            if not any_inside:
                tris.append((i_prev, i_curr, i_next))
                idxs.pop(i)
                ear_found = True
                break
        if not ear_found:
            # Fallback fan if degenerate self-touching ring
            for k in range(1, len(idxs) - 1):
                tris.append((idxs[0], idxs[k], idxs[k + 1]))
            return tris

    if len(idxs) == 3:
        tris.append((idxs[0], idxs[1], idxs[2]))
    return tris


def main():
    t0 = time.time()
    print(f"Loading {INPUT_GEOJSON} ...")
    with open(INPUT_GEOJSON, "r", encoding="utf-8") as f:
        gj = json.load(f)

    features = gj.get("features", [])
    print(f"Loaded {len(features):,} features in {time.time() - t0:.2f}s")

    # First pass: deduplicate vertices and triangulate polygons
    vert_map = {}
    verts_lonlat = []
    tris_a = []
    tris_b = []
    tris_c = []
    tris_depth = []
    poly_depths = []

    min_lon, min_lat = 999.0, 999.0
    max_lon, max_lat = -999.0, -999.0
    complex_polys = 0

    def get_vid(lon, lat):
        # Round to 1e-9 deg (~0.1 mm) to merge identical shared mesh vertices cleanly
        key = (round(lon, 9), round(lat, 9))
        vid = vert_map.get(key)
        if vid is None:
            vid = len(verts_lonlat)
            vert_map[key] = vid
            verts_lonlat.append((lon, lat))
        return vid

    has_cdt = hasattr(shapely, "constrained_delaunay_triangles")

    for ft in features:
        geom = ft.get("geometry")
        if not geom:
            continue
        props = ft.get("properties") or {}
        d = float(props.get("DEPTH2D", 0.0))
        if d <= 0.0:
            continue
        poly_depths.append(d)

        gtype = geom.get("type")
        polys = [geom["coordinates"]] if gtype == "Polygon" else (geom["coordinates"] if gtype == "MultiPolygon" else [])

        for rings in polys:
            if not rings:
                continue
            outer = rings[0]
            # Remove closing duplicate vertex(es)
            pts_ll = []
            for c in outer:
                lon, lat = float(c[0]), float(c[1])
                if lon < min_lon: min_lon = lon
                if lon > max_lon: max_lon = lon
                if lat < min_lat: min_lat = lat
                if lat > max_lat: max_lat = lat
                if not pts_ll or (abs(lon - pts_ll[-1][0]) > 1e-11 or abs(lat - pts_ll[-1][1]) > 1e-11):
                    pts_ll.append((lon, lat))
            if len(pts_ll) > 1 and abs(pts_ll[0][0] - pts_ll[-1][0]) <= 1e-11 and abs(pts_ll[0][1] - pts_ll[-1][1]) <= 1e-11:
                pts_ll.pop()

            n_pts = len(pts_ll)
            if n_pts < 3:
                continue
            elif n_pts == 3 and len(rings) == 1:
                v0 = get_vid(*pts_ll[0])
                v1 = get_vid(*pts_ll[1])
                v2 = get_vid(*pts_ll[2])
                if v0 != v1 and v1 != v2 and v2 != v0:
                    tris_a.append(v0)
                    tris_b.append(v1)
                    tris_c.append(v2)
                    tris_depth.append(d)
            else:
                complex_polys += 1
                used_shapely = False
                if has_cdt and (len(rings) > 1 or n_pts > 16):
                    try:
                        sp = ShapelyPolygon(outer, rings[1:] if len(rings) > 1 else None)
                        if not sp.is_valid:
                            sp = sp.buffer(0)
                        cdt = shapely.constrained_delaunay_triangles(sp)
                        for tri_geom in cdt.geoms:
                            tc = list(tri_geom.exterior.coords)
                            if len(tc) >= 4:
                                v0 = get_vid(tc[0][0], tc[0][1])
                                v1 = get_vid(tc[1][0], tc[1][1])
                                v2 = get_vid(tc[2][0], tc[2][1])
                                if v0 != v1 and v1 != v2 and v2 != v0:
                                    tris_a.append(v0)
                                    tris_b.append(v1)
                                    tris_c.append(v2)
                                    tris_depth.append(d)
                        used_shapely = True
                    except Exception:
                        used_shapely = False

                if not used_shapely:
                    vids = [get_vid(lon, lat) for lon, lat in pts_ll]
                    local_tris = earclip_ring(pts_ll)
                    for i0, i1, i2 in local_tris:
                        v0, v1, v2 = vids[i0], vids[i1], vids[i2]
                        if v0 != v1 and v1 != v2 and v2 != v0:
                            tris_a.append(v0)
                            tris_b.append(v1)
                            tris_c.append(v2)
                            tris_depth.append(d)

    num_verts = len(verts_lonlat)
    num_tris = len(tris_depth)
    print(f"Triangulated {len(poly_depths):,} polygons ({complex_polys:,} non-triangles) -> {num_verts:,} unique verts, {num_tris:,} triangles")

    # Convert vertices to normalized Web Mercator relative to top-left (min_lon, max_lat)
    mx0, my0 = lonlat_to_merc_norm(min_lon, max_lat)
    mx1, my1 = lonlat_to_merc_norm(max_lon, min_lat)

    vert_xy = np.empty((num_verts, 2), dtype=np.float32)
    for i, (lon, lat) in enumerate(verts_lonlat):
        mx, my = lonlat_to_merc_norm(lon, lat)
        vert_xy[i, 0] = mx - mx0
        vert_xy[i, 1] = my - my0

    tri_idx = np.empty((num_tris, 3), dtype=np.uint32)
    tri_idx[:, 0] = np.asarray(tris_a, dtype=np.uint32)
    tri_idx[:, 1] = np.asarray(tris_b, dtype=np.uint32)
    tri_idx[:, 2] = np.asarray(tris_c, dtype=np.uint32)

    tri_depth_arr = np.asarray(tris_depth, dtype=np.float32)

    # Sort triangles ascending by depth so deeper flood triangles render cleanly on top if any overlap
    order = np.argsort(tri_depth_arr, kind="stable")
    tri_idx = tri_idx[order]
    tri_depth_arr = tri_depth_arr[order]

    # Compute smooth vertex depth (area-weighted average of incident triangles)
    p0 = vert_xy[tri_idx[:, 0]].astype(np.float64)
    p1 = vert_xy[tri_idx[:, 1]].astype(np.float64)
    p2 = vert_xy[tri_idx[:, 2]].astype(np.float64)
    tri_areas = np.abs((p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1]) - (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1])) * 0.5
    weights = np.maximum(tri_areas, 1e-18)

    v_weighted_sum = np.zeros(num_verts, dtype=np.float64)
    v_weight_tot = np.zeros(num_verts, dtype=np.float64)
    wd = weights * tri_depth_arr.astype(np.float64)
    for k in range(3):
        np.add.at(v_weighted_sum, tri_idx[:, k], wd)
        np.add.at(v_weight_tot, tri_idx[:, k], weights)

    vert_depth = np.where(v_weight_tot > 0, v_weighted_sum / np.maximum(v_weight_tot, 1e-30), 0.0).astype(np.float32)

    # Write binary file
    with open(OUTPUT_BIN, "wb") as bf:
        header = struct.pack("<4sIIIdd", b"FL2D", 1, num_verts, num_tris, mx0, my0)
        bf.write(header)
        bf.write(vert_xy.tobytes())
        bf.write(vert_depth.tobytes())
        bf.write(tri_idx.tobytes())
        bf.write(tri_depth_arr.tobytes())

    bin_mb = os.path.getsize(OUTPUT_BIN) / (1024 * 1024)
    print(f"Wrote {OUTPUT_BIN} ({bin_mb:.2f} MB)")

    # Discover distinct spatial flood hotspots across the corridor (minimum 900m separation)
    # Score 150m grid cells by (p95 depth * log1p(wet_area)) to find real high-impact flood zones
    tri_cx = (p0[:, 0] + p1[:, 0] + p2[:, 0]) / 3.0
    tri_cy = (p0[:, 1] + p1[:, 1] + p2[:, 1]) / 3.0
    span_x = float(mx1 - mx0)
    span_y = float(my1 - my0)

    gx = np.clip(((tri_cx / span_x) * 28).astype(int), 0, 27)
    gy = np.clip(((tri_cy / span_y) * 64).astype(int), 0, 63)
    cell_ids = gy * 28 + gx

    candidates = []
    for cid in np.unique(cell_ids):
        mask = cell_ids == cid
        cnt = int(np.sum(mask))
        if cnt < 80:
            continue
        ds = tri_depth_arr[mask]
        max_d = float(np.max(ds))
        p90_d = float(np.percentile(ds, 90))
        deep_cnt = int(np.sum(ds >= 1.5))
        if deep_cnt < 25:
            continue
        score = p90_d * math.log1p(deep_cnt) + 0.35 * max_d
        # Center at the centroid of deep triangles in this cell
        deep_mask = mask & (tri_depth_arr >= p90_d * 0.75)
        cx_m = float(np.mean(tri_cx[deep_mask])) + mx0
        cy_m = float(np.mean(tri_cy[deep_mask])) + my0
        clon, clat = merc_norm_to_lonlat(cx_m, cy_m)
        candidates.append({
            "score": score,
            "lon": round(clon, 6),
            "lat": round(clat, 6),
            "maxDepth": round(max_d, 2),
            "p90Depth": round(p90_d, 2),
            "deepCells": deep_cnt,
        })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    hotspots = []
    for c in candidates:
        # Enforce >= 0.009 deg (~1 km) separation so presets span the watershed
        too_close = any(
            math.hypot((c["lon"] - h["lon"]) * 82.0, (c["lat"] - h["lat"]) * 111.0) < 0.95
            for h in hotspots
        )
        if not too_close:
            hotspots.append(c)
    if len(hotspots) >= 5:
        break

    hotspot_labels = [
        ("W Rowan St / Haymount", "Cross Creek channel & overbank"),
        ("Country Club North", "Warrenwood northern tributary"),
        ("Green St / Downtown", "Historic Cross Creek crossing"),
        ("Edgecombe / Chestnut Hill", "Broad neighborhood inundation"),
        ("Rosehill Rd Corridor", "Mid-watershed residential basin"),
    ]
    for idx, h in enumerate(hotspots):
        if idx < len(hotspot_labels):
            h["name"] = hotspot_labels[idx][0]
            h["subtitle"] = hotspot_labels[idx][1]

    # Compute histogram across the 7 calibrated depth bands
    bands = [
        {"min": 0.00, "max": 0.10, "label": "0.00 – 0.10", "stage": "Trace / Sheet Film"},
        {"min": 0.10, "max": 0.25, "label": "0.10 – 0.25", "stage": "Gutter Ponding"},
        {"min": 0.25, "max": 0.50, "label": "0.25 – 0.50", "stage": "Curb Overtopping"},
        {"min": 0.50, "max": 1.00, "label": "0.50 – 1.00", "stage": "Roadway Crown"},
        {"min": 1.00, "max": 2.00, "label": "1.00 – 2.00", "stage": "Vehicle Stall Risk"},
        {"min": 2.00, "max": 5.00, "label": "2.00 – 5.00", "stage": "Structural Inundation"},
        {"min": 5.00, "max": 11.0, "label": "5.00 – 10.5", "stage": "Deep Channel / Crest"},
    ]
    pd_np = np.asarray(poly_depths, dtype=np.float32)
    total_polys = len(pd_np)
    for b in bands:
        if b["max"] >= 11.0:
            c = int(np.sum(pd_np >= b["min"]))
        else:
            c = int(np.sum((pd_np >= b["min"]) & (pd_np < b["max"])))
        b["count"] = c
        b["pct"] = round(100.0 * c / max(1, total_polys), 1)

    # Convert total wet mesh area to acres / sq km (approx scale at 35.09 deg N)
    # 1 normalized Mercator unit = 2 * pi * R_earth * cos(lat) meters
    cos_lat = math.cos(math.radians((min_lat + max_lat) * 0.5))
    meters_per_merc = 40075016.68557849 * cos_lat
    total_area_m2 = float(np.sum(tri_areas)) * (meters_per_merc ** 2)
    total_area_acres = total_area_m2 / 4046.8564224

    meta = {
        "sourceFile": "2yr.json",
        "eventLabel": "2-Year Design Storm",
        "featureCount": total_polys,
        "vertexCount": num_verts,
        "triangleCount": num_tris,
        "wetAreaAcres": round(total_area_acres, 1),
        "wetAreaSqKm": round(total_area_m2 / 1e6, 2),
        "bbox": {
            "west": round(min_lon, 6),
            "south": round(min_lat, 6),
            "east": round(max_lon, 6),
            "north": round(max_lat, 6),
        },
        "mercBounds": {
            "mx0": mx0,
            "my0": my0,
            "mxSpan": span_x,
            "mySpan": span_y,
        },
        "center": {
            "lon": round((min_lon + max_lon) * 0.5, 6),
            "lat": round((min_lat + max_lat) * 0.5, 6),
        },
        "depthStats": {
            "min": round(float(np.min(pd_np)), 4),
            "p25": round(float(np.percentile(pd_np, 25)), 3),
            "median": round(float(np.median(pd_np)), 3),
            "p75": round(float(np.percentile(pd_np, 75)), 3),
            "p90": round(float(np.percentile(pd_np, 90)), 3),
            "p95": round(float(np.percentile(pd_np, 95)), 3),
            "p99": round(float(np.percentile(pd_np, 99)), 3),
            "max": round(float(np.max(pd_np)), 2),
        },
        "bands": bands,
        "hotspots": hotspots,
    }

    with open(OUTPUT_META, "w", encoding="utf-8") as mf:
        json.dump(meta, mf, indent=2)

    print(f"Wrote {OUTPUT_META} in {time.time() - t0:.2f}s total.")
    print("Hotspots:", json.dumps(hotspots, indent=2))


if __name__ == "__main__":
    main()
