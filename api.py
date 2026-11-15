import base64
from io import BytesIO
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import geopandas as gpd
import rioxarray
from rasterio.warp import Resampling
from shapely.geometry import box
from PIL import Image

from src.science_engine import run_scientific_pipeline, RESOLUTION, SUB_BLOCKS

app = FastAPI(title="EO-AI Labs Brahmaputra Flood Intelligence API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class IngestRequest(BaseModel):
    run_label: str = "2025_Monsoon_Assam"
    pre_window: str = "2025-02-01/2025-03-20"
    post_window: str = "2025-07-01/2025-08-25"


@app.get("/api/runs")
def get_runs():
    try:
        df = gpd.read_parquet("data/catalog.parquet").drop(columns=["geometry"]).fillna(0.0)
        return {"runs": df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/triage/{run_label}")
def get_triage(run_label: str, threshold: float = 0.10, index: str = "MNDWI"):
    suffix = "mndwi" if index.upper() == "MNDWI" else "ndwi"
    try:
        pre_da = rioxarray.open_rasterio(f"data/{run_label}_pre_{suffix}.tif").squeeze()
        post_da = rioxarray.open_rasterio(f"data/{run_label}_post_{suffix}.tif").squeeze()
        saved_da = rioxarray.open_rasterio(f"data/{run_label}_change_map.tif").squeeze()
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Run '{run_label}' not found: {e}")

    pre_valid = ~np.isnan(pre_da.values)
    post_valid = (~np.isnan(post_da.values)) & (saved_da.values != 3)

    pre_water = (pre_da.values > threshold) & pre_valid
    post_water = (post_da.values > threshold) & post_valid

    change = np.zeros_like(saved_da.values, dtype=np.uint8)
    change[pre_water & post_water] = 1
    change[(~pre_water) & post_water] = 2
    change[~post_valid] = 3

    pixel_area_km2 = (RESOLUTION * RESOLUTION) / 1e6
    ny, nx = change.shape
    half_y, half_x = ny // 2, nx // 2
    quads = [
        ("Mayong Circle", change[half_y:ny, 0:half_x], SUB_BLOCKS[0]["bbox"]),
        ("Bhuragaon Circle", change[half_y:ny, half_x:nx], SUB_BLOCKS[1]["bbox"]),
        ("Laharighat Circle", change[0:half_y, 0:half_x], SUB_BLOCKS[2]["bbox"]),
        ("Morigaon Sadar", change[0:half_y, half_x:nx], SUB_BLOCKS[3]["bbox"]),
    ]

    rows = []
    for name, arr, bbox in quads:
        perm = float(np.sum(arr == 1) * pixel_area_km2)
        new_f = float(np.sum(arr == 2) * pixel_area_km2)
        cloud = float(np.mean(arr == 3) * 100.0)
        surge = (new_f / max(perm, 0.1)) * 100.0
        rows.append({
            "block_name": name,
            "baseline_water_km2": round(perm, 2),
            "newly_flooded_km2": round(new_f, 2),
            "surge_pct": round(surge, 1),
            "cloud_occluded_pct": round(cloud, 1),
            "geometry": box(*bbox),
        })

    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326").sort_values("newly_flooded_km2", ascending=False).reset_index(drop=True)
    tiers = ["1 - Critical (Immediate Boats)", "2 - High (Relief Dispatch)", "3 - Moderate (Alert)", "4 - Monitor"]
    gdf["priority_tier"] = tiers[:len(gdf)]

    return {
        "summary": {
            "permanent_water_km2": round(float(np.sum(change == 1) * pixel_area_km2), 2),
            "newly_flooded_km2": round(float(np.sum(change == 2) * pixel_area_km2), 2),
            "cloud_occluded_pct": round(float(np.mean(change == 3) * 100.0), 1),
        },
        "geojson": gdf.__geo_interface__,
    }


@app.get("/api/overlay/{run_label}")
def get_overlay(run_label: str, threshold: float = 0.10, index: str = "MNDWI"):
    suffix = "mndwi" if index.upper() == "MNDWI" else "ndwi"
    pre_da = rioxarray.open_rasterio(f"data/{run_label}_pre_{suffix}.tif").squeeze()
    post_da = rioxarray.open_rasterio(f"data/{run_label}_post_{suffix}.tif").squeeze()
    post_ndwi_da = rioxarray.open_rasterio(f"data/{run_label}_post_ndwi.tif").squeeze()
    post_mndwi_da = rioxarray.open_rasterio(f"data/{run_label}_post_mndwi.tif").squeeze()
    saved_da = rioxarray.open_rasterio(f"data/{run_label}_change_map.tif").squeeze()

    pre_valid = ~np.isnan(pre_da.values)
    post_valid = (~np.isnan(post_da.values)) & (saved_da.values != 3)
    pre_water = (pre_da.values > threshold) & pre_valid
    post_water = (post_da.values > threshold) & post_valid

    change = np.zeros_like(saved_da.values, dtype=np.uint8)
    change[pre_water & post_water] = 1
    change[(~pre_water) & post_water] = 2
    change[~post_valid] = 3

    change_4326 = saved_da.copy(data=change).rio.reproject("EPSG:4326", resampling=Resampling.nearest)
    b = change_4326.rio.bounds()
    arr = change_4326.values
    rgba = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
    rgba[arr == 1] = [31, 120, 180, 215]
    rgba[arr == 2] = [235, 35, 25, 240]
    rgba[arr == 3] = [150, 150, 150, 130]

    buf = BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    # Bimodal spectral histogram data (MNDWI vs NDWI)
    bins = np.linspace(-0.5, 0.5, 26)
    m_vals = post_mndwi_da.values[np.isfinite(post_mndwi_da.values)]
    n_vals = post_ndwi_da.values[np.isfinite(post_ndwi_da.values)]
    m_hist, _ = np.histogram(m_vals, bins=bins)
    n_hist, _ = np.histogram(n_vals, bins=bins)
    hist_data = [
        {"bin": round(float((bins[i] + bins[i + 1]) / 2), 2), "MNDWI": int(m_hist[i]), "NDWI": int(n_hist[i])}
        for i in range(len(m_hist))
    ]

    return {
        "image_base64": f"data:image/png;base64,{img_b64}",
        "bounds": [[b[1], b[0]], [b[3], b[2]]],
        "histogram_data": hist_data,
    }


@app.post("/api/ingest")
def trigger_ingest(req: IngestRequest):
    metrics = run_scientific_pipeline(req.pre_window, req.post_window, req.run_label)
    return {"status": "success", "run_label": req.run_label, "validation": metrics}
