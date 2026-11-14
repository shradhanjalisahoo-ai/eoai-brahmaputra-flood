"""
EO-AI Labs (STAr, IIT Kanpur) - Role 01 Technical Assignment
Configuration: R5 (Brahmaputra, Assam) | S1 (Sentinel-2 L2A) | D2 (Surface Water) | T1 (Change Detection)
Layer 1-3: STAC Ingestion, Cloud/Terrain-Masked Data Cube, Change Detection & GeoParquet Catalogue
"""

import os
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
import xarray as xr
import rioxarray
import pystac_client
import planetary_computer
import odc.stac

# --- 1. RESEARCH PARAMETERS & CONFIGURATION (R5-S1-D2-T1) ---
AOI_BBOX = [92.15, 26.25, 92.38, 26.42]  # Centered on Brahmaputra main channel & Morigaon floodplain
CRS_UTM = "EPSG:32646"                   # UTM Zone 46N (meters, Assam)
RESOLUTION = 20                          # 20m uniform grid alignment
NDWI_THRESHOLD = 0.10                    # Calibrated for turbid Brahmaputra floodwater
DATA_DIR = "data"

SUB_BLOCKS = [
    {"block_name": "Mayong Circle", "bbox": [92.15, 26.25, 92.265, 26.335]},
    {"block_name": "Bhuragaon Circle", "bbox": [92.265, 26.25, 92.38, 26.335]},
    {"block_name": "Laharighat Circle", "bbox": [92.15, 26.335, 92.265, 26.42]},
    {"block_name": "Morigaon Sadar", "bbox": [92.265, 26.335, 92.38, 26.42]},
]


def query_best_s2_scene(catalog, bbox, date_window, max_cloud=40):
    """Queries Planetary Computer STAC for the least-cloudy Sentinel-2 L2A scene covering full tile T46RDQ."""
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=date_window,
        query={"eo:cloud_cover": {"lt": max_cloud}},
    )
    items = list(search.items())
    if not items:
        search = catalog.search(collections=["sentinel-2-l2a"], bbox=bbox, datetime=date_window)
        items = list(search.items())
    if not items:
        raise ValueError(f"No Sentinel-2 scenes found for window {date_window}")

    # Prefer full-coverage MGRS tile T46RDQ over edge-clipped T46RDP
    rdq_items = [it for it in items if "T46RDQ" in it.id]
    pool = rdq_items if rdq_items else items
    best_item = min(pool, key=lambda item: item.properties.get("eo:cloud_cover", 100.0))
    return best_item


def load_and_compute_ndwi(item, bbox, like_geobox=None):
    """Loads B03 (Green), B08 (NIR), and SCL (Cloud Mask) onto an aligned UTM grid and computes NDWI."""
    if like_geobox is not None:
        ds = odc.stac.load([item], bands=["B03", "B08", "SCL"], geobox=like_geobox).isel(time=0)
    else:
        ds = odc.stac.load(
            [item],
            bands=["B03", "B08", "SCL"],
            bbox=bbox,
            crs=CRS_UTM,
            resolution=RESOLUTION,
        ).isel(time=0)

    scl = ds["SCL"]
    cloud_mask = scl.isin([3, 8, 9, 10])
    valid_mask = (~cloud_mask) & (scl != 0)

    green = ds["B03"].where(valid_mask).astype("float32")
    nir = ds["B08"].where(valid_mask).astype("float32")

    ndwi = (green - nir) / (green + nir + 1e-6)
    cloud_pct = float(cloud_mask.mean().values * 100.0)
    return ndwi, valid_mask, cloud_pct, ds.odc.geobox


def load_terrain_slope_mask(catalog, bbox, target_geobox, template_da):
    """Loads Copernicus 30m DEM directly into target_geobox and masks out steep terrain (slope > 5 deg)."""
    try:
        dem_search = catalog.search(collections=["cop-dem-glo-30"], bbox=bbox)
        dem_items = list(dem_search.items())
        if not dem_items:
            return xr.ones_like(template_da, dtype=bool)

        dem_ds = odc.stac.load(dem_items, bands=["data"], geobox=target_geobox).isel(time=0)
        elev = dem_ds["data"].astype("float32").values
        dy, dx = np.gradient(elev, RESOLUTION)
        slope_deg = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
        flat_mask = (slope_deg < 5.0) | np.isnan(slope_deg)
        return xr.DataArray(flat_mask, coords=template_da.coords, dims=template_da.dims)
    except Exception as e:
        print(f"DEM fallback (using flat mask): {e}")
        return xr.ones_like(template_da, dtype=bool)


def run_pipeline(
    pre_window="2026-02-01/2026-03-20",
    post_window="2026-07-01/2026-08-25",
    ndwi_thresh=NDWI_THRESHOLD,
    run_label="2026_Monsoon_Assam",
):
    """Executes end-to-end ingestion, change detection, GeoTIFF export, and GeoParquet logging."""
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"\n[1/4] Connecting to Planetary Computer STAC for run '{run_label}'...")
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    print(f"[2/4] Querying Pre-Monsoon ({pre_window}) & Monsoon ({post_window}) Sentinel-2 scenes...")
    pre_item = query_best_s2_scene(catalog, AOI_BBOX, pre_window, max_cloud=20)
    post_item = query_best_s2_scene(catalog, AOI_BBOX, post_window, max_cloud=55)

    pre_date_str = pre_item.datetime.strftime("%Y-%m-%d")
    post_date_str = post_item.datetime.strftime("%Y-%m-%d")
    print(f"      Selected Pre-Scene : {pre_item.id} ({pre_date_str}, Cloud: {pre_item.properties.get('eo:cloud_cover', 0):.1f}%)")
    print(f"      Selected Post-Scene: {post_item.id} ({post_date_str}, Cloud: {post_item.properties.get('eo:cloud_cover', 0):.1f}%)")

    print("[3/4] Building Analysis-Ready Data Cube (Cloud Masking + DEM Slope Filter + NDWI)...")
    pre_ndwi, pre_valid, pre_cloud_pct, target_geobox = load_and_compute_ndwi(pre_item, AOI_BBOX)
    post_ndwi, post_valid, post_cloud_pct, _ = load_and_compute_ndwi(post_item, AOI_BBOX, like_geobox=target_geobox)
    flat_mask = load_terrain_slope_mask(catalog, AOI_BBOX, target_geobox, pre_ndwi)

    pre_water = (pre_ndwi > ndwi_thresh) & pre_valid & flat_mask
    post_water = (post_ndwi > ndwi_thresh) & post_valid & flat_mask

    # 0 = Dry Land, 1 = Permanent Water, 2 = Newly Flooded Land, 3 = Cloud / Unobserved
    change_map = xr.zeros_like(pre_ndwi, dtype="uint8")
    change_map = xr.where(pre_water & post_water, 1, change_map)
    change_map = xr.where((~pre_water) & post_water, 2, change_map)
    change_map = xr.where(~post_valid, 3, change_map)

    pre_ndwi.astype("float32").rio.write_crs(CRS_UTM).rio.to_raster(f"{DATA_DIR}/{run_label}_pre_ndwi.tif")
    post_ndwi.astype("float32").rio.write_crs(CRS_UTM).rio.to_raster(f"{DATA_DIR}/{run_label}_post_ndwi.tif")
    change_map.astype("uint8").rio.write_crs(CRS_UTM).rio.to_raster(f"{DATA_DIR}/{run_label}_change_map.tif")

    pixel_area_km2 = (RESOLUTION * RESOLUTION) / 1e6
    perm_km2 = float((change_map == 1).sum().values * pixel_area_km2)
    new_flood_km2 = float((change_map == 2).sum().values * pixel_area_km2)
    unobserved_pct = float((change_map == 3).mean().values * 100.0)

    print("[4/4] Updating GeoParquet Catalogue & Sub-Block Feature Store...")
    cat_record = gpd.GeoDataFrame(
        [
            {
                "run_label": run_label,
                "pre_scene_id": pre_item.id,
                "post_scene_id": post_item.id,
                "pre_date": pre_date_str,
                "post_date": post_date_str,
                "pre_cloud_pct": round(pre_cloud_pct, 2),
                "post_cloud_pct": round(post_cloud_pct, 2),
                "unobserved_aoi_pct": round(unobserved_pct, 2),
                "permanent_water_km2": round(perm_km2, 2),
                "newly_flooded_km2": round(new_flood_km2, 2),
                "ndwi_threshold": ndwi_thresh,
                "ingested_at": datetime.now(timezone.utc).isoformat(),
                "geometry": box(*AOI_BBOX),
            }
        ],
        crs="EPSG:4326",
    )

    cat_path = f"{DATA_DIR}/catalog.parquet"
    if os.path.exists(cat_path):
        existing_cat = gpd.read_parquet(cat_path)
        existing_cat = existing_cat[existing_cat["run_label"] != run_label]
        cat_record = gpd.GeoDataFrame(pd.concat([existing_cat, cat_record], ignore_index=True), crs="EPSG:4326")
    cat_record.to_parquet(cat_path)

    ny, nx = change_map.shape
    half_y, half_x = ny // 2, nx // 2
    quadrants = [
        ("Mayong Circle", change_map.isel(y=slice(half_y, ny), x=slice(0, half_x))),
        ("Bhuragaon Circle", change_map.isel(y=slice(half_y, ny), x=slice(half_x, nx))),
        ("Laharighat Circle", change_map.isel(y=slice(0, half_y), x=slice(0, half_x))),
        ("Morigaon Sadar", change_map.isel(y=slice(0, half_y), x=slice(half_x, nx))),
    ]

    block_rows = []
    for idx, (b_name, sub_da) in enumerate(quadrants):
        b_perm = float((sub_da == 1).sum().values * pixel_area_km2)
        b_new = float((sub_da == 2).sum().values * pixel_area_km2)
        b_cloud = float((sub_da == 3).mean().values * 100.0)
        surge_pct = (b_new / max(b_perm, 0.1)) * 100.0
        block_rows.append(
            {
                "run_label": run_label,
                "block_name": b_name,
                "pre_date": pre_date_str,
                "post_date": post_date_str,
                "baseline_water_km2": round(b_perm, 2),
                "newly_flooded_km2": round(b_new, 2),
                "surge_pct": round(surge_pct, 1),
                "cloud_obscured_pct": round(b_cloud, 1),
                "geometry": box(*SUB_BLOCKS[idx]["bbox"]),
            }
        )

    block_gdf = gpd.GeoDataFrame(block_rows, crs="EPSG:4326")
    block_gdf = block_gdf.sort_values(by="newly_flooded_km2", ascending=False).reset_index(drop=True)
    priorities = ["1 - Critical (Immediate Boats)", "2 - High (Relief Dispatch)", "3 - Moderate (Alert)", "4 - Monitor"]
    block_gdf["priority_tier"] = priorities[: len(block_gdf)]

    block_path = f"{DATA_DIR}/block_summary.parquet"
    if os.path.exists(block_path):
        existing_blocks = gpd.read_parquet(block_path)
        existing_blocks = existing_blocks[existing_blocks["run_label"] != run_label]
        block_gdf = gpd.GeoDataFrame(pd.concat([existing_blocks, block_gdf], ignore_index=True), crs="EPSG:4326")
    block_gdf.to_parquet(block_path)

    print(f"SUCCESS! Saved GeoTIFFs & GeoParquet files in '{DATA_DIR}/'.")
    print(f"Summary -> Baseline Water: {perm_km2:.2f} km² | Newly Flooded: {new_flood_km2:.2f} km² | Cloud Obscured: {unobserved_pct:.1f}%")


if __name__ == "__main__":
    run_pipeline(
        pre_window="2026-02-01/2026-03-20",
        post_window="2026-07-01/2026-08-25",
        run_label="2026_Monsoon_Assam",
    )
    run_pipeline(
        pre_window="2025-02-01/2025-03-20",
        post_window="2025-07-01/2025-08-25",
        run_label="2025_Monsoon_Assam",
    )
    run_pipeline(
        pre_window="2024-02-01/2024-03-20",
        post_window="2024-07-01/2024-08-15",
        run_label="2024_Monsoon_Benchmark",
    )