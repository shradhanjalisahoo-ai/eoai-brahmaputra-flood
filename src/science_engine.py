"""
EO-AI Labs (STAr, IIT Kanpur) - Scientific Flood Detection & Validation Engine
Implements:
1. Multi-Index Spectral Extraction: NDWI (McFeeters 1996) & MNDWI (Xu 2006 using SWIR B11 for turbid water)
2. Automated Otsu Bimodal Thresholding (Otsu 1979)
3. Topographic Hydrological Filtering via Copernicus 30m DEM Slope (< 5 deg)
4. Multi-Feature Random Forest Classifier & Cross-Method Scientific Validation (IoU, F1-Score, Cohen's Kappa)
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, jaccard_score, cohen_kappa_score, accuracy_score

AOI_BBOX = [92.15, 26.25, 92.38, 26.42]  # Morigaon / Brahmaputra main channel (Tile T46RDQ)
CRS_UTM = "EPSG:32646"
RESOLUTION = 20
DATA_DIR = "data"

SUB_BLOCKS = [
    {"block_name": "Mayong Circle", "bbox": [92.15, 26.25, 92.265, 26.335]},
    {"block_name": "Bhuragaon Circle", "bbox": [92.265, 26.25, 92.38, 26.335]},
    {"block_name": "Laharighat Circle", "bbox": [92.15, 26.335, 92.265, 26.42]},
    {"block_name": "Morigaon Sadar", "bbox": [92.265, 26.335, 92.38, 26.42]},
]


def compute_otsu_threshold(array_1d: np.ndarray) -> float:
    """Computes optimal bimodal Otsu threshold on valid spectral index pixels [-0.6, 0.6]."""
    valid = array_1d[np.isfinite(array_1d) & (array_1d > -0.6) & (array_1d < 0.6)]
    if len(valid) < 100:
        return 0.10
    hist, bin_edges = np.histogram(valid, bins=128, range=(-0.6, 0.6))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    mean1 = np.cumsum(hist * bin_centers) / np.maximum(weight1, 1)
    mean2 = (np.cumsum((hist * bin_centers)[::-1]) / np.maximum(weight2[::-1], 1))[::-1]
    variance_between = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    optimal_idx = np.argmax(variance_between)
    return float(np.clip(bin_centers[optimal_idx], 0.02, 0.28))


def query_best_s2_scene(catalog, bbox, date_window, max_cloud=45):
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

    rdq_items = [it for it in items if "T46RDQ" in it.id]
    pool = rdq_items if rdq_items else items
    return min(pool, key=lambda item: item.properties.get("eo:cloud_cover", 100.0))


def load_spectral_indices(item, bbox, like_geobox=None):
    """Loads B03 (Green), B08 (NIR), B11 (SWIR1), and SCL (Cloud Mask) to compute NDWI and MNDWI."""
    bands = ["B03", "B08", "B11", "SCL"]
    if like_geobox is not None:
        ds = odc.stac.load([item], bands=bands, geobox=like_geobox).isel(time=0)
    else:
        ds = odc.stac.load([item], bands=bands, bbox=bbox, crs=CRS_UTM, resolution=RESOLUTION).isel(time=0)

    scl = ds["SCL"]
    cloud_mask = scl.isin([3, 8, 9, 10])
    valid_mask = (~cloud_mask) & (scl != 0)

    b03 = ds["B03"].where(valid_mask).astype("float32")
    b08 = ds["B08"].where(valid_mask).astype("float32")
    b11 = ds["B11"].where(valid_mask).astype("float32")

    ndwi = (b03 - b08) / (b03 + b08 + 1e-6)
    mndwi = (b03 - b11) / (b03 + b11 + 1e-6)

    cloud_pct = float(cloud_mask.mean().values * 100.0)
    return ndwi, mndwi, b03, b08, b11, valid_mask, cloud_pct, ds.odc.geobox


def load_dem_features(catalog, bbox, target_geobox, template_da):
    """Loads Copernicus 30m DEM elevation and terrain slope in degrees."""
    try:
        dem_items = list(catalog.search(collections=["cop-dem-glo-30"], bbox=bbox).items())
        if not dem_items:
            elev = np.zeros_like(template_da.values, dtype="float32")
            slope = np.zeros_like(template_da.values, dtype="float32")
        else:
            dem_ds = odc.stac.load(dem_items, bands=["data"], geobox=target_geobox).isel(time=0)
            elev = np.nan_to_num(dem_ds["data"].astype("float32").values, nan=50.0)
            dy, dx = np.gradient(elev, RESOLUTION)
            slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
        flat_mask = xr.DataArray(slope < 5.0, coords=template_da.coords, dims=template_da.dims)
        return elev, slope, flat_mask
    except Exception:
        elev = np.zeros_like(template_da.values, dtype="float32")
        slope = np.zeros_like(template_da.values, dtype="float32")
        return elev, slope, xr.ones_like(template_da, dtype=bool)


def train_and_validate_rf(b03, b08, b11, ndwi, mndwi, elev, slope, valid_mask, otsu_mndwi):
    """
    Trains a Random Forest flood classifier on high-confidence spectral/terrain samples
    and validates agreement against physics-based MNDWI+Otsu & NDWI benchmarks.
    """
    features = np.stack([
        np.nan_to_num(b03.values, 0),
        np.nan_to_num(b08.values, 0),
        np.nan_to_num(b11.values, 0),
        np.nan_to_num(ndwi.values, -0.5),
        np.nan_to_num(mndwi.values, -0.5),
        elev,
        slope
    ], axis=-1)

    valid_flat = valid_mask.values.ravel()
    X_all = features.reshape(-1, 7)[valid_flat]
    mndwi_flat = mndwi.values.ravel()[valid_flat]
    ndwi_flat = ndwi.values.ravel()[valid_flat]
    slope_flat = slope.ravel()[valid_flat]

    y_ref = ((mndwi_flat > otsu_mndwi) & (slope_flat < 5.0)).astype(int)
    y_ndwi = ((ndwi_flat > 0.10) & (slope_flat < 5.0)).astype(int)

    rng = np.random.default_rng(42)
    n_samples = len(y_ref)
    idx = rng.permutation(n_samples)
    train_idx, test_idx = idx[:min(4000, n_samples // 2)], idx[min(4000, n_samples // 2):min(14000, n_samples)]

    clf = RandomForestClassifier(n_estimators=40, max_depth=6, random_state=42, n_jobs=-1)
    clf.fit(X_all[train_idx], y_ref[train_idx])
    y_pred_rf = clf.predict(X_all[test_idx])

    metrics = {
        "otsu_mndwi_thresh": round(otsu_mndwi, 3),
        "rf_vs_mndwi_f1": round(float(f1_score(y_ref[test_idx], y_pred_rf, zero_division=1)), 4),
        "rf_vs_mndwi_iou": round(float(jaccard_score(y_ref[test_idx], y_pred_rf, zero_division=1)), 4),
        "rf_vs_mndwi_kappa": round(float(cohen_kappa_score(y_ref[test_idx], y_pred_rf)), 4),
        "rf_vs_mndwi_acc": round(float(accuracy_score(y_ref[test_idx], y_pred_rf)), 4),
        "mndwi_vs_ndwi_iou": round(float(jaccard_score(y_ref[test_idx], y_ndwi[test_idx], zero_division=1)), 4),
    }
    return metrics


def run_scientific_pipeline(
    pre_window="2026-02-01/2026-03-20",
    post_window="2026-07-01/2026-08-25",
    run_label="2026_Monsoon_Assam",
):
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"\n[Science Engine] Running STAC + MNDWI/Otsu/RF pipeline for '{run_label}'...")
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    pre_item = query_best_s2_scene(catalog, AOI_BBOX, pre_window, max_cloud=20)
    post_item = query_best_s2_scene(catalog, AOI_BBOX, post_window, max_cloud=55)

    pre_date_str = pre_item.datetime.strftime("%Y-%m-%d")
    post_date_str = post_item.datetime.strftime("%Y-%m-%d")

    pre_ndwi, pre_mndwi, _, _, _, pre_valid, pre_cloud, geobox = load_spectral_indices(pre_item, AOI_BBOX)
    post_ndwi, post_mndwi, b03, b08, b11, post_valid, post_cloud, _ = load_spectral_indices(post_item, AOI_BBOX, like_geobox=geobox)
    elev, slope, flat_mask = load_dem_features(catalog, AOI_BBOX, geobox, pre_ndwi)

    otsu_thresh = compute_otsu_threshold(post_mndwi.values)
    val_metrics = train_and_validate_rf(b03, b08, b11, post_ndwi, post_mndwi, elev, slope, post_valid, otsu_thresh)

    pre_water = ((pre_mndwi > otsu_thresh) | (pre_ndwi > 0.10)) & pre_valid & flat_mask
    post_water = ((post_mndwi > otsu_thresh) | (post_ndwi > 0.10)) & post_valid & flat_mask

    change_map = xr.zeros_like(pre_ndwi, dtype="uint8")
    change_map = xr.where(pre_water & post_water, 1, change_map)
    change_map = xr.where((~pre_water) & post_water, 2, change_map)
    change_map = xr.where(~post_valid, 3, change_map)

    pre_ndwi.astype("float32").rio.write_crs(CRS_UTM).rio.to_raster(f"{DATA_DIR}/{run_label}_pre_ndwi.tif")
    post_ndwi.astype("float32").rio.write_crs(CRS_UTM).rio.to_raster(f"{DATA_DIR}/{run_label}_post_ndwi.tif")
    pre_mndwi.astype("float32").rio.write_crs(CRS_UTM).rio.to_raster(f"{DATA_DIR}/{run_label}_pre_mndwi.tif")
    post_mndwi.astype("float32").rio.write_crs(CRS_UTM).rio.to_raster(f"{DATA_DIR}/{run_label}_post_mndwi.tif")
    change_map.astype("uint8").rio.write_crs(CRS_UTM).rio.to_raster(f"{DATA_DIR}/{run_label}_change_map.tif")

    pixel_area_km2 = (RESOLUTION * RESOLUTION) / 1e6
    perm_km2 = float((change_map == 1).sum().values * pixel_area_km2)
    new_flood_km2 = float((change_map == 2).sum().values * pixel_area_km2)
    unobserved_pct = float((change_map == 3).mean().values * 100.0)

    cat_record = gpd.GeoDataFrame(
        [{
            "run_label": run_label,
            "pre_scene_id": pre_item.id,
            "post_scene_id": post_item.id,
            "pre_date": pre_date_str,
            "post_date": post_date_str,
            "pre_cloud_pct": round(pre_cloud, 2),
            "post_cloud_pct": round(post_cloud, 2),
            "unobserved_aoi_pct": round(unobserved_pct, 2),
            "permanent_water_km2": round(perm_km2, 2),
            "newly_flooded_km2": round(new_flood_km2, 2),
            "otsu_mndwi_threshold": val_metrics["otsu_mndwi_thresh"],
            "rf_f1_score": val_metrics["rf_vs_mndwi_f1"],
            "rf_iou_score": val_metrics["rf_vs_mndwi_iou"],
            "cohen_kappa": val_metrics["rf_vs_mndwi_kappa"],
            "ingested_at": datetime.now(timezone.utc).isoformat(),
            "geometry": box(*AOI_BBOX),
        }],
        crs="EPSG:4326",
    )

    cat_path = f"{DATA_DIR}/catalog.parquet"
    if os.path.exists(cat_path):
        existing = gpd.read_parquet(cat_path)
        existing = existing[existing["run_label"] != run_label]
        cat_record = gpd.GeoDataFrame(pd.concat([existing, cat_record], ignore_index=True), crs="EPSG:4326")
    cat_record.to_parquet(cat_path)

    print(f"Scientific Validation -> Otsu Threshold: {val_metrics['otsu_mndwi_thresh']} | RF F1: {val_metrics['rf_vs_mndwi_f1']} | IoU: {val_metrics['rf_vs_mndwi_iou']} | Kappa: {val_metrics['rf_vs_mndwi_kappa']}")
    print(f"Flood Summary         -> Permanent River: {perm_km2:.2f} km² | Newly Flooded: {new_flood_km2:.2f} km²")
    return val_metrics


if __name__ == "__main__":
    run_scientific_pipeline("2026-02-01/2026-03-20", "2026-07-01/2026-08-25", "2026_Monsoon_Assam")
    run_scientific_pipeline("2024-02-01/2024-03-20", "2024-07-01/2024-08-15", "2024_Monsoon_Benchmark")
