"""
EO-AI Labs (STAr, IIT Kanpur) - Role 01 Technical Assignment
Configuration: R5 (Brahmaputra, Assam) | S1 (Sentinel-2 L2A + DEM Filter) | D2 (Surface Water) | T1 (Change Detection)
Layer 4: Theme-Adaptive Interactive Geospatial Command Center (Folium Fullscreen + Plotly + GeoParquet)
"""

import os
import numpy as np
import geopandas as gpd
import rioxarray
from rasterio.warp import Resampling
import streamlit as st
import folium
from folium.plugins import Fullscreen
from streamlit_folium import st_folium
import plotly.express as px

from src.ingest import run_pipeline, RESOLUTION

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="EO-AI Labs | Brahmaputra Flood Command Center",
    page_icon="🛰️",
    layout="wide",
)

# --- THEME-ADAPTIVE STYLING (WORKS IN BOTH LIGHT & DARK MODE) ---
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
    div[data-testid="stMetric"] {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.25);
        padding: 12px 16px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER & DECISION FRAMING ---
col_title, col_badge = st.columns([4, 1])
with col_title:
    st.title("🛰️ Brahmaputra Floodplain: Rapid Inundation Intelligence")
    st.caption("**EO-AI Labs (STAr, IIT Kanpur)** | Configuration: `R5` (Assam Floodplain) · `S1` (Sentinel-2 + DEM) · `D2` (Surface Water) · `T1` (Change Detection)")
with col_badge:
    st.info("**Status:** Operational\n**Grid:** 20m UTM 46N")

with st.expander("📋 Operational Use-Case & Decision Protocol (Click to expand)", expanded=False):
    st.markdown("""
    **Decision Context:** Immediately following heavy monsoon rainfall in Assam, the State Disaster Management Authority (ASDMA) desk must determine **which revenue circles/blocks experienced the sharpest surge in newly inundated land** to prioritize rescue boat dispatch and relief allocation before ground reports arrive.
    """)

# --- SIDEBAR: CATALOGUE, THRESHOLD & LIVE STAC INGESTION ---
st.sidebar.header("🎛️ Pipeline & Sensor Controls")

cat_path = "data/catalog.parquet"
block_path = "data/block_summary.parquet"

if not os.path.exists(cat_path):
    st.warning("Initializing STAC catalogue...")
    run_pipeline()

catalog_df = gpd.read_parquet(cat_path)
available_runs = catalog_df["run_label"].tolist()
selected_run = st.sidebar.selectbox("1. Select Flood Event (GeoParquet)", available_runs, index=0)

ndwi_thresh = st.sidebar.slider(
    "2. NDWI Water Sensitivity Threshold",
    min_value=0.00,
    max_value=0.35,
    value=0.10,
    step=0.05,
    help="Lower (0.05–0.10) captures turbid, silt-laden Brahmaputra floodwater; higher (0.20+) isolates deep clear water."
)

layer_opacity = st.sidebar.slider("3. Flood Overlay Opacity", 0.2, 1.0, 0.75, 0.05)

st.sidebar.divider()
st.sidebar.subheader("🌐 Live STAC On-Demand Fetch")
with st.sidebar.expander("Ingest New Date Window"):
    new_label = st.text_input("Event Label", value="2025_Monsoon_Assam")
    new_pre = st.text_input("Pre-Monsoon Window", value="2025-02-01/2025-03-20")
    new_post = st.text_input("Monsoon Window", value="2025-07-01/2025-08-25")
    if st.button("🚀 Run STAC Ingestion", width="stretch"):
        with st.spinner("Querying Planetary Computer STAC & computing NDWI cube..."):
            run_pipeline(pre_window=new_pre, post_window=new_post, ndwi_thresh=ndwi_thresh, run_label=new_label)
        st.success(f"Added {new_label}!")
        st.rerun()

# --- LOAD RASTERS & REPROJECT TO WGS84 (EPSG:4326) FOR INTERACTIVE WEB MAP ---
@st.cache_data
def prepare_web_layers(run_name: str, thresh: float):
    pre_da = rioxarray.open_rasterio(f"data/{run_name}_pre_ndwi.tif").squeeze()
    post_da = rioxarray.open_rasterio(f"data/{run_name}_post_ndwi.tif").squeeze()
    saved_da = rioxarray.open_rasterio(f"data/{run_name}_change_map.tif").squeeze()

    pre_valid = ~np.isnan(pre_da.values)
    post_valid = (~np.isnan(post_da.values)) & (saved_da.values != 3)

    pre_water = (pre_da.values > thresh) & pre_valid
    post_water = (post_da.values > thresh) & post_valid

    # 0 = Dry, 1 = Permanent River, 2 = Newly Flooded, 3 = Cloud
    change = np.zeros_like(saved_da.values, dtype=np.uint8)
    change[pre_water & post_water] = 1
    change[(~pre_water) & post_water] = 2
    change[~post_valid] = 3

    change_xr = saved_da.copy(data=change)
    change_4326 = change_xr.rio.reproject("EPSG:4326", resampling=Resampling.nearest)
    bounds = change_4326.rio.bounds()
    folium_bounds = [[bounds[1], bounds[0]], [bounds[3], bounds[2]]]

    arr = change_4326.values
    rgba = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
    rgba[arr == 1] = [31, 120, 180, 210]   # Permanent River: Deep Blue
    rgba[arr == 2] = [230, 35, 25, 235]    # Newly Flooded: Alert Red
    rgba[arr == 3] = [140, 140, 140, 130]  # Cloud Masked: Translucent Gray

    return change, rgba, folium_bounds

run_meta = catalog_df[catalog_df["run_label"] == selected_run].iloc[0]
dynamic_change, rgba_overlay, folium_bounds = prepare_web_layers(selected_run, ndwi_thresh)

# --- CALCULATE REAL METRICS ---
pixel_area_km2 = (RESOLUTION * RESOLUTION) / 1e6
perm_water_km2 = float(np.sum(dynamic_change == 1) * pixel_area_km2)
new_flood_km2 = float(np.sum(dynamic_change == 2) * pixel_area_km2)
cloud_pct = float(np.mean(dynamic_change == 3) * 100.0)
surge_pct = (new_flood_km2 / max(perm_water_km2, 0.1)) * 100.0

# --- KPI SUMMARY CARDS ---
k1, k2, k3, k4 = st.columns(4)
k1.metric("Target Event Date", f"{run_meta['post_date']}", f"Baseline: {run_meta['pre_date']}")
k2.metric("Permanent River Channel", f"{perm_water_km2:.2f} km²", "Pre-Monsoon Baseline")
k3.metric("Newly Flooded Land", f"{new_flood_km2:.2f} km²", f"+{surge_pct:.0f}% vs Baseline", delta_color="inverse")
k4.metric("Cloud Occlusion (NoData)", f"{cloud_pct:.1f}%", "Masked via Sentinel-2 SCL")

# --- COMPUTE SUB-BLOCK STATS ---
blocks_gdf = gpd.read_parquet(block_path)
blocks_gdf = blocks_gdf[blocks_gdf["run_label"] == selected_run].copy()

ny, nx = dynamic_change.shape
half_y, half_x = ny // 2, nx // 2
quad_map = {
    "Mayong Circle": dynamic_change[half_y:ny, 0:half_x],
    "Bhuragaon Circle": dynamic_change[half_y:ny, half_x:nx],
    "Laharighat Circle": dynamic_change[0:half_y, 0:half_x],
    "Morigaon Sadar": dynamic_change[0:half_y, half_x:nx],
}

updated_rows = []
for _, row in blocks_gdf.iterrows():
    b_name = row["block_name"]
    sub_arr = quad_map[b_name]
    b_perm = float(np.sum(sub_arr == 1) * pixel_area_km2)
    b_new = float(np.sum(sub_arr == 2) * pixel_area_km2)
    b_cloud = float(np.mean(sub_arr == 3) * 100.0)
    b_surge = (b_new / max(b_perm, 0.1)) * 100.0
    updated_rows.append({
        "block_name": b_name,
        "Baseline Water (km²)": round(b_perm, 2),
        "Newly Flooded (km²)": round(b_new, 2),
        "Surge (%)": round(b_surge, 1),
        "Cloud Occluded (%)": round(b_cloud, 1),
        "geometry": row["geometry"],
    })

triage_gdf = gpd.GeoDataFrame(updated_rows, crs="EPSG:4326").sort_values(by="Newly Flooded (km²)", ascending=False).reset_index(drop=True)
tiers = ["1 - Critical (Immediate Boats)", "2 - High (Relief Dispatch)", "3 - Moderate (Alert)", "4 - Monitor"]
triage_gdf["Priority Tier"] = tiers[:len(triage_gdf)]

# --- MAIN INTERACTIVE WORKSPACE: LEFT = FULLSCREEN MAP | RIGHT = ANALYTICS ---
st.subheader("🗺️ Interactive Flood Inundation Map & Sub-Block Priority Analytics")
map_col, chart_col = st.columns([1.35, 1])

with map_col:
    center_lat = (folium_bounds[0][0] + folium_bounds[1][0]) / 2
    center_lon = (folium_bounds[0][1] + folium_bounds[1][1]) / 2

    # Default: Esri World Imagery (High-Resolution Satellite Basemap — no API key required)
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Satellite Imagery (Default)",
    )

    # Fullscreen button on the top-left of the map
    Fullscreen(
        position="topleft",
        title="Expand Map to Full Screen",
        title_cancel="Exit Full Screen",
        force_separate_button=True,
    ).add_to(m)

    # Second option: Esri World Street Map (Roads, Towns & Rivers — no API key required)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Street Map",
        name="Street & Towns Map",
        overlay=False,
        control=True,
    ).add_to(m)

    # Third option: Esri Dark Gray Canvas (Clean Contrast Basemap — no API key required)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Esri Dark Gray Canvas",
        name="Dark Contrast Map",
        overlay=False,
        control=True,
    ).add_to(m)

    # Flood Change Raster Overlay
    folium.raster_layers.ImageOverlay(
        image=rgba_overlay,
        bounds=folium_bounds,
        opacity=layer_opacity,
        name="Surface-Water Change (Red=New Flood, Blue=River)",
        interactive=True,
        cross_origin=False,
    ).add_to(m)

    # Interactive Hoverable Revenue Circle Boundaries (High-contrast orange/gold border)
    folium.GeoJson(
        triage_gdf,
        name="Revenue Circle Boundaries",
        style_function=lambda feature: {
            "fillColor": "transparent",
            "color": "#d97706",
            "weight": 2.5,
            "dashArray": "6, 4",
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["block_name", "Priority Tier", "Newly Flooded (km²)", "Baseline Water (km²)", "Cloud Occluded (%)"],
            aliases=["Block:", "Priority:", "New Flood:", "Baseline River:", "Cloud Cover:"],
            localize=True,
            sticky=True,
        ),
    ).add_to(m)

    # Collapsed dropdown icon in top-right so it never blocks the map view
    folium.LayerControl(collapsed=True, position="topright").add_to(m)

    st_folium(m, height=480, width="stretch")
    st.caption("Tip: Click the **[ ⛶ ]** icon on the top-left of the map for Full Screen, or the **Layers icon** on the top-right to switch between Satellite, Street, and Dark basemaps.")

with chart_col:
    plot_df = triage_gdf.drop(columns=["geometry"]).melt(
        id_vars=["block_name", "Priority Tier"],
        value_vars=["Baseline Water (km²)", "Newly Flooded (km²)"],
        var_name="Water Type",
        value_name="Area (km²)"
    )
    fig_bar = px.bar(
        plot_df,
        x="Area (km²)",
        y="block_name",
        color="Water Type",
        orientation="h",
        barmode="group",
        title="Inundation Comparison by Revenue Circle",
        color_discrete_map={
            "Baseline Water (km²)": "#1f78b4",
            "Newly Flooded (km²)": "#e62319",
        },
        height=275,
    )
    fig_bar.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(autorange="reversed", title=""),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_bar, width="stretch")

    st.markdown("**Actionable Triage Matrix**")
    st.dataframe(
        triage_gdf[["Priority Tier", "block_name", "Newly Flooded (km²)", "Cloud Occluded (%)"]],
        width="stretch",
        hide_index=True,
    )

# --- BOTTOM EXPANDER: STAC PROVENANCE & CATALOGUE AUDIT ---
with st.expander("🔍 Inspect STAC Provenance & GeoParquet Feature Store"):
    st.dataframe(catalog_df.drop(columns=["geometry"]), width="stretch", hide_index=True)