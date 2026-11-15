# EO-AI Labs (STAr, IIT Kanpur) - Technical Assignment
**Role 01: Data & Model Infrastructure Engineer (EO-AI)**
**Configuration:** R5-S1-D2-T1 (Brahmaputra Floodplain, Assam | Sentinel-2 L2A + Copernicus DEM | Surface Water | Two-Date Change Detection)

## Live Deployment Links

* **Interactive Decision Dashboard (Lovable React + Leaflet):** https://brahmaputra-flood-eoai.lovable.app/
* **Geospatial Scientific API (Render FastAPI):** https://eoai-brahmaputra-flood.onrender.com/docs

## Quickstart (Run Locally)

1. `git clone https://github.com/shradhanjalisahoo-ai/eoai-brahmaputra-flood.git && cd eoai-brahmaputra-flood`
2. `pip install -r requirements.txt`
3. `uvicorn api:app --host 0.0.0.0 --port 8000`

## Operational Use-Case
A state disaster-response desk (Assam State Disaster Management Authority) requires a rapid surface-water extent snapshot across the Brahmaputra floodplain (Morigaon district, Tile T46RDQ) immediately following heavy monsoon rains to isolate newly inundated land from permanent river channels and prioritize which revenue circles to reach first.

## Repository Structure
- `api.py`: FastAPI geospatial REST service (`/api/runs`, `/api/triage`, `/api/overlay`, `/api/ingest`)
- `src/science_engine.py`: STAC ingestion, SCL cloud masking, DEM slope filtering, MNDWI/NDWI, Otsu & Random Forest validation
- `data/`: Pre-cached Cloud-Optimized GeoTIFFs & GeoParquet catalog (`catalog.parquet`)
- `requirements.txt`: Pinned Python dependencies

## Quickstart (Run Locally)
1. `git clone [https://github.com/shradhanjalisahoo-ai/eoai-brahmaputra-flood.git](https://github.com/shradhanjalisahoo-ai/eoai-brahmaputra-flood.git) && cd eoai-brahmaputra-flood`
2. `pip install -r requirements.txt`
3. `python src/science_engine.py`
4. `uvicorn api:app --host 0.0.0.0 --port 8000`
