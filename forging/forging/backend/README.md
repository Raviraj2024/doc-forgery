# Document Forgery Backend

FastAPI backend for document forgery analysis with:

- PDF and image ingestion
- multi-page rendering
- segmentation-based tamper localization from `working_forgery_best.pth`
- ELA, SRM, Noiseprint, DINO-style anomaly maps
- OCR anomaly checks
- duplicate detection with MD5 and pHash
- SQLite persistence and stable artifact URLs for frontend use

## Important checkpoint note

The markdown spec says to try a 12-channel `efficientnet-b4` UNet first. The supplied checkpoint does not match that contract.

- The checkpoint's first stem convolution expects `13` input channels.
- The bundled Kaggle training script builds `smp.Unet(encoder_name="efficientnet-b3", in_channels=13, classes=1)`.
- The loader still tries the requested `b4:12` and `b3:12` variants first, then falls back to checkpoint-derived candidates (`b3:13`, `b4:13`, `b2:13`) so the shipped `.pth` can actually load.

The segmentation tensor implemented here follows the real checkpoint contract:

- channels `0-2`: RGB
- channels `3-5`: ELA
- channels `6-8`: Laplacian residuals
- channel `9`: OCR proxy map
- channel `10`: SRM map
- channel `11`: Noiseprint approximation
- channel `12`: DINO-style anomaly map

## Project layout

```text
backend/
  app/
  data/
  tests/
  requirements.txt
  .env.example
  Dockerfile
  README.md
```

## Setup

1. Create a Python 3.11 environment.
2. Install dependencies:

```bash
pip install -r backend/requirements.txt
```

3. Copy the env file if you want overrides:

```bash
copy backend\\.env.example backend\\.env
```

4. Make sure the checkpoint exists at `./working_forgery_best.pth`, or set `CHECKPOINT_PATH`.

## Run locally

```bash
set PYTHONPATH=backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open:

- Swagger UI: `http://localhost:8000/docs`
- Health: `http://localhost:8000/api/v1/health`
- Model info: `http://localhost:8000/api/v1/model/info`

## API summary

- `GET /api/v1/health`
- `GET /api/v1/model/info`
- `POST /api/v1/analyze`
- `GET /api/v1/analyze/{analysis_id}`
- `GET /api/v1/analyze?page=1&page_size=10`
- `GET /api/v1/artifacts/{analysis_id}/{filename}`
- `DELETE /api/v1/analyze/{analysis_id}`

## Analyze request

```bash
curl -X POST "http://localhost:8000/api/v1/analyze" \
  -F "file=@sample.pdf" \
  -F "document_type=invoice" \
  -F "submitter_id=frontend-demo"
```

## Response contract

The main response includes:

- `analysis_id`
- `verdict`
- `forensic_risk_score`
- `engine_scores`
- `ocr_anomalies`
- `duplicate_check`
- `pages[]` with per-page artifact URLs and tampered regions
- `warnings`

Artifacts are written to disk under `backend/data/artifacts/<analysis_id>/` and served back through stable URLs like:

```text
/api/v1/artifacts/<analysis_id>/page_1_overlay.png
```

## Frontend integration notes

Use the response directly for a page-based viewer:

- `pages[].artifacts.original_url` for the raw page
- `pages[].artifacts.overlay_url` for mask overlay
- `pages[].artifacts.ela_heatmap_url`
- `pages[].artifacts.srm_heatmap_url`
- `pages[].artifacts.noiseprint_heatmap_url`
- `pages[].artifacts.dino_heatmap_url`
- `pages[].artifacts.combined_heatmap_url`
- `pages[].artifacts.contours_url`
- `pages[].tampered_regions` for region tables or callouts

Document-level panels can render:

- `verdict`
- `forensic_risk_score`
- `engine_scores`
- `ocr_anomalies`
- `duplicate_check`
- `warnings`

## OCR and DINO fallbacks

- OCR: tries `PaddleOCR` if installed at runtime, then `EasyOCR`, otherwise returns `ocr_anomaly_score = 0.0` and records an OCR warning.
- DINO: tries a timm ViT backend when available, otherwise falls back to a deterministic patch-statistics anomaly map.
- Segmentation: if the checkpoint cannot be loaded, the API still returns the full contract with blank masks and a warning instead of crashing.

## Storage

SQLite tables:

- `analyses`
- `pages`
- `regions`
- `duplicate_fingerprints`
- `ocr_anomalies`

Filesystem outputs:

- uploads: `backend/data/uploads/`
- analysis JSON: `backend/data/outputs/<analysis_id>/analysis.json`
- artifacts: `backend/data/artifacts/<analysis_id>/`

## Tests

```bash
set PYTHONPATH=backend
pytest backend/tests
```

Some tests are skipped automatically if optional runtime packages such as `segmentation-models-pytorch` or `PyMuPDF` are not installed in the current environment.

## Calibration and parity tooling

Generate a calibration profile from a labeled benchmark dataset:

```bash
set PYTHONPATH=backend
python backend/scripts/benchmark_and_calibrate.py path\to\dataset --dataset-name my-benchmark
```

Run an exact training/inference parity review over one image or a directory:

```bash
set PYTHONPATH=backend
python backend/scripts/review_training_parity.py path\to\dataset\images --sample-limit 10
```

The benchmark run writes:

- calibration profile: `backend/data/calibration/latest.json`
- parity report: `backend/data/parity/latest.json`

## Docker

Build from the repository root because the Dockerfile needs both `backend/` and the checkpoint:

```bash
docker build -f backend/Dockerfile -t doc-forging-backend .
docker run -p 8000:8000 doc-forging-backend
```
