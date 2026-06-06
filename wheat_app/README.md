# Wheat Health & Yield — Streamlit Demo

Serves all four thesis models behind one local web app.

## Run

```bash
cd wheat_app
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Opens at http://localhost:8501

## Drop your model files into `models/`

| File                        | Model                          |
|-----------------------------|--------------------------------|
| `hybridvit.onnx`            | Project 1 — disease (15 cls)   |
| `efficientnet_b0.onnx`      | Project 2 — rust severity      |
| `lightgbm_yield.txt`        | Project 3 — yield (booster)    |
| `egypt_linreg.pkl`          | Project 4 — Egypt forecast     |

## Wiring checklist (the only things you must fill in)

1. **`inference/disease.py`** → `CLASS_NAMES` (15, in training order; ImageFolder = alphabetical).
2. **`inference/rust.py`** → `SEVERITY_GRADES` (or switch to regression in `predict`).
3. **`inference/vision_common.py`** → confirm `size`/crop matches your training transform.
4. **`inference/egypt.py`** → `_make_features` to match your recursion (trend / lag / both) and `HISTORY`.
5. **(Optional)** `disease.grad_cam_overlay` → import your `HybridViT` class + `.pth`, target stage-3.

LightGBM needs nothing — the input form is generated from `booster.feature_name()`.

## Going beyond localhost

- **Free hosting:** push to GitHub → deploy on Streamlit Community Cloud (point it at `app.py`).
- **Containerized:** `streamlit run app.py --server.port 8501 --server.address 0.0.0.0` inside a slim Python image.
- **Production-style:** if the thesis wants a real API, the `inference/` modules are already framework-agnostic — they drop straight into FastAPI endpoints later.
