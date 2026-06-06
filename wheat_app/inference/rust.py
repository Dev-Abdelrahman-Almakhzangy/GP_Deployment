"""Project 2 — Yellow rust severity estimation (EfficientNet-B0).

Runs through ONNX Runtime. Treated here as a classifier over severity grades;
if you trained a regression head instead, see the note in `predict`.
"""
from pathlib import Path
import numpy as np
import onnxruntime as ort

from .vision_common import preprocess, softmax

MODEL_PATH = Path(__file__).parent.parent / "models" / "efficientnet_b0.onnx"

# ---- FILL IN: severity grades in training order ----
SEVERITY_GRADES = ["0", "MR", "MRMS", "MS", "R", "S"]  # example — replace with yours


def load_session() -> ort.InferenceSession:
    return ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])


def predict(session: ort.InferenceSession, img, size: int = 224):
    """Return list of (grade, probability) sorted high->low.

    If your model outputs a single continuous severity value (regression),
    replace the softmax block with: value = float(session.run(...)[0].squeeze())
    and return that instead.
    """
    x = preprocess(img, size=size)
    input_name = session.get_inputs()[0].name
    out = session.run(None, {input_name: x})[0]          # (1, n_grades)
    probs = softmax(out)[0]
    ranked = sorted(zip(SEVERITY_GRADES, probs.tolist()), key=lambda t: t[1], reverse=True)
    return ranked
