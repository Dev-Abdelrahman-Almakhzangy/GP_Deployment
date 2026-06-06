"""Project 1 — Wheat disease classification (HybridViT, 15 classes).

Predictions run through ONNX Runtime (your existing export).
Grad-CAM is optional and requires the PyTorch model (see grad_cam_overlay).
"""
from pathlib import Path
import numpy as np
import onnxruntime as ort

from .vision_common import preprocess, softmax

MODEL_PATH = Path(__file__).parent.parent / "models" / "hybridvit.onnx"

# ---- FILL IN: 15 class names in the SAME order as training ----
# If you used torchvision ImageFolder, the order is alphabetical by folder name.
CLASS_NAMES = [
    "Aphid", "Black Rust", "Blast", "Brown Rust", "Common Root Rot",
    "Fusarium Head Blight", "Healthy", "Leaf Blight", "Mildew", "Mite",
    "Septoria", "Smut", "Stem Fly", "Tan Spot", "Yellow Rust",
]

def load_session() -> ort.InferenceSession:
    return ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])


def predict(session: ort.InferenceSession, img, size: int = 256):
    """Return list of (class_name, probability) sorted high->low."""
    x = preprocess(img, size=size)
    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: x})[0]      # (1, 15)
    probs = softmax(logits)[0]
    ranked = sorted(zip(CLASS_NAMES, probs.tolist()), key=lambda t: t[1], reverse=True)
    return ranked


# ---------------------------------------------------------------------------
# OPTIONAL: Grad-CAM. Requires torch + your HybridViT class + a .pth checkpoint,
# because CAM needs intermediate activations that ONNX doesn't expose.
# Targets stage-3 per your finalized design. Uncomment and wire your model in.
# ---------------------------------------------------------------------------
def grad_cam_overlay(img, size: int = 224):
    """Stub. To enable:
      1. import your model class and load weights:
            from your_model_def import HybridViT
            model = HybridViT(num_classes=15).eval()
            model.load_state_dict(torch.load("models/hybridvit.pth", map_location="cpu"))
      2. pick the stage-3 conv target layer (finer spatial resolution).
      3. run pytorch_grad_cam.GradCAM and overlay with show_cam_on_image.
    Returns an RGB np.uint8 array or None if unavailable.
    """
    return None
