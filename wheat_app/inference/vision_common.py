"""Shared helpers for the two image models (HybridViT, EfficientNet-B0)."""
import numpy as np
from PIL import Image

# ImageNet stats — correct for both ConvNeXt-Tiny and EfficientNet-B0 pretrained backbones.
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(img: Image.Image, size: int = 224) -> np.ndarray:
    """PIL image -> (1, 3, size, size) float32 batch, ImageNet-normalized.

    NOTE: confirm `size` and any center-crop matches your training transform.
    If you trained with a resize-then-centercrop, replicate that here.
    """
    img = img.convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0       # H,W,C in [0,1]
    arr = (arr - _MEAN) / _STD                             # normalize
    arr = arr.transpose(2, 0, 1)[None]                     # -> 1,C,H,W
    return np.ascontiguousarray(arr)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)
