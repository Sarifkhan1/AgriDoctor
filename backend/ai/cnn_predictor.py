"""
AgriDoctor AI — Local CNN predictor for serving (OPTIONAL, off by default).

Loads the checkpoint produced by `scripts/train_real.py` (a MobileNetV3-small
classifier over crop-disease classes mapped to AgriDoctor's taxonomy) and runs
inference. Lazy-loads torch + the model on first use so the API starts instantly
without PyTorch when the CNN is disabled.

IMPORTANT: this classifier has NO out-of-scope reject class, so it must not be the
default engine — see `Settings.use_local_cnn` (default False). The Groq
vision–language model + server-side safety layer remains the default so non-leaf /
unsupported-subject images are correctly rejected. When enabled, the CNN is a fast
local path for images already known to be in-scope crop leaves; low-confidence
results fall back to the hosted model.
"""

import io
import logging
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image

logger = logging.getLogger(__name__)

_torch = None
_F = None
_T = None
_models = None


def _ensure_torch():
    global _torch, _F, _T, _models
    if _torch is None:
        try:
            import torch
            import torch.nn.functional as functional
            import torchvision.models as tv_models
            import torchvision.transforms as transforms

            _torch = torch
            _F = functional
            _T = transforms
            _models = tv_models
        except ImportError as exc:
            raise ImportError(
                "PyTorch is required for local CNN inference. "
                "Install with: pip install -r requirements-ml.txt"
            ) from exc


def _build_model(arch: str, num_classes: int):
    """Recreate the architecture used by scripts/train_real.py."""
    if arch == "mobilenet_v3_small":
        model = _models.mobilenet_v3_small(weights=None)
        in_f = model.classifier[-1].in_features
        model.classifier[-1] = _torch.nn.Linear(in_f, num_classes)
        return model
    raise ValueError(f"Unsupported CNN arch in checkpoint: {arch!r}")


class CNNPredictor:
    """Run inference with the trained local classifier (thread-safe, eval mode)."""

    def __init__(self, model_path: str, device: str = "auto") -> None:
        self._model_path = Path(model_path)
        self._device_str = device
        self._model = None
        self._device = None
        self._transform = None
        self._idx_to_label: Dict[int, str] = {}
        self._loaded = False
        self._load_failed = False

    @property
    def available(self) -> bool:
        return self._model_path.exists()

    def _load(self) -> bool:
        if self._loaded:
            return True
        if self._load_failed:
            return False
        if not self._model_path.exists():
            logger.info("CNN model not found at %s — using Groq", self._model_path)
            self._load_failed = True
            return False
        try:
            _ensure_torch()
            if self._device_str == "auto":
                if _torch.cuda.is_available():
                    self._device = "cuda"
                elif getattr(_torch.backends, "mps", None) and _torch.backends.mps.is_available():
                    self._device = "mps"
                else:
                    self._device = "cpu"
            else:
                self._device = self._device_str

            ckpt = _torch.load(
                self._model_path, map_location=self._device, weights_only=False
            )
            arch = ckpt.get("arch", "mobilenet_v3_small")
            num_classes = ckpt["num_classes"]
            self._model = _build_model(arch, num_classes)
            self._model.load_state_dict(ckpt["model_state_dict"])
            self._model.to(self._device)
            self._model.eval()

            # idx_to_label keys may be ints or strings after (de)serialisation.
            raw_map = ckpt.get("idx_to_label", {})
            self._idx_to_label = {int(k): v for k, v in raw_map.items()}

            size = int(ckpt.get("input_size", 224))
            mean = ckpt.get("normalize_mean", [0.485, 0.456, 0.406])
            std = ckpt.get("normalize_std", [0.229, 0.224, 0.225])
            self._transform = _T.Compose(
                [_T.Resize((size, size)), _T.ToTensor(), _T.Normalize(mean, std)]
            )

            self._loaded = True
            logger.info(
                "CNN loaded: %s (arch=%s, classes=%d, device=%s, best_f1=%.4f)",
                self._model_path.name,
                arch,
                num_classes,
                self._device,
                float(ckpt.get("best_f1", 0.0)),
            )
            return True
        except Exception:
            logger.exception("Failed to load CNN model from %s", self._model_path)
            self._load_failed = True
            return False

    def predict(self, image_bytes: bytes) -> Optional[Dict]:
        if not self._load():
            return None
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            tensor = self._transform(image).unsqueeze(0).to(self._device)
            with _torch.no_grad():
                logits = self._model(tensor)[0]
            probs = _F.softmax(logits, dim=0)
            k = min(5, probs.numel())
            top_probs, top_idx = probs.topk(k)
            predictions: List[Dict] = []
            for p, i in zip(top_probs.cpu().numpy(), top_idx.cpu().numpy()):
                label = self._idx_to_label.get(int(i))
                if label:
                    predictions.append({"label": label, "confidence": round(float(p), 4)})
            if not predictions:
                return None
            return {
                "primary_label": predictions[0]["label"],
                "confidence": predictions[0]["confidence"],
                "top_predictions": predictions,
                "source": "local_cnn",
            }
        except Exception:
            logger.exception("CNN prediction failed")
            return None


_default_predictor: Optional[CNNPredictor] = None


def get_predictor(model_path: str = "") -> CNNPredictor:
    global _default_predictor
    if _default_predictor is None:
        if not model_path:
            root = Path(__file__).resolve().parent.parent.parent
            model_path = str(root / "data" / "models" / "best_model.pt")
        _default_predictor = CNNPredictor(model_path)
    return _default_predictor
