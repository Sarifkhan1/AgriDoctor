"""
AgriDoctor AI — Local CNN predictor for serving.

Wraps the trained CropDiseaseViT model for inference. Lazy-loads the model
on first call to avoid slowing down app startup if no model file is present.
Falls back gracefully to None when no model is available, allowing the Groq
vision pipeline to handle the request instead.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image
import io

logger = logging.getLogger(__name__)

# Lazy imports — torch is optional for the serving API
_torch = None
_F = None
_T = None


def _ensure_torch():
    """Import torch lazily so the backend can start without it."""
    global _torch, _F, _T
    if _torch is None:
        try:
            import torch
            import torch.nn.functional as functional
            import torchvision.transforms as transforms

            _torch = torch
            _F = functional
            _T = transforms
        except ImportError:
            raise ImportError(
                "PyTorch is required for local CNN inference. "
                "Install with: pip install torch torchvision"
            )


# Import the model class and labels — these are pure Python, no torch needed
# at import time.
_CropDiseaseViT = None
_IDX_TO_LABEL = None
_NUM_CLASSES = None


def _ensure_model_class():
    """Import the model definition lazily."""
    global _CropDiseaseViT, _IDX_TO_LABEL, _NUM_CLASSES
    if _CropDiseaseViT is None:
        import sys
        from pathlib import Path as _Path

        # Add project root to path so we can import src.models
        project_root = _Path(__file__).resolve().parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from src.models.train_image_model import (
            CropDiseaseViT,
            IDX_TO_LABEL,
            NUM_CLASSES,
        )

        _CropDiseaseViT = CropDiseaseViT
        _IDX_TO_LABEL = IDX_TO_LABEL
        _NUM_CLASSES = NUM_CLASSES


class CNNPredictor:
    """
    Run inference with a trained CropDiseaseViT model.

    Thread-safe: the model is loaded once and used in eval mode (no gradient
    tracking). Multiple concurrent requests can call predict() safely.
    """

    def __init__(self, model_path: str, device: str = "auto") -> None:
        self._model_path = Path(model_path)
        self._device_str = device
        self._model = None  # loaded lazily
        self._device = None
        self._transform = None
        self._loaded = False
        self._load_failed = False

    @property
    def available(self) -> bool:
        """Whether a trained model file exists on disk."""
        return self._model_path.exists()

    def _load(self) -> bool:
        """Load the model from disk. Returns True on success."""
        if self._loaded:
            return True
        if self._load_failed:
            return False

        if not self._model_path.exists():
            logger.info("CNN model not found at %s — will use Groq fallback", self._model_path)
            self._load_failed = True
            return False

        try:
            _ensure_torch()
            _ensure_model_class()

            # Resolve device
            if self._device_str == "auto":
                if _torch.cuda.is_available():
                    self._device = "cuda"
                elif hasattr(_torch.backends, "mps") and _torch.backends.mps.is_available():
                    self._device = "mps"
                else:
                    self._device = "cpu"
            else:
                self._device = self._device_str

            # Load checkpoint
            checkpoint = _torch.load(
                self._model_path, map_location=self._device, weights_only=False
            )

            self._model = _CropDiseaseViT(
                num_classes=checkpoint.get("num_classes", _NUM_CLASSES),
                backbone=checkpoint.get("backbone", "vit_b_16"),
                pretrained=False,
            )
            self._model.load_state_dict(checkpoint["model_state_dict"])
            self._model.to(self._device)
            self._model.eval()

            self._transform = _T.Compose(
                [
                    _T.Resize((224, 224)),
                    _T.ToTensor(),
                    _T.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

            self._loaded = True
            logger.info(
                "CNN model loaded: %s (backbone=%s, device=%s, F1=%.4f)",
                self._model_path.name,
                checkpoint.get("backbone", "?"),
                self._device,
                checkpoint.get("best_f1", 0),
            )
            return True

        except Exception:
            logger.exception("Failed to load CNN model from %s", self._model_path)
            self._load_failed = True
            return False

    def predict(self, image_bytes: bytes) -> Optional[Dict]:
        """
        Classify a crop disease image.

        Args:
            image_bytes: Raw bytes of the image (JPEG/PNG).

        Returns:
            A dict with prediction results, or None if the model is
            not available / failed to load.

            Example return:
            {
                "primary_label": "TOM_EARLY_BLIGHT",
                "confidence": 0.92,
                "severity": 0.65,
                "top_predictions": [
                    {"label": "TOM_EARLY_BLIGHT", "confidence": 0.92},
                    {"label": "TOM_LATE_BLIGHT", "confidence": 0.05},
                    ...
                ],
                "source": "local_cnn"
            }
        """
        if not self._load():
            return None

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            tensor = self._transform(image).unsqueeze(0).to(self._device)

            with _torch.no_grad():
                outputs = self._model(tensor, return_features=False)

            logits = outputs["logits"][0]
            probs = _F.softmax(logits, dim=0)

            top_k = min(5, len(probs))
            top_probs, top_indices = probs.topk(top_k)

            predictions: List[Dict] = []
            for prob, idx in zip(
                top_probs.cpu().numpy(), top_indices.cpu().numpy()
            ):
                predictions.append(
                    {
                        "label": _IDX_TO_LABEL[int(idx)],
                        "confidence": round(float(prob), 4),
                    }
                )

            severity_val = float(outputs["severity"][0].cpu())

            return {
                "primary_label": predictions[0]["label"],
                "confidence": predictions[0]["confidence"],
                "severity": round(severity_val, 4),
                "top_predictions": predictions,
                "source": "local_cnn",
            }

        except Exception:
            logger.exception("CNN prediction failed")
            return None


# --------------------------------------------------------------------------- #
# Module-level singleton (created lazily by the analyzer)
# --------------------------------------------------------------------------- #
_default_predictor: Optional[CNNPredictor] = None


def get_predictor(model_path: str = "") -> CNNPredictor:
    """Return (or create) the singleton CNN predictor."""
    global _default_predictor
    if _default_predictor is None:
        if not model_path:
            # Default path relative to project root
            project_root = Path(__file__).resolve().parent.parent.parent
            model_path = str(project_root / "data" / "models" / "best_model.pt")
        _default_predictor = CNNPredictor(model_path)
    return _default_predictor
