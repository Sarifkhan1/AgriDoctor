"""
AgriDoctor AI — local CNN inference (the primary diagnosis engine).

Loads the checkpoint from `scripts/train_cnn.py` and turns an image into a
*routing decision*, not just a label. The model is trained with two explicit
out-of-scope classes, so unlike a plain N-way classifier it can decline:

    DIAGNOSIS    confident, in-scope crop disease  -> answer locally, no API call
    NOT_PLANT    confidently not vegetation        -> reject locally, no API call
    ESCALATE     other plant, or low confidence    -> hand to the hosted VLM

That third branch is not a failure path. AgriDoctor's taxonomy covers rice,
chili, cucumber, eggplant and four livestock species that the public PlantVillage
corpus has no images for, so the CNN structurally cannot cover them — they are
*meant* to escalate. See backend/ai/analyzer.py for the routing.

Confidence is read after temperature scaling fitted on the validation split
(see fit_temperature in the training script). Thresholding a raw fine-tuned
softmax would mean acting on systematically overconfident numbers.

Torch is imported lazily so the API starts without it when the CNN is disabled.
"""

import io
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image

logger = logging.getLogger(__name__)

# Class names produced by scripts/prepare_dataset.py for the reject buckets.
OOS_OTHER_PLANT = "OOS_OTHER_PLANT"
OOS_NOT_PLANT = "OOS_NOT_PLANT"
REJECT_CLASSES = {OOS_OTHER_PLANT, OOS_NOT_PLANT}

# Decisions returned by CNNPredictor.predict()
DECISION_DIAGNOSIS = "diagnosis"
DECISION_NOT_PLANT = "not_plant"
DECISION_ESCALATE = "escalate"

_torch = None
_F = None
_T = None
_tv_models = None


def _ensure_torch():
    global _torch, _F, _T, _tv_models
    if _torch is None:
        try:
            import torch
            import torch.nn.functional as functional
            import torchvision.models as tv_models
            import torchvision.transforms as transforms

            _torch, _F, _T, _tv_models = torch, functional, transforms, tv_models
        except ImportError as exc:
            raise ImportError(
                "PyTorch is required for local CNN inference. "
                "Install with: pip install -r requirements-ml.txt"
            ) from exc


def _build_model(arch: str, num_classes: int, dropout: float = 0.3):
    """Recreate the architecture used at training time (weights loaded after)."""
    nn = _torch.nn
    if arch == "efficientnet_b0":
        m = _tv_models.efficientnet_b0(weights=None)
        in_f = m.classifier[-1].in_features
        m.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))
        return m
    if arch == "resnet50":
        m = _tv_models.resnet50(weights=None)
        m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, num_classes))
        return m
    if arch == "mobilenet_v3_large":
        m = _tv_models.mobilenet_v3_large(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m
    if arch == "mobilenet_v3_small":
        # Legacy checkpoint from the original scripts/train_real.py.
        m = _tv_models.mobilenet_v3_small(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m
    raise ValueError(f"Unsupported CNN arch in checkpoint: {arch!r}")


class CNNPredictor:
    """Thread-safe lazy-loading wrapper around the trained classifier."""

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        confidence_threshold: float = 0.75,
        margin_threshold: float = 0.10,
        reject_threshold: float = 0.60,
        use_tta: bool = True,
    ) -> None:
        self._model_path = Path(model_path)
        self._device_str = device
        self.confidence_threshold = confidence_threshold
        self.margin_threshold = margin_threshold
        self.reject_threshold = reject_threshold
        self.use_tta = use_tta

        self._model = None
        self._device = None
        self._transform = None
        self._classes: List[str] = []
        self._temperature = 1.0
        self._meta: Dict = {}
        self._loaded = False
        self._load_failed = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    @property
    def available(self) -> bool:
        return self._model_path.exists()

    @property
    def classes(self) -> List[str]:
        return list(self._classes)

    @property
    def info(self) -> Dict:
        """Model card fields for /health and the UI ('what am I running?')."""
        return {
            "path": str(self._model_path),
            "loaded": self._loaded,
            **self._meta,
        }

    def _resolve_device(self) -> str:
        if self._device_str != "auto":
            return self._device_str
        if _torch.cuda.is_available():
            return "cuda"
        mps = getattr(_torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"

    def _load(self) -> bool:
        if self._loaded:
            return True
        if self._load_failed:
            return False
        with self._lock:
            if self._loaded:
                return True
            if self._load_failed:
                return False
            if not self._model_path.exists():
                logger.info("CNN checkpoint not found at %s", self._model_path)
                self._load_failed = True
                return False
            try:
                _ensure_torch()
                self._device = self._resolve_device()
                ckpt = _torch.load(
                    self._model_path, map_location=self._device, weights_only=False
                )

                arch = ckpt.get("arch", "efficientnet_b0")
                num_classes = ckpt["num_classes"]
                self._model = _build_model(arch, num_classes, ckpt.get("dropout", 0.3))
                self._model.load_state_dict(ckpt["model_state_dict"])
                self._model.to(self._device)
                self._model.eval()

                # Prefer the explicit class list; fall back to the legacy map.
                if ckpt.get("classes"):
                    self._classes = list(ckpt["classes"])
                else:
                    raw = {int(k): v for k, v in ckpt.get("idx_to_label", {}).items()}
                    self._classes = [raw[i] for i in sorted(raw)]

                self._temperature = float(ckpt.get("temperature", 1.0)) or 1.0

                size = int(ckpt.get("input_size", 224))
                mean = ckpt.get("normalize_mean", [0.485, 0.456, 0.406])
                std = ckpt.get("normalize_std", [0.229, 0.224, 0.225])
                self._transform = _T.Compose(
                    [
                        _T.Resize(int(size * 1.14)),
                        _T.CenterCrop(size),
                        _T.ToTensor(),
                        _T.Normalize(mean, std),
                    ]
                )

                self._meta = {
                    "arch": arch,
                    "num_classes": num_classes,
                    "device": self._device,
                    "temperature": round(self._temperature, 4),
                    "best_macro_f1": round(float(ckpt.get("best_macro_f1", 0.0)), 4),
                    "has_reject_classes": bool(REJECT_CLASSES & set(self._classes)),
                }
                self._loaded = True
                logger.info("CNN loaded: %s", self._meta)
                return True
            except Exception:
                logger.exception("Failed to load CNN from %s", self._model_path)
                self._load_failed = True
                return False

    # ------------------------------------------------------------------ #
    def _forward_probs(self, image: Image.Image):
        """Calibrated class probabilities, optionally averaged over TTA views."""
        views = [image]
        if self.use_tta:
            views.append(image.transpose(Image.FLIP_LEFT_RIGHT))

        batch = _torch.stack([self._transform(v) for v in views]).to(self._device)
        with _torch.no_grad():
            logits = self._model(batch) / self._temperature
            probs = _F.softmax(logits, dim=1)
        return probs.mean(dim=0).cpu()

    def predict(self, image_bytes: bytes) -> Optional[Dict]:
        """Classify an image and decide how the request should be routed.

        Returns None only when the model is unavailable or inference blew up, so
        callers can fall back cleanly.
        """
        if not self._load():
            return None
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception:
            logger.warning("Could not decode image for CNN", exc_info=True)
            return None

        try:
            probs = self._forward_probs(image)
        except Exception:
            logger.exception("CNN inference failed")
            return None

        k = min(5, probs.numel())
        top_probs, top_idx = probs.topk(k)
        top = [
            {"label": self._classes[int(i)], "confidence": round(float(p), 4)}
            for p, i in zip(top_probs.tolist(), top_idx.tolist())
        ]

        label = top[0]["label"]
        confidence = top[0]["confidence"]
        margin = confidence - (top[1]["confidence"] if len(top) > 1 else 0.0)

        decision, reason = self._decide(label, confidence, margin)
        return {
            "decision": decision,
            "reason": reason,
            "primary_label": label,
            "confidence": confidence,
            "margin": round(margin, 4),
            "top_predictions": top,
            "in_scope_predictions": [t for t in top if t["label"] not in REJECT_CLASSES],
            "source": "local_cnn",
            "model": self._meta.get("arch"),
        }

    def _decide(self, label: str, confidence: float, margin: float):
        """Map (label, confidence, margin) onto a routing decision.

        The margin guard catches the case the confidence threshold misses: two
        classes at 0.44/0.43 clears no useful bar, but a single softmax peak at
        0.76 with the runner-up at 0.02 is genuinely decisive.
        """
        if label == OOS_NOT_PLANT:
            if confidence >= self.reject_threshold:
                return DECISION_NOT_PLANT, "model is confident this is not a plant"
            return DECISION_ESCALATE, "possibly not a plant, but not confidently"

        if label == OOS_OTHER_PLANT:
            return DECISION_ESCALATE, "a leaf, but not a crop the CNN was trained on"

        if confidence < self.confidence_threshold:
            return DECISION_ESCALATE, (
                f"confidence {confidence:.2f} below threshold "
                f"{self.confidence_threshold:.2f}"
            )
        if margin < self.margin_threshold:
            return DECISION_ESCALATE, (
                f"top-2 margin {margin:.2f} below {self.margin_threshold:.2f}"
            )
        return DECISION_DIAGNOSIS, "confident in-scope prediction"


_default_predictor: Optional[CNNPredictor] = None
_predictor_lock = threading.Lock()


def get_predictor(model_path: str = "", **kwargs) -> CNNPredictor:
    """Process-wide singleton so the weights load once."""
    global _default_predictor
    if _default_predictor is None:
        with _predictor_lock:
            if _default_predictor is None:
                if not model_path:
                    root = Path(__file__).resolve().parent.parent.parent
                    model_path = str(root / "data" / "models" / "agridoctor_cnn_6crop.pt")
                _default_predictor = CNNPredictor(model_path, **kwargs)
    return _default_predictor


def reset_predictor() -> None:
    """Drop the singleton (tests / hot-reload after retraining)."""
    global _default_predictor
    with _predictor_lock:
        _default_predictor = None
