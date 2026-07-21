"""
Grad-CAM explainability for the local CNN.

Produces a heatmap over the input showing which pixels drove the predicted class,
so a diagnosis can be shown to the farmer (and to an examiner) as "here is the
lesion I reacted to" rather than an unaccountable label.

This is also a quality check on the model itself: PlantVillage images have plain
lab backgrounds, and a model that has latched onto the background instead of the
lesion produces heatmaps that light up the corners. That failure is invisible in
an accuracy number and obvious in a Grad-CAM.

Implementation is the standard Selvaraju et al. (2017) formulation: weight the
final conv feature maps by the mean gradient of the target logit w.r.t. each
channel, ReLU the weighted sum, upsample to input size.
"""

import base64
import io
import logging
from typing import Dict, Optional

from PIL import Image

logger = logging.getLogger(__name__)


def _target_layer(model, arch: str):
    """Last convolutional stage — the deepest layer that still has spatial extent."""
    if arch.startswith("efficientnet") or arch.startswith("mobilenet"):
        return model.features[-1]
    if arch.startswith("resnet"):
        return model.layer4[-1]
    raise ValueError(f"No Grad-CAM target layer known for arch {arch!r}")


class GradCAM:
    """Compute Grad-CAM overlays for a loaded CNNPredictor."""

    def __init__(self, predictor) -> None:
        self._predictor = predictor

    def explain(
        self,
        image_bytes: bytes,
        class_index: Optional[int] = None,
        alpha: float = 0.45,
    ) -> Optional[Dict]:
        """Return {'overlay_png_b64', 'class_index', 'label', 'peak_fraction'}.

        `peak_fraction` is the share of the image whose activation exceeds half the
        maximum — a rough "how focused was the model" number. Values near 1.0 mean
        the model responded to the whole frame, which is a red flag for background
        shortcutting.
        """
        # Reuse the predictor's lazily-loaded model/transform/device.
        if not self._predictor._load():  # noqa: SLF001 - intentional internal reuse
            return None

        import numpy as np
        import torch
        import torch.nn.functional as F

        model = self._predictor._model  # noqa: SLF001
        device = self._predictor._device  # noqa: SLF001
        transform = self._predictor._transform  # noqa: SLF001
        arch = self._predictor.info.get("arch", "efficientnet_b0")

        try:
            layer = _target_layer(model, arch)
        except ValueError:
            logger.warning("Grad-CAM unsupported for arch %s", arch)
            return None

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception:
            logger.warning("Grad-CAM could not decode image", exc_info=True)
            return None

        activations, gradients = {}, {}

        def fwd_hook(_m, _inp, out):
            activations["v"] = out

        def bwd_hook(_m, _gin, gout):
            gradients["v"] = gout[0]

        h1 = layer.register_forward_hook(fwd_hook)
        h2 = layer.register_full_backward_hook(bwd_hook)

        try:
            tensor = transform(image).unsqueeze(0).to(device)
            # Grad-CAM needs gradients, so no torch.no_grad() here.
            model.zero_grad(set_to_none=True)
            logits = model(tensor)
            idx = int(logits.argmax(1).item()) if class_index is None else int(class_index)
            logits[0, idx].backward()

            acts = activations.get("v")
            grads = gradients.get("v")
            if acts is None or grads is None:
                logger.warning("Grad-CAM hooks captured nothing")
                return None

            weights = grads.mean(dim=(2, 3), keepdim=True)      # channel importance
            cam = F.relu((weights * acts).sum(dim=1, keepdim=True))
            cam = F.interpolate(
                cam, size=(image.height, image.width), mode="bilinear", align_corners=False
            )
            cam = cam[0, 0].detach().cpu().numpy()
        except Exception:
            logger.exception("Grad-CAM computation failed")
            return None
        finally:
            h1.remove()
            h2.remove()
            model.zero_grad(set_to_none=True)

        span = float(cam.max() - cam.min())
        cam = (cam - cam.min()) / (span + 1e-8)
        peak_fraction = float((cam > 0.5).mean())

        overlay = self._render_overlay(image, cam, alpha)
        buf = io.BytesIO()
        overlay.save(buf, "PNG", optimize=True)

        labels = self._predictor.classes
        return {
            "overlay_png_b64": base64.b64encode(buf.getvalue()).decode("ascii"),
            "class_index": idx,
            "label": labels[idx] if idx < len(labels) else None,
            "peak_fraction": round(peak_fraction, 4),
        }

    @staticmethod
    def _render_overlay(image: Image.Image, cam, alpha: float) -> Image.Image:
        """Blend a blue->red heatmap over the original image (no matplotlib dep)."""
        import numpy as np

        base = np.asarray(image).astype(np.float32)

        # Simple perceptual ramp: cold/transparent at low activation, hot at high.
        heat = np.zeros_like(base)
        heat[..., 0] = np.clip(cam * 2.0, 0, 1) * 255          # red rises first
        heat[..., 1] = np.clip(1.0 - np.abs(cam - 0.5) * 2, 0, 1) * 255  # green mid-band
        heat[..., 2] = np.clip((1.0 - cam) * 2.0, 0, 1) * 255  # blue fades out

        # Weight the blend by activation so calm regions keep the original pixels.
        w = (cam[..., None] * alpha).astype(np.float32)
        blended = base * (1 - w) + heat * w
        return Image.fromarray(np.clip(blended, 0, 255).astype("uint8"))


def explain_image(predictor, image_bytes: bytes, **kwargs) -> Optional[Dict]:
    """Convenience wrapper for one-off explanations."""
    return GradCAM(predictor).explain(image_bytes, **kwargs)
