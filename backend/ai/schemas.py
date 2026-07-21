"""
AI analysis output contract.

Everything the UI renders is defined here. The vision model is instructed to
produce exactly this shape; the backend validates it (see analyzer.py). The two
must stay in sync.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ResultKind(str, Enum):
    DIAGNOSIS = "diagnosis"  # supported crop, disease identified
    HEALTHY = "healthy"  # supported crop, no disease
    UNSUPPORTED_CROP = "unsupported_crop"  # a plant/leaf, but not one of the 6
    NOT_A_LEAF = "not_a_leaf"  # not a plant/leaf at all
    LOW_CONFIDENCE = "low_confidence"  # image unusable (blurry/dark/too far)


class Urgency(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class AdviceBlock(BaseModel):
    summary: str = ""
    what_to_do_now: list[str] = Field(default_factory=list)
    prevention: list[str] = Field(default_factory=list)
    when_to_get_help: list[str] = Field(default_factory=list)


DEFAULT_SAFETY_NOTE = (
    "This is AI-generated guidance, not a substitute for a local agronomist. "
    "Confirm the diagnosis and any chemical treatment with an expert before applying."
)


class AnalysisResult(BaseModel):
    kind: ResultKind

    # Perception
    is_plant: bool = False
    is_leaf: bool = False
    detected_crop: Optional[str] = None
    crop_supported: bool = False
    matches_hint: Optional[bool] = None
    image_quality: str = "ok"  # ok | blurry | dark | too_far | occluded

    # Diagnosis (present only for DIAGNOSIS / HEALTHY)
    primary_label: Optional[str] = None
    disease_name: Optional[str] = None
    category: Optional[str] = None
    secondary_labels: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    severity_score: float = 0.0
    urgency_level: Urgency = Urgency.medium
    visual_evidence: Optional[str] = None

    # Explainability (local CNN only — the hosted model exposes no activations).
    # Base64 PNG of the Grad-CAM overlay showing which pixels drove the label.
    heatmap_png_b64: Optional[str] = None
    # Share of the image above half-peak activation. Near 1.0 means the model
    # responded to the whole frame rather than a lesion — treat as a weak signal.
    heatmap_focus: Optional[float] = None

    # Guidance
    advice: Optional[AdviceBlock] = None
    message: Optional[str] = None
    safety_note: str = DEFAULT_SAFETY_NOTE

    # Provenance (filled by the backend, not the model)
    provider: str = "groq"
    model_id: Optional[str] = None

    model_config = {"use_enum_values": True, "validate_assignment": True}
