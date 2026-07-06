# 03 — AI Engine Specification (the core)

This is the heart of the rebuild. It defines exactly how the leaf image + voice + text become a validated diagnosis using Groq's open-source models.

---

## 3.1 Dependencies

Add to `requirements.txt` (and remove the heavy unused ML stack from the **serving** image — see 06/07):

```
groq>=0.11.0
pydantic-settings>=2.1.0
python-magic>=0.4.27      # magic-byte MIME sniffing for upload validation
Pillow>=10.1.0            # already present; used for dimension checks
```

Groq's SDK is OpenAI-compatible, so the mental model is familiar.

## 3.2 Configuration (`backend/config.py`)

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Security — no insecure defaults; app must fail without these
    secret_key: str
    access_token_expire_minutes: int = 60 * 24

    # Groq
    groq_api_key: str
    groq_vision_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    groq_audio_model: str = "whisper-large-v3-turbo"
    ai_timeout_seconds: int = 30
    ai_max_retries: int = 2

    # Uploads / CORS
    max_upload_mb: int = 10
    allowed_origins: str = "http://localhost:3000"

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

settings = Settings()  # raises at import if secret_key / groq_api_key missing → fail fast
```

## 3.3 The output contract (`backend/ai/schemas.py`)

Everything the UI renders is defined here. The model is *instructed* to produce exactly this shape, and we *validate* it — the two must stay in sync.

```python
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

class ResultKind(str, Enum):
    DIAGNOSIS = "diagnosis"            # supported crop, disease identified
    HEALTHY = "healthy"               # supported crop, no disease
    UNSUPPORTED_CROP = "unsupported_crop"  # a plant/leaf, but not one of the 6
    NOT_A_LEAF = "not_a_leaf"         # not a plant/leaf at all
    LOW_CONFIDENCE = "low_confidence" # image unusable (blurry/dark/too far)

class Urgency(str, Enum):
    low = "low"; medium = "medium"; high = "high"

class AdviceBlock(BaseModel):
    summary: str
    what_to_do_now: list[str] = Field(default_factory=list)
    prevention: list[str] = Field(default_factory=list)
    when_to_get_help: list[str] = Field(default_factory=list)

class AnalysisResult(BaseModel):
    kind: ResultKind

    # Perception
    is_plant: bool = False
    is_leaf: bool = False
    detected_crop: Optional[str] = None      # free text: what the AI actually sees
    crop_supported: bool = False
    matches_hint: Optional[bool] = None       # did detected crop match the user's Step-1 pick?
    image_quality: str = "ok"                 # ok | blurry | dark | too_far | occluded

    # Diagnosis (present only for DIAGNOSIS / HEALTHY)
    primary_label: Optional[str] = None       # taxonomy code e.g. TOM_EARLY_BLIGHT
    disease_name: Optional[str] = None        # human name
    category: Optional[str] = None            # fungal|bacterial|viral|pest|nutrient|healthy
    secondary_labels: list[str] = Field(default_factory=list)
    confidence: float = 0.0                    # 0..1
    severity_score: float = 0.0                # 0..1
    urgency_level: Urgency = Urgency.medium
    visual_evidence: Optional[str] = None      # what in the image supports this

    # Guidance
    advice: Optional[AdviceBlock] = None
    message: Optional[str] = None              # user-facing note for non-diagnosis kinds
    safety_note: str = (
        "This is AI-generated guidance, not a substitute for a local agronomist. "
        "Confirm before applying chemical treatments."
    )

    # Provenance (filled by backend, not the model)
    model_id: Optional[str] = None
    provider: str = "groq"
```

## 3.4 The taxonomy (`backend/ai/taxonomy.py`)

The 38 labels from [CROP_DISEASE_TAXONOMY.md](../CROP_DISEASE_TAXONOMY.md) become a machine-readable single source, injected into the prompt so the model returns **our** label codes (not free-form disease names). Ship it as `data/taxonomy.json` and load once:

```python
import json
from pathlib import Path

SUPPORTED_CROPS = ["tomato", "potato", "rice", "maize", "chili", "cucumber"]

_TAXONOMY = json.loads((Path(__file__).parent.parent.parent
                        / "data" / "taxonomy.json").read_text())

def labels_for_prompt() -> str:
    """Render the allowed label codes + descriptions for the system prompt."""
    lines = []
    for crop, entries in _TAXONOMY.items():
        lines.append(f"\n{crop.upper()}:")
        for e in entries:
            lines.append(f"  - {e['label_id']}: {e['disease_name']} "
                         f"({e['category']}) — {e['description']}")
    return "\n".join(lines)
```

`data/taxonomy.json` is generated once from the existing taxonomy doc (a small script or hand-authored). Each entry: `{label_id, crop, disease_name, category, description}`.

## 3.5 The vision system prompt (`backend/ai/prompts.py`)

This prompt is the product. It encodes the crop-support check, the leaf/OOD rejection, the taxonomy constraint, and the strict JSON shape.

```python
from .taxonomy import SUPPORTED_CROPS, labels_for_prompt

SYSTEM_PROMPT = f"""You are AgriDoctor, an expert agricultural plant pathologist.
You analyze a photo of a plant leaf (optionally with a farmer's spoken/typed notes)
and return a STRICT JSON diagnosis. You reason from what is ACTUALLY VISIBLE in the
image — never from assumptions or the farmer's crop guess.

SUPPORTED CROPS (the only ones you can diagnose): {", ".join(SUPPORTED_CROPS)}.

DECISION PROCEDURE (follow in order):
1. Is the image a plant/leaf at all? If it shows a person, object, animal, screenshot,
   or anything non-botanical → set kind="not_a_leaf", is_plant=false, and explain in
   "message". Do NOT invent a diagnosis.
2. If it IS a plant, identify the crop species from leaf shape/venation/margin/color.
   Put your best identification in "detected_crop" (free text, e.g. "banana").
3. Is detected_crop one of the supported crops? If NOT → set kind="unsupported_crop",
   crop_supported=false, and write a friendly "message" naming what you see and listing
   the supported crops. Do NOT diagnose.
4. If the image is too blurry/dark/far/occluded to judge → kind="low_confidence",
   set image_quality accordingly, and ask the farmer (in "message") to retake a close,
   well-lit photo of a single affected leaf.
5. If it is a SUPPORTED crop and judgeable:
   - Diagnose the most likely condition. Choose primary_label ONLY from the allowed
     label codes below for that crop. Add up to 2 differentials in secondary_labels.
   - If no disease is visible → kind="healthy", primary_label the crop's *_HEALTHY code.
   - Otherwise kind="diagnosis".
   - Set confidence (0..1) = your genuine certainty. Set severity_score (0..1) from the
     visible extent of damage. Set urgency_level from severity + spread info in notes.
   - Fill "visual_evidence" with the specific visible signs you used.
   - Write "advice" tailored to the crop, disease, AND the farmer's notes (weather,
     duration, spread, treatments already tried). Be concrete and safe: prefer cultural
     controls first, name chemical options generically, and always include dosage/safety
     caveats and organic alternatives where relevant.

ALLOWED LABEL CODES:
{labels_for_prompt()}

OUTPUT RULES:
- Respond with a SINGLE JSON object matching the AnalysisResult schema. No prose, no
  markdown, no code fences.
- Use ONLY the label codes listed above for primary_label/secondary_labels.
- Never fabricate a diagnosis for unsupported crops or non-leaves.
- confidence and severity_score are numbers between 0 and 1.
"""

def build_user_context(*, crop_hint, transcript, notes, onset_days, spread, nlu_hints):
    """Assemble the farmer-provided context that accompanies the image."""
    parts = ["Analyze the attached leaf image."]
    if crop_hint:
        parts.append(f"Farmer's crop guess (a HINT ONLY, verify against the image): {crop_hint}.")
    if transcript:
        parts.append(f"Farmer's voice note (transcribed): \"{transcript}\"")
    if notes:
        parts.append(f"Farmer's typed notes: \"{notes}\"")
    if onset_days is not None:
        parts.append(f"Symptoms first noticed ~{onset_days} days ago.")
    if spread:
        parts.append(f"Reported spread speed: {spread}.")
    if nlu_hints:
        parts.append(f"Extracted symptom keywords: {nlu_hints}.")
    return " ".join(parts)
```

## 3.6 Provider interface + Groq implementation (`backend/ai/provider.py`)

The interface (D1 in decisions) makes the vendor swappable.

```python
from abc import ABC, abstractmethod
from groq import Groq
from ..config import settings

class AIProvider(ABC):
    @abstractmethod
    def transcribe(self, audio_bytes: bytes, filename: str) -> str: ...
    @abstractmethod
    def analyze_image(self, image_data_url: str, system_prompt: str,
                      user_context: str) -> str: ...  # returns raw JSON string

class GroqProvider(AIProvider):
    def __init__(self):
        self._client = Groq(api_key=settings.groq_api_key,
                            timeout=settings.ai_timeout_seconds)

    def transcribe(self, audio_bytes: bytes, filename: str) -> str:
        resp = self._client.audio.transcriptions.create(
            file=(filename, audio_bytes),
            model=settings.groq_audio_model,
            response_format="text",
        )
        return resp if isinstance(resp, str) else getattr(resp, "text", "")

    def analyze_image(self, image_data_url, system_prompt, user_context) -> str:
        completion = self._client.chat.completions.create(
            model=settings.groq_vision_model,
            temperature=0.2,                      # low → consistent, factual
            max_tokens=1400,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": user_context},
                    {"type": "image_url",
                     "image_url": {"url": image_data_url}},
                ]},
            ],
        )
        return completion.choices[0].message.content
```

Notes:
- **Image is passed as a base64 `data:` URL** (`data:image/jpeg;base64,....`). Groq accepts data URLs for vision models; no need to host the image publicly.
- `response_format={"type":"json_object"}` forces valid JSON.
- `temperature=0.2` keeps diagnoses stable across identical images.

## 3.7 Orchestrator (`backend/ai/analyzer.py`)

```python
import base64, json
from pydantic import ValidationError
from .provider import AIProvider, GroqProvider
from .prompts import SYSTEM_PROMPT, build_user_context
from .schemas import AnalysisResult, ResultKind
from ..config import settings

class Analyzer:
    def __init__(self, provider: AIProvider | None = None):
        self.provider = provider or GroqProvider()

    def _data_url(self, image_bytes: bytes, mime: str) -> str:
        b64 = base64.b64encode(image_bytes).decode()
        return f"data:{mime};base64,{b64}"

    def analyze(self, *, image_bytes, image_mime, audio=None, crop_hint=None,
                notes=None, onset_days=None, spread=None, nlu_hints=None) -> AnalysisResult:
        transcript = ""
        if audio:
            try:
                transcript = self.provider.transcribe(audio["bytes"], audio["filename"])
            except Exception:
                transcript = ""   # image-only fallback; note it downstream

        context = build_user_context(
            crop_hint=crop_hint, transcript=transcript, notes=notes,
            onset_days=onset_days, spread=spread, nlu_hints=nlu_hints,
        )
        data_url = self._data_url(image_bytes, image_mime)

        raw = self.provider.analyze_image(data_url, SYSTEM_PROMPT, context)
        result = self._parse(raw, retry_ctx=(data_url, context))
        result.model_id = settings.groq_vision_model
        result.provider = "groq"
        # Server-side guardrails (never trust the model blindly):
        result = self._enforce_invariants(result, crop_hint)
        return result

    def _parse(self, raw: str, retry_ctx) -> AnalysisResult:
        try:
            return AnalysisResult(**json.loads(raw))
        except (json.JSONDecodeError, ValidationError):
            # One repair attempt with a stricter nudge
            data_url, context = retry_ctx
            fixed = self.provider.analyze_image(
                data_url,
                SYSTEM_PROMPT + "\nREMINDER: return ONLY valid JSON for AnalysisResult.",
                context,
            )
            try:
                return AnalysisResult(**json.loads(fixed))
            except Exception:
                return AnalysisResult(
                    kind=ResultKind.LOW_CONFIDENCE,
                    message="We couldn't analyze this image reliably. "
                            "Please retake a clear, close photo of a single affected leaf.",
                )

    def _enforce_invariants(self, r: AnalysisResult, crop_hint) -> AnalysisResult:
        # Clamp numerics
        r.confidence = min(max(r.confidence, 0.0), 1.0)
        r.severity_score = min(max(r.severity_score, 0.0), 1.0)
        # Non-diagnosis kinds must not carry a disease label
        if r.kind in (ResultKind.UNSUPPORTED_CROP, ResultKind.NOT_A_LEAF,
                      ResultKind.LOW_CONFIDENCE):
            r.primary_label = None; r.disease_name = None; r.advice = None
        # A supported diagnosis must use a known crop
        from .taxonomy import SUPPORTED_CROPS
        if r.kind in (ResultKind.DIAGNOSIS, ResultKind.HEALTHY):
            r.crop_supported = (r.detected_crop or "").lower() in SUPPORTED_CROPS
        # Hint match flag for the UI ("you selected tomato but this looks like potato")
        if crop_hint and r.detected_crop:
            r.matches_hint = crop_hint.lower() == r.detected_crop.lower()
        return r
```

## 3.8 Out-of-distribution / "not a supported crop" behavior (explicit)

This is a first-class requirement. Three distinct rejection states, each with its own UI (see 05):

| Situation | `kind` | What the user sees |
|-----------|--------|--------------------|
| Photo isn't a plant/leaf (selfie, object, screenshot) | `not_a_leaf` | "This doesn't look like a plant leaf. Please photograph a leaf." |
| A leaf, but crop ∉ 6 supported | `unsupported_crop` | "This looks like a **banana** leaf. AgriDoctor currently supports Tomato, Potato, Rice, Maize, Chili, and Cucumber." |
| Supported crop but image unusable | `low_confidence` | "Image is too blurry/dark. Retake a close, well-lit photo." |

Two layers enforce this so a single hallucination can't leak a fake diagnosis:
1. **Prompt** — the decision procedure forbids diagnosing non-leaves/unsupported crops.
2. **Server invariant** — `_enforce_invariants` strips any disease fields if `crop_supported` is false or `kind` is a rejection state.

## 3.9 Confidence & severity calibration

The model's self-reported `confidence` is a starting point, not gospel. Guardrails:
- If `confidence < 0.45` on a `diagnosis`, downgrade the UI presentation to "possible" and surface the top differential prominently (don't hard-assert).
- `severity_score` drives the infection-level meter and, combined with reported spread speed, the `urgency_level`.
- Log `raw_response` for every analysis so you can audit calibration over time and later fine-tune the prompt.

## 3.10 Cost & latency notes (Groq free tier)

- One vision call ≈ hundreds–low-thousands of tokens; Whisper turbo transcribes short notes in well under a second.
- Groq free tier has **per-minute request/token limits** — enforce app-side rate limiting (04 §4.5) so a burst doesn't 429 real users.
- Cache nothing image-specific (each leaf is unique), but do short-circuit obviously bad uploads (blur/size) *before* the API call to save quota.

## 3.11 Testing the engine (see 07 for the full plan)

- **Golden images**: keep a small `tests/fixtures/` set — a diseased tomato leaf, a healthy tomato leaf, a banana leaf (unsupported), and a non-plant photo. Assert the `kind` and, for the tomato pair, that the labels differ.
- **Provider mock**: `AIProvider` is an ABC — inject a `FakeProvider` returning canned JSON to unit-test `Analyzer._parse` / `_enforce_invariants` with **no network and no cost**.
