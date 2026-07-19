"""Unit tests for the analyzer's parsing + safety invariants (no network)."""

from backend.ai.analyzer import Analyzer
from backend.ai.schemas import ResultKind
from tests.fakes import FakeProvider

IMG = b"fake-image-bytes"


def run(payload, transcript="", **kw):
    az = Analyzer(provider=FakeProvider(payload, transcript))
    return az.analyze(image_bytes=IMG, image_mime="image/png", **kw)


def test_valid_diagnosis_enriches_and_filters():
    r = run(
        {
            "kind": "diagnosis",
            "detected_crop": "tomato",
            "primary_label": "TOM_EARLY_BLIGHT",
            "secondary_labels": ["TOM_SEPTORIA", "NOT_A_REAL_LABEL"],
            "confidence": 0.88,
            "severity_score": 0.5,
            "advice": {"summary": "x", "what_to_do_now": ["y"]},
        },
        crop_hint="tomato",
    )
    assert r.kind == ResultKind.DIAGNOSIS.value
    assert r.disease_name == "Early Blight"  # enriched from taxonomy
    assert r.category == "fungal"
    assert r.secondary_labels == ["TOM_SEPTORIA"]  # bogus label dropped
    assert r.matches_hint is True
    assert r.crop_supported is True


def test_unsupported_crop_is_rejected():
    r = run(
        {
            "kind": "unsupported_crop",
            "detected_crop": "banana",
            "message": "banana leaf",
        }
    )
    assert r.kind == ResultKind.UNSUPPORTED_CROP.value
    assert r.primary_label is None
    assert r.advice is None
    assert r.message


def test_not_a_leaf():
    r = run({"kind": "not_a_leaf", "is_plant": False, "message": "a car"})
    assert r.kind == ResultKind.NOT_A_LEAF.value
    assert r.primary_label is None
    assert r.advice is None


def test_hallucinated_diagnosis_on_unsupported_crop_is_stripped():
    """The critical safety invariant: a disease label on a non-supported crop
    must never leak through."""
    r = run(
        {
            "kind": "diagnosis",
            "detected_crop": "banana",
            "primary_label": "TOM_EARLY_BLIGHT",
            "confidence": 0.95,
            "advice": {"summary": "fake"},
        }
    )
    assert r.kind == ResultKind.UNSUPPORTED_CROP.value
    assert r.primary_label is None
    assert r.advice is None
    assert r.crop_supported is False


def test_livestock_diagnosis_is_supported():
    """Livestock (cattle/goat/sheep/poultry) is a first-class supported subject."""
    r = run(
        {
            "kind": "diagnosis",
            "detected_crop": "cattle",
            "primary_label": "CATTLE_FMD",
            "confidence": 0.8,
            "severity_score": 0.6,
            "advice": {"summary": "call a vet", "what_to_do_now": ["isolate animal"]},
        }
    )
    assert r.kind == ResultKind.DIAGNOSIS.value
    assert r.crop_supported is True
    assert r.disease_name == "Foot and Mouth Disease"
    assert r.category == "viral"


def test_new_crops_pepper_and_eggplant_supported():
    r = run(
        {
            "kind": "diagnosis",
            "detected_crop": "eggplant",
            "primary_label": "EGG_LEAF_SPOT",
            "confidence": 0.7,
        }
    )
    assert r.kind == ResultKind.DIAGNOSIS.value
    assert r.crop_supported is True
    assert r.disease_name == "Cercospora Leaf Spot"


def test_cross_subject_label_is_rejected():
    """A crop label on an animal subject (or vice versa) must not leak through."""
    r = run(
        {
            "kind": "diagnosis",
            "detected_crop": "cattle",
            "primary_label": "TOM_EARLY_BLIGHT",  # a crop label on an animal
            "confidence": 0.95,
            "advice": {"summary": "fake"},
        }
    )
    assert r.kind == ResultKind.LOW_CONFIDENCE.value
    assert r.primary_label is None
    assert r.advice is None


def test_healthy_label_normalizes_kind():
    r = run(
        {"kind": "diagnosis", "detected_crop": "potato", "primary_label": "POT_HEALTHY"}
    )
    assert r.kind == ResultKind.HEALTHY.value


def test_unknown_label_downgrades_to_low_confidence():
    r = run({"kind": "diagnosis", "detected_crop": "rice", "primary_label": "MADE_UP"})
    assert r.kind == ResultKind.LOW_CONFIDENCE.value
    assert r.primary_label is None


def test_confidence_is_clamped():
    r = run(
        {
            "kind": "diagnosis",
            "detected_crop": "tomato",
            "primary_label": "TOM_HEALTHY",
            "confidence": 5.0,
            "severity_score": -3.0,
        }
    )
    assert 0.0 <= r.confidence <= 1.0
    assert 0.0 <= r.severity_score <= 1.0


def test_malformed_json_falls_back():
    class Bad(FakeProvider):
        def analyze_image(self, *a):
            return "totally not json"

    az = Analyzer(provider=Bad({}))
    r = az.analyze(image_bytes=IMG, image_mime="image/png")
    assert r.kind == ResultKind.LOW_CONFIDENCE.value
    assert r.message


def test_serializes_kind_as_string():
    r = run(
        {
            "kind": "diagnosis",
            "detected_crop": "tomato",
            "primary_label": "TOM_EARLY_BLIGHT",
        }
    )
    dumped = r.model_dump()
    assert isinstance(dumped["kind"], str)
    assert isinstance(dumped["urgency_level"], str)
