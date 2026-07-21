"""
Tests for the local-CNN-first pipeline.

These cover the decision logic and the analyzer routing built on top of it, using
a fake predictor rather than a trained checkpoint — the routing must be verifiable
without a 40k-image training run, and must behave identically whether or not
anyone has trained a model locally.

The safety cases matter most: a wrong route sends a farmer a confident diagnosis
for a photo of a car, or burns an API call on something answerable offline.
"""

import pytest

from backend.ai.analyzer import Analyzer
from backend.ai.cnn_predictor import (
    DECISION_DIAGNOSIS,
    DECISION_ESCALATE,
    DECISION_NOT_PLANT,
    OOS_NOT_PLANT,
    OOS_OTHER_PLANT,
    CNNPredictor,
)
from backend.ai.schemas import ResultKind
from tests.fakes import FakeProvider


class FakePredictor:
    """Stands in for CNNPredictor with a scripted prediction."""

    def __init__(self, result):
        self._result = result
        self.available = True
        self.calls = 0

    def predict(self, image_bytes):
        self.calls += 1
        return self._result


def cnn_output(label, decision, confidence=0.95, margin=0.5):
    return {
        "decision": decision,
        "reason": "test",
        "primary_label": label,
        "confidence": confidence,
        "margin": margin,
        "top_predictions": [{"label": label, "confidence": confidence}],
        "in_scope_predictions": [{"label": label, "confidence": confidence}],
        "source": "local_cnn",
        "model": "efficientnet_b0",
    }


# --------------------------------------------------------------------------- #
# Decision thresholds
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "label,conf,margin,expected",
    [
        # Confident, in-scope, decisive -> answer locally.
        ("TOM_LATE_BLIGHT", 0.95, 0.80, DECISION_DIAGNOSIS),
        # Below the confidence bar -> escalate rather than guess.
        ("TOM_LATE_BLIGHT", 0.60, 0.50, DECISION_ESCALATE),
        # Confident overall but two classes nearly tied -> escalate.
        ("TOM_LATE_BLIGHT", 0.90, 0.02, DECISION_ESCALATE),
        # Another species: the CNN can't cover it, the hosted model might.
        (OOS_OTHER_PLANT, 0.99, 0.90, DECISION_ESCALATE),
        # Confidently not a plant -> reject locally, no API call.
        (OOS_NOT_PLANT, 0.95, 0.90, DECISION_NOT_PLANT),
        # Weakly "not a plant" -> don't reject on a coin flip; escalate.
        (OOS_NOT_PLANT, 0.40, 0.10, DECISION_ESCALATE),
    ],
)
def test_decide(label, conf, margin, expected):
    p = CNNPredictor("nonexistent.pt", confidence_threshold=0.75,
                     margin_threshold=0.10, reject_threshold=0.60)
    decision, reason = p._decide(label, conf, margin)
    assert decision == expected, reason


def test_missing_checkpoint_is_not_available():
    """A fresh clone with no trained model must degrade, not crash."""
    p = CNNPredictor("data/models/definitely_not_here.pt")
    assert p.available is False
    assert p.predict(b"irrelevant") is None


# --------------------------------------------------------------------------- #
# Analyzer routing
# --------------------------------------------------------------------------- #
def _analyzer(cnn_result, provider_payload=None):
    fake_cnn = FakePredictor(cnn_result)
    provider = FakeProvider(provider_payload or {}) if provider_payload else None
    return Analyzer(provider=provider, cnn_predictor=fake_cnn), fake_cnn


def test_confident_diagnosis_answered_locally(monkeypatch, png_bytes):
    """A confident in-scope prediction must not touch the network at all."""
    analyzer, fake_cnn = _analyzer(
        cnn_output("TOM_LATE_BLIGHT", DECISION_DIAGNOSIS)
    )
    # Any provider access would raise, proving no hosted call was made.
    monkeypatch.setattr(
        Analyzer, "provider",
        property(lambda self: pytest.fail("hosted model must not be called")),
    )

    result = analyzer.analyze(image_bytes=png_bytes, image_mime="image/png")

    assert fake_cnn.calls == 1
    assert result.kind == ResultKind.DIAGNOSIS
    assert result.primary_label == "TOM_LATE_BLIGHT"
    assert result.detected_crop == "tomato"
    assert result.provider == "local_cnn"
    # Advice must come from the offline knowledge base, fully populated.
    assert result.advice is not None
    assert result.advice.what_to_do_now
    assert result.advice.prevention


def test_healthy_label_maps_to_healthy_kind(monkeypatch, png_bytes):
    analyzer, _ = _analyzer(cnn_output("POT_HEALTHY", DECISION_DIAGNOSIS))
    monkeypatch.setattr(
        Analyzer, "provider",
        property(lambda self: pytest.fail("hosted model must not be called")),
    )
    result = analyzer.analyze(image_bytes=png_bytes, image_mime="image/png")
    assert result.kind == ResultKind.HEALTHY
    assert result.severity_score == 0.0


def test_not_plant_rejected_locally(monkeypatch, png_bytes):
    """The reject class must short-circuit before any API call."""
    analyzer, _ = _analyzer(cnn_output(OOS_NOT_PLANT, DECISION_NOT_PLANT))
    monkeypatch.setattr(
        Analyzer, "provider",
        property(lambda self: pytest.fail("hosted model must not be called")),
    )
    result = analyzer.analyze(image_bytes=png_bytes, image_mime="image/png")
    assert result.kind == ResultKind.NOT_A_LEAF
    # No disease content may leak on a rejection.
    assert result.primary_label is None
    assert result.advice is None


def test_escalation_reaches_hosted_model(png_bytes):
    """An unknown species must be handed to the hosted model, not guessed at."""
    analyzer, _ = _analyzer(
        cnn_output(OOS_OTHER_PLANT, DECISION_ESCALATE),
        provider_payload={
            "kind": "diagnosis",
            "is_plant": True,
            "is_leaf": True,
            "detected_crop": "rice",
            "primary_label": "RICE_BLAST",
            "confidence": 0.88,
        },
    )
    result = analyzer.analyze(image_bytes=png_bytes, image_mime="image/png")
    assert result.kind == ResultKind.DIAGNOSIS
    assert result.detected_crop == "rice"
    assert result.provider == "groq"


def test_escalation_without_api_key_is_honest(monkeypatch, png_bytes):
    """With no key configured we say we're unsure — never invent a diagnosis."""
    from backend import config

    settings = config.get_settings()
    monkeypatch.setattr(settings, "groq_api_key", "", raising=False)

    analyzer, _ = _analyzer(cnn_output(OOS_OTHER_PLANT, DECISION_ESCALATE))
    result = analyzer.analyze(image_bytes=png_bytes, image_mime="image/png")

    assert result.kind == ResultKind.LOW_CONFIDENCE
    assert result.primary_label is None
    assert result.message


# --------------------------------------------------------------------------- #
# Offline advice knowledge base
# --------------------------------------------------------------------------- #
def test_every_advice_label_is_in_the_taxonomy():
    """Advice keyed to a label the taxonomy doesn't know would never be served."""
    from backend.ai import local_advice
    from backend.ai.taxonomy import all_label_ids

    unknown = set(local_advice._ADVICE) - set(all_label_ids())
    assert not unknown, f"advice for unknown labels: {unknown}"


def test_advice_blocks_are_complete():
    """Every non-healthy entry needs actionable content in all three sections."""
    from backend.ai import local_advice

    for label in local_advice._ADVICE:
        advice = local_advice.advice_for(label)
        assert advice is not None, label
        assert advice.summary.strip(), label
        assert advice.what_to_do_now, label
        assert advice.prevention, label
        assert advice.when_to_get_help, label


def test_urgency_tracks_severity():
    from backend.ai import local_advice
    from backend.ai.schemas import Urgency

    # Late blight can destroy a crop in days — it must not be a "medium".
    assert local_advice.urgency_for("TOM_LATE_BLIGHT") == Urgency.high
    assert local_advice.urgency_for("POT_HEALTHY") == Urgency.low


def test_contextual_notes_react_to_farmer_input():
    from backend.ai import local_advice

    assert local_advice.contextual_notes(onset_days=1)
    assert local_advice.contextual_notes(spread="fast")
    assert local_advice.contextual_notes(notes="it has been raining all week")
    assert local_advice.contextual_notes() == []
