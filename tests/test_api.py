"""API integration tests via TestClient (FakeProvider injected — no network)."""

import uuid

DIAGNOSIS_PAYLOAD = {
    "kind": "diagnosis",
    "is_plant": True,
    "is_leaf": True,
    "detected_crop": "tomato",
    "primary_label": "TOM_EARLY_BLIGHT",
    "secondary_labels": ["TOM_SEPTORIA"],
    "confidence": 0.87,
    "severity_score": 0.44,
    "urgency_level": "medium",
    "visual_evidence": "concentric brown rings",
    "advice": {
        "summary": "Early blight",
        "what_to_do_now": ["Prune infected leaves"],
        "prevention": ["Rotate crops"],
        "when_to_get_help": ["Stem lesions appear"],
    },
}

UNSUPPORTED_PAYLOAD = {
    "kind": "unsupported_crop",
    "is_plant": True,
    "is_leaf": True,
    "detected_crop": "banana",
    "message": "This looks like a banana leaf.",
}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_analyze_diagnosis_anonymous(client, use_fake, png_bytes):
    use_fake(DIAGNOSIS_PAYLOAD)
    r = client.post(
        "/api/analyze",
        files={"image": ("leaf.png", png_bytes, "image/png")},
        data={"crop_hint": "tomato"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["kind"] == "diagnosis"
    assert body["disease_name"] == "Early Blight"
    assert body["case_id"] is None  # anonymous -> not persisted


def test_analyze_unsupported_crop(client, use_fake, png_bytes):
    use_fake(UNSUPPORTED_PAYLOAD)
    r = client.post(
        "/api/analyze", files={"image": ("leaf.png", png_bytes, "image/png")}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["kind"] == "unsupported_crop"
    assert body["primary_label"] is None
    assert "banana" in (body["message"] or "").lower()


def test_analyze_rejects_bad_image(client):
    r = client.post(
        "/api/analyze", files={"image": ("x.txt", b"not an image", "text/plain")}
    )
    assert r.status_code == 400


def test_authed_analyze_persists_and_shows_in_history(client, use_fake, png_bytes):
    email = f"u_{uuid.uuid4().hex[:8]}@example.com"
    reg = client.post(
        "/api/auth/register",
        json={"email": email, "password": "strongpass1", "full_name": "U"},
    )
    assert reg.status_code == 200
    token = reg.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    use_fake(DIAGNOSIS_PAYLOAD)
    r = client.post(
        "/api/analyze",
        headers=headers,
        files={"image": ("leaf.png", png_bytes, "image/png")},
        data={"crop_hint": "tomato", "notes": "spreading fast"},
    )
    assert r.status_code == 200
    case_id = r.json()["case_id"]
    assert case_id  # persisted for authed user

    hist = client.get("/api/cases", headers=headers)
    assert hist.status_code == 200
    ids = [c["id"] for c in hist.json()]
    assert case_id in ids

    detail = client.get(f"/api/cases/{case_id}", headers=headers)
    assert detail.status_code == 200
    assert detail.json()["prediction"]["disease_name"] == "Early Blight"


def test_weak_password_rejected(client):
    r = client.post(
        "/api/auth/register", json={"email": "w@example.com", "password": "short"}
    )
    assert r.status_code == 422


def test_heatmap_is_returned_but_not_persisted(client, use_fake, png_bytes, monkeypatch):
    """Grad-CAM must reach the client but must never be written to the database.

    The overlay is ~200KB of base64 per diagnosis and is derived data — it can be
    recomputed from the stored image. Persisting it would put a quarter-megabyte
    blob in every `predictions` row, and `GET /api/cases/{id}` does `SELECT *`, so
    every history view would read and ship it back too.
    """
    from backend.routers import analyze as analyze_router
    from backend.ai.schemas import AnalysisResult

    email = f"u_{uuid.uuid4().hex[:8]}@example.com"
    reg = client.post(
        "/api/auth/register",
        json={"email": email, "password": "strongpass1", "full_name": "U"},
    )
    token = reg.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    fake_heatmap = "QUJDRA==" * 200  # stand-in for a real base64 PNG

    # Attach a heatmap the way the local-CNN path does, after analysis completes.
    real_analyze = analyze_router._service._analyzer.analyze

    def with_heatmap(*a, **kw) -> AnalysisResult:
        result = real_analyze(*a, **kw)
        result.heatmap_png_b64 = fake_heatmap
        result.heatmap_focus = 0.37
        return result

    use_fake(DIAGNOSIS_PAYLOAD)
    monkeypatch.setattr(analyze_router._service._analyzer, "analyze", with_heatmap)

    r = client.post(
        "/api/analyze",
        headers=headers,
        files={"image": ("leaf.png", png_bytes, "image/png")},
        data={"crop_hint": "tomato"},
    )
    assert r.status_code == 200
    body = r.json()

    # The live response carries it — the UI needs it to render the overlay.
    assert body["heatmap_png_b64"] == fake_heatmap
    assert body["heatmap_focus"] == 0.37

    # The stored copy must not.
    case_id = body["case_id"]
    detail = client.get(f"/api/cases/{case_id}", headers=headers)
    assert detail.status_code == 200
    raw = detail.json()["prediction"].get("raw_response") or ""
    assert fake_heatmap not in raw, "Grad-CAM blob leaked into the database"
    assert "heatmap_png_b64" not in raw
