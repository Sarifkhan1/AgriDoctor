"""Shared test fixtures. Sets env + a temp DB BEFORE importing the app."""

import os
import tempfile
from io import BytesIO

# Must be set before backend.config / backend.db are imported.
os.environ.setdefault("SECRET_KEY", "test-secret-key-not-for-production")
os.environ.setdefault("GROQ_API_KEY", "test-groq-key")
os.environ.setdefault("ALLOWED_ORIGINS", "http://localhost:3000")
_tmp_db = os.path.join(tempfile.mkdtemp(prefix="agridoctor_test_"), "test.db")
os.environ["DATABASE_PATH"] = _tmp_db

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from tests.fakes import FakeProvider  # noqa: E402


@pytest.fixture(scope="session")
def app():
    from backend.db import init_db
    from backend.main import app as fastapi_app

    init_db()  # lifespan doesn't run under a bare TestClient
    return fastapi_app


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def png_bytes():
    buf = BytesIO()
    Image.new("RGB", (128, 128), (40, 160, 90)).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def use_fake():
    """Inject a FakeProvider into the live analyze service for the test."""
    from backend.routers import analyze as analyze_router

    def _set(payload: dict, transcript: str = ""):
        analyze_router._service._analyzer._provider = FakeProvider(payload, transcript)

    yield _set
    analyze_router._service._analyzer._provider = None
