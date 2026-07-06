# 07 — Roadmap, Tasks & Acceptance

Phased so each phase is independently shippable and testable. Effort is rough (solo dev, focused).

---

## Phase 0 — Security hardening & cleanup  *(prerequisite, ~0.5–1 day)*

> Full detail in [06-SECURITY-HARDENING.md](./06-SECURITY-HARDENING.md).

- [ ] Rotate VPS root password; delete `ssh_run.py`; scrub git history; force-push.
- [ ] Generate fresh `SECRET_KEY`; remove insecure default; require via env.
- [ ] Populate `.gitignore`; confirm DB/logs/uploads/.env untracked.
- [ ] Restrict CORS; add CSP + security headers.
- [ ] Get a free **Groq API key** → `.env` (`GROQ_API_KEY`).

**Exit:** no secrets in repo; app refuses to start without `SECRET_KEY`/`GROQ_API_KEY`.

---

## Phase 1 — Real AI engine (backend)  *(~2–3 days)*

> Detail in [03-AI-ENGINE-SPEC.md](./03-AI-ENGINE-SPEC.md) & [04-BACKEND-API.md](./04-BACKEND-API.md).

- [ ] Add deps: `groq`, `pydantic-settings`, `python-magic`.
- [ ] `backend/config.py` — Settings with fail-fast secrets.
- [ ] Generate `data/taxonomy.json` from [CROP_DISEASE_TAXONOMY.md](../CROP_DISEASE_TAXONOMY.md); `backend/ai/taxonomy.py`.
- [ ] `backend/ai/schemas.py` — `AnalysisResult`, `AdviceBlock`, `ResultKind`.
- [ ] `backend/ai/prompts.py` — system prompt + context builder.
- [ ] `backend/ai/provider.py` — `AIProvider` ABC + `GroqProvider` (vision + Whisper).
- [ ] `backend/ai/analyzer.py` — orchestration, JSON parse/repair, invariants.
- [ ] `backend/services/validation.py` — image/audio validation.
- [ ] `backend/services/analysis.py` — orchestrator (threadpool + best-effort persist).
- [ ] `backend/routers/analyze.py` — `POST /api/analyze`; `get_current_user_optional`.
- [ ] Wire into `main.py`; keep old routes; **delete `generate_mock_prediction`**.
- [ ] `ALTER TABLE predictions` (02 §2.4) for richer output.

**Exit:** `curl -F image=@tomato_blight.jpg http://localhost:8000/api/analyze` returns a real, image-derived diagnosis; a banana leaf returns `unsupported_crop`; a non-plant returns `not_a_leaf`.

---

## Phase 2 — Voice + text fusion  *(~1 day)*

- [ ] Groq Whisper transcription path in `GroqProvider.transcribe`.
- [ ] `backend/services/nlu_hints.py` wrapping [`src/preprocessing/text_nlu.py`](../../src/preprocessing/text_nlu.py) to extract deterministic hints.
- [ ] Inject transcript + notes + NLU hints into the vision context (03 §3.5).
- [ ] Verify a voice note changes the advice/urgency vs image-only.

**Exit:** identical image + "spreading fast after heavy rain" yields higher urgency and rain-aware advice than the image alone.

---

## Phase 3 — Frontend redesign (app UI)  *(~2–3 days)*

> Detail in [05-FRONTEND-REDESIGN.md](./05-FRONTEND-REDESIGN.md).

- [ ] Bottom navigation bar (mobile) + desktop top nav; raised center "Diagnose" CTA.
- [ ] `frontend/js/ui.js` — `esc`/`mdSafe` + XSS-safe `render*` builders for every `kind`.
- [ ] `api.analyze()` client method; **remove `simulateAnalysis()`** and the silent fallback.
- [ ] Result states: diagnosis, healthy, unsupported_crop, not_a_leaf, low_confidence, error.
- [ ] Crop Step-1 "Skip — auto-detect" option; crop-mismatch nudge banner.
- [ ] Responsive pass at 360/390/768/1024/1440; dark mode; safe-area insets.
- [ ] PWA: `manifest.webmanifest`, `sw.js` (app-shell cache, never `/api`), icons.

**Exit:** fully usable on a 360px phone, installs to home screen, renders all six result states correctly, no XSS.

---

## Phase 4 — Hardening, persistence & polish  *(~1–2 days)*

- [ ] Rate limiting (`/api/analyze` 10/min, `/api/auth/login` 5/min).
- [ ] Backend logging/observability (04 §4.6); store `raw_response`.
- [ ] History page reads real saved analyses (authed); anonymous stays ephemeral.
- [ ] Pre-flight blur/size check (reuse `preprocess_images.detect_blur`) to save quota.
- [ ] Slim the serving Docker image — the API no longer needs `torch`/`whisper`/`opencv` at runtime (move heavy ML deps to a separate `requirements-ml.txt` used only by `src/` training).

**Exit:** predictable quota use, real history, meaningful logs, smaller image.

---

## Phase 5 — Testing, docs & deploy  *(~1–2 days)*

- [ ] Unit tests with a `FakeProvider` (no network/cost) for `Analyzer` parse/invariants.
- [ ] Golden-image integration tests (diseased/healthy tomato, banana, non-plant).
- [ ] `pytest` + FastAPI `TestClient`; in-memory SQLite fixture; CI (GitHub Actions) running lint (`black`/`isort`) + tests.
- [ ] Update `README.md` and `.env.example`.
- [ ] Deploy via existing `docker-compose` (with real env secrets); verify on `agridoctor.cloud`.

**Exit:** green CI, reproducible deploy, docs match reality.

---

## Test plan (concrete)

| Test | Input | Expected |
|------|-------|----------|
| Diseased vs healthy differ | two tomato photos | different `primary_label`; not both "early blight" |
| Real image dependence | same crop_hint, different leaves | different diagnoses (proves image is read) |
| Unsupported crop | banana leaf | `kind=unsupported_crop`, names "banana", no disease fields |
| Not a leaf | a selfie / object | `kind=not_a_leaf`, no diagnosis |
| Bad image | blurry/dark | `kind=low_confidence`, retake message |
| Voice fusion | image + "spreading fast after rain" | higher urgency + rain-aware advice |
| Oversized upload | 20 MB file | HTTP 400 |
| Spoofed type | `.jpg` that's actually a PDF | HTTP 400 (magic-byte check) |
| XSS | notes = `<img src=x onerror=alert(1)>` | rendered as inert text, no script |
| Provider down | Groq unreachable | HTTP 503, honest error UI, **no fake result** |
| Anonymous analyze | no token | 200 with result, no `case_id` |
| Authed analyze | valid token | 200 with result + `case_id`; appears in history |

---

## Effort summary

| Phase | Est. |
|-------|------|
| 0 — Security | 0.5–1 d |
| 1 — AI engine | 2–3 d |
| 2 — Voice fusion | 1 d |
| 3 — Frontend | 2–3 d |
| 4 — Hardening/polish | 1–2 d |
| 5 — Test/deploy | 1–2 d |
| **Total** | **~8–12 days** |

---

## Definition of Done (whole project)

- [ ] The engine reads the actual image (proven by the diseased-vs-healthy and image-dependence tests).
- [ ] Unsupported crops and non-leaves are rejected clearly — never a fabricated diagnosis.
- [ ] Voice/text measurably influence the result.
- [ ] Results return < 4s p95; no silent fake fallback anywhere.
- [ ] No secrets in the repo; XSS/CORS/upload issues closed.
- [ ] Modern, responsive, installable app UI with a bottom navbar.
- [ ] Green CI with unit + golden-image tests.
