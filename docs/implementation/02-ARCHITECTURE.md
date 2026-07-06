# 02 — Architecture

## 2.1 High-level system diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        Browser (PWA)                             │
│  Responsive UI · bottom navbar · camera/upload · voice recorder  │
│  frontend/js/{api,app,ui}.js  ·  XSS-safe DOM rendering          │
└───────────────┬─────────────────────────────────────────────────┘
                │  HTTPS (multipart: image + optional audio + meta)
                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI backend (backend/)                   │
│                                                                 │
│  routers/analyze.py ──► services/analysis.py (orchestrator)     │
│        │                        │                               │
│        │                        ├─► ai/analyzer.py              │
│        │                        │      ├─ ai/provider.py  ──────┼──► Groq API
│        │                        │      │    (GroqProvider)      │   • Llama 4 Scout (vision)
│        │                        │      ├─ ai/prompts.py         │   • Whisper-v3 (audio)
│        │                        │      └─ ai/schemas.py (Pydantic)
│        │                        │                               │
│        │                        └─► persistence (SQLite)        │
│        │                              cases · media · predictions│
│  routers/{auth,cases}.py  (existing, refactored)                │
└─────────────────────────────────────────────────────────────────┘
```

## 2.2 Request flow — the new instant path

```
1. User selects/captures image  (crop pick optional → sent as a "hint")
2. User optionally records voice + fills onset/spread/notes
3. Frontend POSTs multipart to  POST /api/analyze
        image=<file> [audio=<file>] crop_hint=tomato onset_days=3 spread=fast notes="..."
4. Backend:
   a. Validate upload (size, mime, magic bytes, dimensions)      → 06-SECURITY
   b. If audio present: Groq Whisper → transcript text
   c. Run Text NLU on transcript+notes (src/preprocessing/text_nlu.py) → structured hints
   d. Build the vision prompt: system rules + taxonomy + fused context
   e. Groq Llama 4 Scout (image + text) → raw JSON string
   f. Parse + validate against Pydantic AnalysisResult; repair/fallback if invalid
   g. Persist case + media + prediction (best-effort; never blocks the response)
   h. Return AnalysisResult JSON
5. Frontend renders one of: DIAGNOSIS | UNSUPPORTED_CROP | NOT_A_LEAF | LOW_CONFIDENCE | ERROR
```

Total model calls: **1** (vision) or **2** (vision + Whisper if voice supplied). Both hit Groq's low-latency endpoints, so p95 stays under ~4s.

## 2.3 Backend module layout (target)

```
backend/
├── main.py                 # app factory, middleware, router registration, lifespan
├── config.py               # Settings (pydantic-settings): SECRET_KEY, GROQ_API_KEY, models, limits
├── db.py                   # connection, init, context manager (extracted from main.py)
├── security.py             # password hashing, JWT, get_current_user (extracted)
├── schemas.py              # request/response Pydantic models (extracted)
├── routers/
│   ├── auth.py             # register/login/me
│   ├── cases.py            # list/get history, media
│   └── analyze.py          # POST /api/analyze  (the new instant endpoint)
├── services/
│   └── analysis.py         # orchestrator: validate → transcribe → nlu → vision → persist
└── ai/
    ├── provider.py         # AIProvider ABC + GroqProvider (vision + transcription)
    ├── prompts.py          # system prompt + taxonomy injection + context builder
    ├── schemas.py          # AnalysisResult, AdviceBlock (the strict output contract)
    ├── taxonomy.py         # the 38 labels + metadata, loaded from data/ (single source)
    └── analyzer.py         # ties prompt + provider + schema together, with validation/repair
```

> This is the *refactored* target. You do **not** have to complete the full split before wiring the AI — see the incremental plan in [07-ROADMAP-AND-TASKS.md](./07-ROADMAP-AND-TASKS.md). But new AI code goes in `backend/ai/` from day one, never inside `main.py`.

## 2.4 Data model changes

The existing schema ([`init_db`](../../backend/main.py)) is well-designed and mostly stays. Add columns to `predictions` to store the richer AI output:

```sql
ALTER TABLE predictions ADD COLUMN detected_crop TEXT;      -- what the AI saw
ALTER TABLE predictions ADD COLUMN crop_supported INTEGER;  -- 0/1
ALTER TABLE predictions ADD COLUMN is_leaf INTEGER;         -- 0/1
ALTER TABLE predictions ADD COLUMN visual_evidence TEXT;    -- reasoning text
ALTER TABLE predictions ADD COLUMN raw_response TEXT;       -- full JSON for audit/debug
ALTER TABLE predictions ADD COLUMN provider TEXT;           -- "groq"
ALTER TABLE predictions ADD COLUMN model_id TEXT;           -- e.g. llama-4-scout...
```

Store the transcript on `media` (already has room) or in `metadata.notes_text`. Keep `advice_json`, `secondary_labels_json` as-is.

> For MVP, SQLite is fine. If you later need concurrent writes at scale, migrate to Postgres — the repository pattern in `services/` keeps that a localized change.

## 2.5 Configuration & secrets (summary; full detail in 04 & 06)

All config comes from environment (via `pydantic-settings`), never hardcoded:

```
SECRET_KEY=<required, app refuses to start without it>
GROQ_API_KEY=<required for AI>
GROQ_VISION_MODEL=meta-llama/llama-4-scout-17b-16e-instruct
GROQ_AUDIO_MODEL=whisper-large-v3-turbo
MAX_UPLOAD_MB=10
ALLOWED_ORIGINS=https://agridoctor.cloud,http://localhost:3000
AI_TIMEOUT_SECONDS=30
```

## 2.6 Where the existing `src/` ML pipeline fits

The `src/` code (ViT trainer, multimodal fusion, preprocessing, ASR, NLU) is **not deleted**. Its role changes:

| Module | New role in this architecture |
|--------|-------------------------------|
| `src/preprocessing/text_nlu.py` | **Reused now** — runs on transcript+notes to extract structured hints (crop synonyms, symptoms, urgency keywords) that we inject into the vision prompt. Cheap, deterministic, offline. |
| `src/preprocessing/preprocess_images.py` | **Optionally reused** — blur detection to warn "image too blurry, retake" before spending an API call. |
| `src/preprocessing/asr_transcribe.py` | Superseded by Groq Whisper for the live path; kept for offline/batch transcription. |
| `src/models/train_*.py` | **Future accelerator.** If a labeled dataset is later collected, a fine-tuned ViT can act as a fast, free first-pass classifier, with Llama 4 as the reasoning/advice layer. Documented, not built now. |

This keeps the repo honest: no dead code masquerading as the engine, and a clear upgrade path.

## 2.7 Failure & degradation strategy

| Failure | Behavior |
|---------|----------|
| Groq unreachable / timeout | Return HTTP 503 with a clear message; **do NOT** silently fall back to fake results (current frontend bug — see 06) |
| Model returns malformed JSON | One automatic repair retry with a stricter instruction; if still bad → `LOW_CONFIDENCE`/error state, log raw output |
| Image fails validation | HTTP 400 with actionable message (too large / wrong type / not an image) |
| Audio transcription fails | Proceed with image-only analysis; note "voice could not be processed" |
| `crop_supported=false` | Return a normal 200 with the `UNSUPPORTED_CROP` result — this is a valid outcome, not an error |
