# 01 — Overview & Decisions

## 1.1 The problem (precisely stated)

The current system *simulates* intelligence:

| Claim in the UI | Reality in code |
|-----------------|-----------------|
| "Our AI analyzes visual patterns" | The uploaded image is written to disk and **never read again** ([`upload_image`](../../backend/main.py)) |
| "Instantly identify crop diseases from photos" | Result is picked from a static dict keyed on the crop **button**, not the image ([`generate_mock_prediction`](../../backend/main.py)) |
| "Powered by advanced machine learning" | The ViT / multimodal models in [`src/models/`](../../src/models/) are never imported or called by the backend |
| Voice description influences diagnosis | Audio is uploaded but never transcribed or used |

**Consequence:** every tomato photo returns "Early Blight," every potato returns "Late Blight," regardless of what the leaf actually shows. A photo of a car, a hand, or a banana leaf returns a confident crop-disease diagnosis. This is the root defect we are fixing.

## 1.2 Target product

A farmer opens the app, takes/uploads a photo of a leaf, optionally holds to record a voice note ("leaves turning yellow with brown spots, spreading fast after rain"), and taps **Analyze**. Within a couple of seconds they get:

1. **Crop identification** — what plant the AI actually sees in the image.
2. **Support check** — if it's not one of the 6 supported crops (or not a leaf/plant at all), a clear, friendly message instead of a fake diagnosis.
3. **Disease diagnosis** — primary condition from the taxonomy, with differential (secondary) possibilities.
4. **Visual evidence** — what in the image led to the conclusion (e.g., "concentric brown rings on lower leaves").
5. **Severity, urgency, and confidence** — calibrated 0–100%.
6. **Actionable advice** — immediate steps, prevention, and escalation triggers, fused with the farmer's described symptoms/weather/duration.
7. **A safety disclaimer** — always.

All behind a modern, mobile-first, installable (PWA) interface with an app-style bottom navigation bar.

## 1.3 Why a vision-language model instead of training CNNs

The repo's original design was to train a ViT per-crop classifier on a large labeled dataset. That is a valid long-term path but has real blockers **right now**:

- There is **no dataset** in the repo — training cannot happen.
- A 38-class classifier gives a label but **no reasoning, no advice, and no out-of-distribution rejection** ("this is a banana leaf") — all explicitly requested.
- It cannot fuse free-form voice/text.

A modern open-source vision-language model (Llama 4 Scout on Groq) delivers all of the above **today**, with zero training, and returns natural-language reasoning + structured advice in one call. We therefore make it the **primary engine** and reposition `src/` as an optional future accelerator (see [02-ARCHITECTURE.md §2.6](./02-ARCHITECTURE.md)).

## 1.4 Key decisions

| # | Decision | Rationale | Reversible? |
|---|----------|-----------|-------------|
| D1 | **Groq + open-source Llama 4 Scout** is the vision engine | Fastest inference, free tier, OSS weights, strong reasoning | Yes — hidden behind `AIProvider` interface (§03) |
| D2 | **Whisper-large-v3-turbo on Groq** for voice | OSS, sub-second, same provider/key | Yes |
| D3 | **New synchronous `POST /api/analyze`** returns the full result in one request | "Everything instant" — removes the create→upload→job→poll dance | Yes — old async job flow can stay for batch |
| D4 | **AI auto-detects the crop**; the Step-1 crop pick becomes an optional *hint* | Enables true image-based diagnosis + OOD detection | Yes |
| D5 | **Strict JSON output contract** validated with Pydantic | Deterministic UI rendering; guards against model drift | Yes |
| D6 | **Phase 0 security hardening is mandatory first** | Live root password + JWT key are committed to git today | No — non-negotiable |
| D7 | **Keep FastAPI + SQLite + vanilla JS** | The bones are fine; a framework rewrite is wasted effort | Yes |
| D8 | **Anonymous "instant" analysis allowed; login only gates history** | Farmers shouldn't need an account to get help | Yes |

## 1.5 Non-goals (explicitly out of scope for this rebuild)

- Livestock analysis (deferred to V1 per [PROJECT_SCOPE.md](../PROJECT_SCOPE.md)).
- Training/fine-tuning custom models (the `src/` pipeline stays as-is, unused at serving time for now).
- Multi-tenant orgs, billing, or admin dashboards.
- Native iOS/Android apps (we ship a PWA that installs like an app).

## 1.6 Success criteria for "done"

- [ ] A photo of a **tomato leaf with early blight** and a photo of a **healthy tomato leaf** return **different, correct** diagnoses.
- [ ] A photo of a **banana leaf** (or a non-plant) returns a clear "not a supported crop" state, **not** a fake diagnosis.
- [ ] A **voice note** measurably changes the advice (e.g., "spreading fast after rain" raises urgency).
- [ ] End-to-end result returns in **< 4s p95** on the free Groq tier.
- [ ] **No secrets in the repo**; XSS, CORS, and upload-validation issues from the audit are closed.
- [ ] UI is **fully usable on a 360px-wide phone** with a bottom navbar and installs as a PWA.
