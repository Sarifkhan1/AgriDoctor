# AgriDoctor AI — Implementation Documentation

> **Goal:** Transform AgriDoctor from a hardcoded demo into a genuinely functional AI crop-disease platform that *actually looks at the leaf*, detects unsupported crops, fuses voice + text, and returns detailed, instant treatment advice — behind a modern, fully responsive, app-like UI.

This folder is the **single source of truth** for the rebuild. Read the documents in order.

---

## The one-paragraph summary

Today the "AI" is [`generate_mock_prediction()`](../../backend/main.py) — a hardcoded lookup keyed on the crop **button the user clicks**. The uploaded image is saved to disk but never analyzed, and the entire `src/` ML stack is never called at serving time. We replace this with a real **vision-language model on Groq** (open-source **Llama 4 Scout** for image understanding + **Whisper-large-v3** for voice) that inspects the actual leaf pixels, identifies the crop, rejects images that are **not one of the 6 supported crops** (or not a leaf at all), diagnoses the disease against our 38-label taxonomy with reasoning, scores severity/confidence, fuses in the farmer's voice/text description, and returns a strict JSON diagnosis in ~1–3 seconds.

---

## Documents

| # | Document | What it covers |
|---|----------|----------------|
| 01 | [Overview & Decisions](./01-OVERVIEW-AND-DECISIONS.md) | Problem statement, target product, key decisions (Groq, instant flow), non-goals |
| 02 | [Architecture](./02-ARCHITECTURE.md) | System diagram, request flow, components, data model changes, `src/` positioning |
| 03 | [AI Engine Spec](./03-AI-ENGINE-SPEC.md) | **The core.** Groq integration, provider interface, vision system prompt, OOD/leaf detection, voice transcription, strict JSON output contract, validation & fallbacks |
| 04 | [Backend & API](./04-BACKEND-API.md) | New `/api/analyze` instant endpoint, schema migration, persistence, rate limiting, config |
| 05 | [Frontend Redesign](./05-FRONTEND-REDESIGN.md) | App-style bottom navbar, responsive layout, new result states, XSS-safe rendering, PWA |
| 06 | [Security Hardening](./06-SECURITY-HARDENING.md) | Fixes for every critical/major issue from the audit (secrets, XSS, CORS, uploads) |
| 07 | [Roadmap & Tasks](./07-ROADMAP-AND-TASKS.md) | Phased plan, task checklists, acceptance criteria, test plan, effort estimates |

---

## Chosen stack (decided)

| Concern | Choice | Why |
|---------|--------|-----|
| Vision AI | **Groq — `meta-llama/llama-4-scout-17b-16e-instruct`** | Open-source weights, fastest inference available, free tier, strong multimodal reasoning |
| Voice AI | **Groq — `whisper-large-v3-turbo`** | Open-source, sub-second transcription |
| Advice refinement (optional) | **Groq — `llama-3.3-70b-versatile`** | Fast text model if we split diagnosis from advice generation |
| Backend | FastAPI (existing) + new `backend/ai/` module | Keep what works; isolate the AI engine behind an interface |
| DB | SQLite (existing) for MVP | Fine for current scale; documented migration path to Postgres |
| Frontend | Vanilla JS (existing) rebuilt as a responsive PWA | No framework rewrite needed; fix XSS + redesign |

> ⚠️ **Model IDs change.** Verify the current Groq model list at <https://console.groq.com/docs/models> before implementing. The IDs above are the intended targets; `03-AI-ENGINE-SPEC.md` defines them in one config constant so a change is a one-line edit.

---

## Prerequisites before coding

1. A **free Groq API key** from <https://console.groq.com> → stored in `.env` as `GROQ_API_KEY` (never committed).
2. Complete **Phase 0 (Security Hardening)** first — it removes committed secrets that currently block safe development. See [06-SECURITY-HARDENING.md](./06-SECURITY-HARDENING.md).
