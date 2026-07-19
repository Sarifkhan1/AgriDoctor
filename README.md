# AI-4-Creativity-Project-Mohammad-Sarif-Khan-AgriDoctor 🌿

> **An AI-powered assistant that diagnoses crop-leaf diseases from a photo, with
> optional voice input, and returns treatment advice — for six crops.**

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![License](https://img.shields.io/badge/license-MIT-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688)
![AI](https://img.shields.io/badge/AI-Groq%20Qwen3.6%20Vision-orange)

---

## ⚡ Quick Start — real AI diagnosis

AgriDoctor performs **genuine image-based diagnosis** using a hosted
vision–language model on **Groq**. It looks at the actual leaf pixels, identifies
the crop, **rejects photos that aren't one of the 6 supported crops (or aren't a
leaf at all)**, fuses in the farmer's voice/text notes, and returns instant,
structured treatment advice. There are **no hard-coded or mock predictions** —
every result is generated at run time from the uploaded image.

> The full engineering plan lives in **[`docs/implementation/`](docs/implementation/)**.

```bash
# 1. Configure secrets (never commit .env)
cp .env.example .env
python -c "import secrets; print('SECRET_KEY=' + secrets.token_urlsafe(48))" >> .env
#   then add your FREE Groq key from https://console.groq.com/keys  ->  GROQ_API_KEY=gsk_...

# 2. Install + run the API
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000

# 3. Serve the frontend (separate terminal)
cd frontend && python3 -m http.server 3000
#   open http://localhost:3000

# 4. Run the tests (no network / no API key needed — uses a fake provider)
pytest
```

Or start both servers at once:

```bash
chmod +x start.sh && ./start.sh     # backend :8000 + frontend :3000
```

**Key endpoint:** `POST /api/analyze` (multipart: `image`, optional `audio`,
`crop_hint`, `onset_days`, `spread`, `notes`) → returns the full diagnosis in one
call. Result `kind` is one of: `diagnosis`, `healthy`, `unsupported_crop`,
`not_a_leaf`, `low_confidence`.

> **Note on the free Groq tier:** the first analysis after startup can take
> 5–40 seconds, and the free tier allows roughly one image per minute. If you send
> images faster you'll get a friendly "AI is busy, try again shortly" message —
> that's the rate limit, not a bug.

---

## 📋 Student Information

| Field              | Details                                                                                                                |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| **Student Name**   | Mohammad Sarif Khan                                                                                                    |
| **Student Number** | 2238572                                                                                                                |
| **Project**        | AI-4-Creativity Project                                                                                                |
| **Repository**     | [GitHub - Sarifkhan1/AgriDoctor](https://github.com/Sarifkhan1/AI-4-Creativity-Project-Mohammad-Sarif-Khan-AgriDoctor) |

---

## 🎥 Project Video Recording

📺 **https://youtu.be/SOlaeAC4mLw**

---

## 📖 Table of Contents

- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [Supported Crops](#-supported-crops)
- [Technology Stack](#-technology-stack)
- [System Architecture](#-system-architecture)
- [Setup Instructions](#-setup-instructions)
- [Running the Application](#-running-the-application)
- [Project Structure](#-project-structure)
- [API Documentation](#-api-documentation)
- [Experimental Model Training (not used at serving time)](#-experimental-model-training-not-used-at-serving-time)
- [Configuration](#-configuration)
- [Testing](#-testing)
- [License & Disclaimer](#-license--disclaimer)

---

## 🌟 Project Overview

**AgriDoctor** is a full-stack web application that helps farmers diagnose
crop-leaf diseases from a photograph. A user uploads or captures an image of an
affected leaf, optionally adds a spoken or typed description of the symptoms, and
receives a structured diagnosis: the identified crop, the most likely disease, a
confidence and severity estimate, and an actionable treatment and prevention plan.

The diagnosis engine is a **hosted vision–language model** (`qwen/qwen3.6-27b`,
served on Groq for low latency), wrapped in a **server-side safety layer** that is
the core contribution of the project: the model's output is never trusted blindly.
The server independently enforces that a diagnosis is only produced for a
**supported crop** with a valid disease label, and converts anything else (a
non-leaf photo, an unsupported crop, an unusable image) into a polite, honest
rejection rather than a fabricated result. Optional voice input is transcribed
with **Whisper** and fused into the analysis.

**Design principle:** honesty about capability. When the system does not know, it
says so — it will refuse a photo of a car instead of diagnosing it.

---

## ✨ Key Features

| Feature                       | Description                                                                                          |
| ----------------------------- | --------------------------------------------------------------------------------------------------- |
| 🖼️ **Real image diagnosis**   | Every uploaded leaf photo is analysed at run time by a vision–language model — no mock predictions.  |
| 🚫 **Out-of-scope rejection** | Non-leaf images, unsupported crops, and unusable photos are detected and refused, not force-fit.     |
| 🎤 **Voice input**            | Describe symptoms with a voice note; it is transcribed (Whisper) and fused into the diagnosis.       |
| 💡 **Actionable advice**      | Structured treatment plan: immediate actions, long-term prevention, and when to escalate.            |
| 📱 **Responsive PWA**         | Mobile-first, installable Progressive Web App that works on phones, tablets, and desktops.           |
| 🔒 **Optional accounts**      | JWT auth with PBKDF2 hashing; login only gates saved history — anonymous diagnosis works without it. |
| 📊 **Case history**           | Logged-in users can save and revisit past diagnoses.                                                 |

---

## 🌾 Supported Crops

AgriDoctor diagnoses **six crops** (see [`data/taxonomy.json`](data/taxonomy.json)
for the full label set across fungal, bacterial, viral, pest, and nutrient
categories):

| Crop        | Example conditions detected                                              |
| ----------- | ----------------------------------------------------------------------- |
| 🍅 Tomato   | Early Blight, Late Blight, Leaf Mold, Septoria, Spider Mites, Mosaic, Bacterial Spot |
| 🥔 Potato   | Early Blight, Late Blight, Blackleg, Common Scab, Viral Mosaic, Aphids   |
| 🌾 Rice     | Blast, Brown Spot, Bacterial Leaf Blight, Tungro, Sheath Blight, Stem Borer |
| 🌽 Maize    | Northern Leaf Blight, Common Rust, Gray Leaf Spot, Smut, Stem Borer      |
| 🌶️ Chili    | Anthracnose, Bacterial Wilt, Leaf Curl Virus, Powdery Mildew, Thrips     |
| 🥒 Cucumber | Powdery Mildew, Downy Mildew, Angular Leaf Spot, Mosaic Virus, Anthracnose, Aphids |

> **Not supported (yet):** livestock diagnosis and other crops. These are listed
> as future work only — the application does not claim to diagnose them, and it
> explicitly rejects unsupported crops rather than guessing.

---

## 🛠️ Technology Stack

### Serving stack (what actually runs the product)

- **FastAPI** (Python 3.11+) — backend API (`POST /api/analyze` + auth/history)
- **Groq** hosted models — `qwen/qwen3.6-27b` (vision), `whisper-large-v3-turbo`
  (voice), `llama-3.3-70b-versatile` (text advice for the optional local-CNN path)
- **SQLite** — users, cases, and diagnosis history (with automatic schema migration)
- **JWT + PBKDF2** — authentication and password hashing
- **Pillow** — image validation/decoding
- **Vanilla HTML/CSS/JavaScript** — a dependency-free, installable PWA front end

The serving API is deliberately lightweight and does **not** require PyTorch.

### Experimental / offline stack (in `src/`, not used at serving time)

- **PyTorch / TorchVision** — used for an optional, offline-trained image
  classifier. A compact MobileNetV3 model **was actually trained** (`scripts/train_real.py`)
  on 17 PlantVillage-derived classes across Tomato/Potato/Pepper/Maize, reaching
  **99.3% held-out accuracy** (see `data/models/metrics.json`; note the plain-background
  caveat). The **multimodal ViT + DistilBERT fusion** model is also implemented but
  **not trained** (no image+text dataset). None of these are the default serving
  engine — the hosted VLM is. See the training section below.

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                           CLIENT (PWA)                                 │
│   Frontend (HTML/CSS/JS): image upload/camera • voice • results        │
└──────────────────────────────────────────────────────────────────────┘
                                  │  POST /api/analyze (multipart)
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        FastAPI BACKEND                                 │
│   Upload validation (byte-signature + Pillow) • optional JWT auth      │
│   • rate limiting • persistence                                        │
└──────────────────────────────────────────────────────────────────────┘
                                  │
             ┌────────────────────┴───────────────────┐
             ▼                                         ▼
┌──────────────────────────┐              ┌───────────────────────────┐
│  Whisper (voice → text)  │              │  Vision–Language model     │
│  transcript fused into   │─────────────▶│  Groq · qwen/qwen3.6-27b   │
│  the analysis context    │              │  → strict JSON diagnosis   │
└──────────────────────────┘              └───────────────────────────┘
                                                        │
                                                        ▼
┌──────────────────────────────────────────────────────────────────────┐
│               SERVER-SIDE SAFETY-INVARIANT LAYER                       │
│   • crop must be one of the 6 supported crops                          │
│   • label must exist in the taxonomy                                   │
│   • non-leaf / unsupported / unusable → polite rejection (no diagnosis)│
│   • confidence & severity clamped; hallucinations stripped            │
└──────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     SQLite (users • cases • history)                   │
└──────────────────────────────────────────────────────────────────────┘
```

> An optional local-CNN path (`backend/ai/cnn_predictor.py`) is wired in so that
> *if* a trained checkpoint were provided it would run first and fall back to the
> hosted model. No checkpoint is shipped, so this path is inactive and the hosted
> model handles every request.

---

## 🚀 Setup Instructions

### Prerequisites

- **Python 3.11 or higher** — [Download Python](https://www.python.org/downloads/)
- **Git** — [Download Git](https://git-scm.com/downloads)
- A **free Groq API key** — [console.groq.com/keys](https://console.groq.com/keys)
- **Docker & Docker Compose** *(optional, for containerized deployment)*

### Option 1: Virtual Environment (recommended)

```bash
git clone https://github.com/Sarifkhan1/AI-4-Creativity-Project-Mohammad-Sarif-Khan-AgriDoctor.git
cd AI-4-Creativity-Project-Mohammad-Sarif-Khan-AgriDoctor

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt      # slim serving deps (no PyTorch)

cp .env.example .env                 # then edit .env: SECRET_KEY + GROQ_API_KEY
```

> The heavy ML libraries (torch/torchvision/transformers) are only needed for the
> experimental training code in `src/`. Install those separately with
> `pip install -r requirements-ml.txt` — the serving API does not need them.

### Option 2: Docker

```bash
docker compose up -d          # build + start
docker compose logs -f        # view logs
docker compose down           # stop
```

Access: Frontend `http://localhost:3000` · Backend `http://localhost:8000` ·
API docs `http://localhost:8000/docs`.

---

## 🏃 Running the Application

### Quick start (both servers)

```bash
chmod +x start.sh
./start.sh
```

### Manual start

```bash
# Terminal 1 — backend
source .venv/bin/activate
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 — frontend
cd frontend && python -m http.server 3000
```

| Service                | URL                         | Description                   |
| ---------------------- | --------------------------- | ----------------------------- |
| **Frontend**           | http://localhost:3000       | Main web application          |
| **Backend API**        | http://localhost:8000       | REST API endpoints            |
| **API Docs (Swagger)** | http://localhost:8000/docs  | Interactive API documentation |

---

## 📁 Project Structure

```
AgriDoctor/
├── backend/
│   ├── main.py                 # FastAPI app factory, middleware, rate limiting
│   ├── config.py               # env-driven settings (fail-fast on missing secrets)
│   ├── db.py                   # SQLite layer + idempotent schema migration
│   ├── security.py             # JWT auth, password hashing
│   ├── ai/
│   │   ├── provider.py         # AIProvider interface + Groq impl (model auto-resolve)
│   │   ├── analyzer.py         # orchestration + server-side safety invariants
│   │   ├── prompts.py          # constrained system/user prompts
│   │   ├── schemas.py          # AnalysisResult contract
│   │   ├── taxonomy.py         # supported crops + labels (from data/taxonomy.json)
│   │   └── cnn_predictor.py    # optional local-CNN path (inactive without weights)
│   ├── routers/                # analyze, auth, cases, media
│   └── services/               # analysis pipeline, upload validation, NLU hints
│
├── frontend/                   # PWA: index.html, css/, js/ (api.js, app.js, ui.js), sw.js
│
├── src/                        # EXPERIMENTAL training code (not served) — see below
│   ├── models/                 # train_image_model.py (ViT/Swin), train_multimodal.py
│   └── preprocessing/          # image, ASR, and NLU preprocessing
│
├── tools/annotator_app.py      # Streamlit annotation tool
├── data/taxonomy.json          # the 6-crop disease taxonomy
├── tests/                      # pytest suite (uses a fake provider — no network)
├── docs/implementation/        # full engineering plan (01–07)
├── requirements.txt            # slim serving deps
├── requirements-ml.txt         # heavy deps for src/ training only
├── start.sh · Dockerfile · docker-compose.yml · nginx.conf
└── README.md
```

---

## 📡 API Documentation

### Primary endpoint

| Method | Endpoint       | Description                                            | Auth       |
| ------ | -------------- | ----------------------------------------------------- | ---------- |
| POST   | `/api/analyze` | Analyze a leaf image (+ optional voice/context) → full diagnosis in one call | Optional\* |

\* Anonymous works; if authenticated, the case is saved to history.

```bash
curl -X POST "http://localhost:8000/api/analyze" \
  -F "image=@leaf.jpg" \
  -F "crop_hint=tomato" \
  -F "notes=spots started after heavy rain"
```

### Auth & history endpoints

| Method | Endpoint             | Description           | Auth |
| ------ | -------------------- | --------------------- | ---- |
| POST   | `/api/auth/register` | Register new user     | ❌   |
| POST   | `/api/auth/login`    | Login and get JWT     | ❌   |
| GET    | `/api/auth/me`       | Current user info     | ✅   |
| GET    | `/api/cases`         | List saved diagnoses  | ✅   |
| GET    | `/api/cases/{id}`    | Get a saved case      | ✅   |

---

## 🧠 Model Training (optional — not the default serving engine)

The deployed engine is the hosted vision–language model. Separately, the repo
includes a **real, reproducible training run** for an optional local classifier:

```bash
# Train a compact classifier on a public PlantVillage-derived dataset
# (auto-downloads via HuggingFace; runs on Apple-Silicon MPS / CPU / CUDA)
.venv/bin/python scripts/train_real.py --per-class 300 --epochs 8
```

**Measured result:** MobileNetV3-small over 17 classes across Tomato/Potato/Pepper/
Maize (4,080 train / 1,020 val) → **99.3% validation accuracy, 0.9931 macro-F1**
(`data/models/metrics.json`). ⚠️ *Honest caveat:* PlantVillage images have plain
backgrounds, so this is **not** field accuracy — messy real-world photos would score
much lower. This is a genuine pipeline demonstration, not a claim that it beats the
VLM. Once trained, `data/models/best_model.pt` is used as an optional local fast-path
(config `USE_LOCAL_CNN`) for the crop classes it covers, with the hosted model as
fallback.

The larger **multimodal fusion** model (`src/models/train_multimodal.py`, ViT +
DistilBERT + cross-attention) is implemented but **not trained** (no image+text
dataset assembled) — it remains a documented design, not a result:

```bash
# Experimental only — requires an image+text dataset you provide
python src/models/train_multimodal.py \
    --labels data/labels.csv --images data/images --epochs 30

# 4. annotate data
streamlit run tools/annotator_app.py
```

If a checkpoint is produced at `data/models/best_model.pt` and `USE_LOCAL_CNN=True`,
the serving analyzer will use it first and fall back to the hosted model on low
confidence.

---

## ⚙️ Configuration

Set via a `.env` file in the project root (never commit it — it's gitignored).

| Variable | Default | Description |
| --- | --- | --- |
| `SECRET_KEY` | *(required)* | JWT signing secret (app refuses to start without it) |
| `GROQ_API_KEY` | *(required)* | Free key from console.groq.com |
| `GROQ_VISION_MODEL` | `qwen/qwen3.6-27b` | Groq vision model (auto-resolves to an available one if this ID is retired) |
| `GROQ_AUDIO_MODEL` | `whisper-large-v3-turbo` | Groq ASR model |
| `GROQ_TEXT_MODEL` | `llama-3.3-70b-versatile` | Text model for advice on the local-CNN path |
| `AI_TIMEOUT_SECONDS` | `30` | Timeout for AI calls |
| `AI_MAX_RETRIES` | `2` | Retries (also used for rate-limit backoff) |
| `USE_LOCAL_CNN` | `True` | Enable the optional local-CNN path (inactive unless a checkpoint exists) |
| `CNN_MODEL_PATH` | `data/models/best_model.pt` | Path to a trained checkpoint, if any |
| `CNN_CONFIDENCE_THRESHOLD` | `0.80` | Confidence needed to skip the hosted vision call |
| `ALLOWED_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` | CORS allow-list |
| `LOG_LEVEL` | `INFO` | Logging level |

> The vision model default is chosen to match Groq's currently available models.
> Groq periodically retires model IDs; the provider **auto-resolves** a working
> vision model from a fallback list against your account's live model list, so a
> retired default won't silently break diagnosis.

---

## 🧪 Testing

```bash
pytest                                   # full suite (fake provider, no network/key)
pytest --cov=backend --cov-report=html   # with coverage
```

---

## 📄 License & Disclaimer

Licensed under the **MIT License** — see [LICENSE](LICENSE).

> ⚠️ **Disclaimer:** AgriDoctor provides AI-generated guidance and is **not** a
> substitute for professional agronomic advice. It is a credible first opinion,
> not a validated diagnostic instrument; its accuracy has not been measured
> against a labelled test set. Always confirm a diagnosis and any chemical
> treatment with a local agricultural expert before acting.

---

## 👨‍💻 Author

**Mohammad Sarif Khan** · Student Number 2238572 · AI-4-Creativity Project

## 🙏 Acknowledgements

- FastAPI, Groq, OpenAI Whisper, Hugging Face, and PyTorch teams
- The FAO and the PlantVillage dataset for domain context

<div align="center">

**Made with 💚 for farmers** · 🌱 _Healthy Crops, Healthy Future_ 🌱

</div>
