# Final-Year-Project-Mohammad-Sarif-Khan-AgriDoctor 🌿

> **An AI-powered assistant that diagnoses crop-leaf diseases from a photo, with
> optional voice input, and returns treatment advice — running a locally-trained
> CNN first, and escalating to a hosted vision model only when it needs to.**

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![License](https://img.shields.io/badge/license-MIT-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688)
![CNN](https://img.shields.io/badge/CNN-EfficientNet--B0%20(local)-success)
![AI](https://img.shields.io/badge/fallback-Groq%20Qwen3.6%20Vision-orange)

---

## ⚡ Quick Start — real AI diagnosis

AgriDoctor performs **genuine image-based diagnosis** with a locally-trained CNN,
falling back to a hosted vision–language model for anything the CNN wasn't
trained on. It looks at the actual leaf pixels, identifies the crop, **rejects
photos that aren't a supported crop (or aren't a leaf at all)**, fuses in the
farmer's voice/text notes, and returns structured treatment advice. There are
**no hard-coded or mock predictions** — every result is generated at run time
from the uploaded image.

> The full engineering plan lives in **[`docs/implementation/`](docs/implementation/)**.

```bash
# 1. Configure secrets (never commit .env)
cp .env.example .env
python -c "import secrets; print('SECRET_KEY=' + secrets.token_urlsafe(48))" >> .env
#   Optional: add a FREE Groq key (https://console.groq.com/keys) -> GROQ_API_KEY=gsk_...
#   Only needed for crops the local CNN doesn't cover — see "Two-tier engine" below.

# 2. Install + run the API
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-ml.txt      # includes torch, needed to run the CNN
uvicorn backend.main:app --reload --port 8000

# 3. Serve the frontend (separate terminal)
cd frontend && python3 -m http.server 3000
#   open http://localhost:3000

# 4. Run the tests (no network / no API key / no model weights needed)
pytest
```

Or start both servers at once:

```bash
chmod +x start.sh && ./start.sh     # backend :8000 + frontend :3000
```

**Getting the model.** Weights are not committed (they're large and rebuildable).
Build the corpus and train, then check what the server sees:

```bash
python scripts/prepare_dataset.py            # downloads + caches ~47k images
python scripts/train_cnn.py                  # writes data/models/agridoctor_cnn.pt
python scripts/evaluate_cnn.py               # held-out metrics + confusion matrix
curl localhost:8000/health                   # reports which engines are live
```

Without a checkpoint the app still runs — every request simply escalates to the
hosted model, or returns an honest "not confident" result if no key is set.

**Key endpoint:** `POST /api/analyze` (multipart: `image`, optional `audio`,
`crop_hint`, `onset_days`, `spread`, `notes`) → returns the full diagnosis in one
call. Result `kind` is one of: `diagnosis`, `healthy`, `unsupported_crop`,
`not_a_leaf`, `low_confidence`.

> **On the free Groq tier:** it allows roughly one image per minute. This is why
> the CNN runs first — a tomato, potato, pepper or maize leaf is answered locally
> in well under a second and never touches the API, so the rate limit only
> applies to images that genuinely need escalating.

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

The diagnosis engine is a **locally-trained convolutional neural network**
(EfficientNet-B0, trained by [`scripts/train_cnn.py`](scripts/train_cnn.py)), with
a **hosted vision–language model** (`qwen/qwen3.6-27b` on Groq) as an escalation
tier. Both sit behind a **server-side safety layer** that is the core
contribution of the project: no model output is trusted blindly.

What makes local-first workable is that the CNN is trained with two explicit
**out-of-scope classes** rather than being forced to pick a disease for every
image. It therefore returns a routing decision, not just a label:

| CNN decision | Meaning | What happens |
| ------------ | ------- | ------------ |
| **diagnosis** | Confident, and a crop it was trained on | Answered locally — no network call |
| **not a plant** | Confidently not vegetation | Rejected locally — no network call |
| **escalate** | Another species, or not confident enough | Handed to the hosted vision model |

Escalation is a designed tier, not a failure path. The taxonomy covers eight
crops and four livestock species, but the public training corpus only contains
tomato, potato, pepper and maize — rice, chili, cucumber, eggplant and all
livestock *cannot* be learned locally and are meant to escalate.

The practical effect: the common case (a tomato or potato leaf) costs **no API
call at all**, which removes the free-tier rate limit from everyday use. Optional
voice input is transcribed with **Whisper** and fused into the analysis.

Locally-diagnosed results also carry a **Grad-CAM heatmap** showing which pixels
drove the prediction, so a diagnosis can be inspected rather than taken on trust.

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
| 🌐 **Multilingual advice**    | Advice returned in the farmer's language (English, Hindi, Nepali, Bengali, Spanish) via a `language` field. |
| 📱 **Responsive PWA**         | Mobile-first, installable Progressive Web App that works on phones, tablets, and desktops.           |
| 🔒 **Optional accounts**      | JWT auth with PBKDF2 hashing; login only gates saved history — anonymous diagnosis works without it. |
| 📊 **Case history**           | Logged-in users can save and revisit past diagnoses.                                                 |

---

## 🌾 Supported Crops

The taxonomy ([`data/taxonomy.json`](data/taxonomy.json)) defines **eight crops
and four livestock species** across fungal, bacterial, viral, pest and nutrient
categories. Which engine answers depends on whether the local CNN was trained on
that subject — the **Engine** column below is the honest split:

| Subject      | Example conditions detected                                              | Engine |
| ------------ | ------------------------------------------------------------------------ | ------ |
| 🍅 Tomato    | Early Blight, Late Blight, Leaf Mold, Septoria, Spider Mites, Mosaic, Bacterial Spot, Target Spot, Yellow Leaf Curl | 🧠 Local CNN |
| 🥔 Potato    | Early Blight, Late Blight, Blackleg, Common Scab, Viral Mosaic, Aphids    | 🧠 Local CNN |
| 🫑 Pepper    | Bacterial Spot, Phytophthora, Anthracnose, Mosaic, Powdery Mildew         | 🧠 Local CNN |
| 🌽 Maize     | Northern Leaf Blight, Common Rust, Gray Leaf Spot, Smut, Stem Borer       | 🧠 Local CNN |
| 🌾 Rice      | Blast, Brown Spot, Bacterial Leaf Blight, Tungro, Sheath Blight, Stem Borer | ☁️ Escalates |
| 🌶️ Chili     | Anthracnose, Bacterial Wilt, Leaf Curl Virus, Powdery Mildew, Thrips      | ☁️ Escalates |
| 🥒 Cucumber  | Powdery Mildew, Downy Mildew, Angular Leaf Spot, Mosaic Virus, Anthracnose | ☁️ Escalates |
| 🍆 Eggplant  | Verticillium Wilt, Fruit Rot, Leaf Spot, Flea Beetle, Mites               | ☁️ Escalates |
| 🐄 Cattle · 🐐 Goat · 🐑 Sheep · 🐔 Poultry | Visible skin/eye/foot conditions (FMD, LSD, mange, orf, footrot, pinkeye…) | ☁️ Escalates |

The split isn't a design preference — it's a **data** constraint. The public
PlantVillage corpus only contains tomato, potato, pepper and maize, so those four
are the only subjects a local model can be trained on from freely available data.
Everything else escalates to the hosted vision model, and if no API key is
configured those requests return an honest "not confident" rather than a guess.

> **Not supported at all:** any other crop or animal. The system rejects these
> rather than guessing — a grape leaf is refused, not relabelled as tomato.

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
  **not trained** on real farmer text (no image+text dataset). These were the
  earlier experiments; the current serving engine is the EfficientNet-B0 CNN from
  `scripts/train_cnn.py`. See the training section below.

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
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│              TIER 1 — LOCAL CNN  (backend/ai/cnn_predictor.py)         │
│   EfficientNet-B0 · 21 classes = 19 in-scope + 2 out-of-scope          │
│   Calibrated confidence (temperature-scaled) → routing decision        │
└──────────────────────────────────────────────────────────────────────┘
         │                        │                         │
   confident,               confidently                other species /
   in-scope                 not a plant                low confidence
         │                        │                         │
         ▼                        ▼                         ▼
┌──────────────────┐   ┌──────────────────┐   ┌───────────────────────────┐
│ Offline advice   │   │ Reject locally   │   │ TIER 2 — hosted VLM        │
│ + Grad-CAM       │   │                  │   │ Groq · qwen/qwen3.6-27b    │
│ NO API CALL      │   │ NO API CALL      │   │ (+ Whisper for voice)      │
└──────────────────┘   └──────────────────┘   └───────────────────────────┘
         │                        │                         │
         └────────────────────────┴─────────────────────────┘
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│               SERVER-SIDE SAFETY-INVARIANT LAYER                       │
│   Every path exits through here — local and hosted alike.              │
│   • subject must be a supported crop or animal                         │
│   • label must exist in the taxonomy AND belong to that subject        │
│   • non-leaf / unsupported / unusable → polite rejection (no diagnosis)│
│   • confidence & severity clamped; hallucinations stripped            │
└──────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     SQLite (users • cases • history)                   │
└──────────────────────────────────────────────────────────────────────┘
```

> Two independent safety mechanisms operate here, and they are deliberately not
> the same thing. The CNN's reject classes let the *model* decline an image it
> shouldn't answer; the invariant layer is a deterministic server-side check that
> catches anything either model gets wrong regardless. Removing either one would
> leave a gap the other does not cover.

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
│   │   ├── cnn_predictor.py    # PRIMARY engine: local CNN + routing decision
│   │   ├── local_advice.py     # curated offline treatment advice (no LLM call)
│   │   └── gradcam.py          # Grad-CAM heatmaps for local diagnoses
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

## 🧠 Model Training (the primary serving engine)

The local CNN is what answers most requests. It is trained end to end from public
data with two reproducible scripts:

```bash
# 1. Build the corpus: ~47k images, 21 classes, stratified train/val/test
#    (auto-downloads via HuggingFace and caches resized JPEGs to disk)
.venv/bin/python scripts/prepare_dataset.py

# 2. Train (Apple-Silicon MPS / CUDA / CPU)
.venv/bin/python scripts/train_cnn.py

# 3. Evaluate on the held-out test split
.venv/bin/python scripts/evaluate_cnn.py
```

**The corpus.** 19 in-scope disease/healthy classes across tomato, potato, pepper
and maize, plus the two reject classes that make local-first safe:

| Class group | Source | Purpose |
| ----------- | ------ | ------- |
| 19 in-scope labels | `minhhungg/plant-disease-dataset` | The diseases actually diagnosed |
| `OOS_OTHER_PLANT` | The same dataset's other 19 species (apple, grape, orange, soybean…) | Real leaves of crops we don't cover → escalate |
| `OOS_NOT_PLANT` | `zh-plus/tiny-imagenet` | Non-vegetation photos → reject locally |

The reject classes are the whole reason the CNN can be trusted as the primary
engine. A plain 19-way softmax has no way to say "this is a car" — it must assign
*some* crop disease to every image. That is why the earlier version of this
project kept the CNN switched off.

**The training recipe** goes beyond a plain fine-tune, specifically to fight the
PlantVillage weakness described below:

- Two-stage schedule — head-only warmup, then full fine-tune on per-batch cosine decay
- MixUp + CutMix, so the network can't bet everything on one dominant cue (the background)
- Resolution jitter on *every* class, so "low-res" can't become a shortcut for the reject class
- Class-balanced sampling, label smoothing, RandAugment, random erasing
- EMA weights, evaluated against the raw weights each epoch — whichever genuinely scores higher is saved
- Temperature scaling fitted on validation, so the confidence the router thresholds is calibrated rather than the usual overconfident softmax

**Measured results** (7,109-image held-out test split, `data/models/eval_report.json`):

| Metric | Value |
| ------ | ----- |
| Accuracy / macro-F1 | **0.9979 / 0.9976** |
| Calibration error (ECE, post-scaling) | **0.0016** |
| False-accept rate (out-of-scope → confident diagnosis) | **0.0006** |
| OOD AUROC (in-scope vs out-of-scope) | **1.00** |
| Median inference latency | **24 ms** |
| Routing mix | 74% local diagnosis · 13% local reject · 13% escalate |

At the default 0.75 threshold, ~87% of requests are resolved locally with no API
call. The threshold sweep shows local coverage stays above 99% anywhere from 0.50
to 0.90, so the operating point is not delicate.

⚠️ **The honest caveat:** PlantVillage images are shot on plain backgrounds under
even lighting, so 0.9979 is in-distribution accuracy and overstates field
performance — the well-documented domain-shift problem, which no architecture
change fixes. `scripts/evaluate_cnn.py` therefore also runs an 8-corruption
robustness suite; measured mean accuracy under corruption is **0.9825** (worst
case, heavy blur, **0.9223**). That clean-vs-corrupted gap is the honest measure
of what a real phone photo costs, and it is precisely why the hosted model is
retained as the escalation tier.

### Multimodal fusion ablation (real)

```bash
# Frozen ViT-B/16 + DistilBERT -> attention fusion; runs image/text/fusion ablation
.venv/bin/python scripts/train_fusion_real.py --per-class 120 --epochs 30
```

**Measured** (17 classes, `data/models/fusion_ablation.json`): image-only **0.907**,
text-only **0.778**, **fusion 0.990** macro-F1 → fusion **+8.3 pts** over the best
single modality. ⚠️ *Caveat:* the text is templated symptom descriptions (not real
farmer input) and encoders are frozen — it demonstrates the pipeline, not a
real-world claim. The full trainable-encoder version (`src/models/train_multimodal.py`)
remains a documented design:

```bash
# Full version — requires an image+text dataset you provide
python src/models/train_multimodal.py \
    --labels data/labels.csv --images data/images --epochs 30

# 4. annotate data
streamlit run tools/annotator_app.py
```

Both of the above are earlier offline experiments, kept for the record. The model
the server actually loads is `data/models/agridoctor_cnn.pt` from
`scripts/train_cnn.py` — see the training section above.

---

## ⚙️ Configuration

Set via a `.env` file in the project root (never commit it — it's gitignored).

| Variable | Default | Description |
| --- | --- | --- |
| `SECRET_KEY` | *(required)* | JWT signing secret (app refuses to start without it) |
| `USE_LOCAL_CNN` | `True` | Run the local CNN first. If no checkpoint exists it reports unavailable and everything escalates |
| `CNN_MODEL_PATH` | `data/models/agridoctor_cnn.pt` | Trained checkpoint from `scripts/train_cnn.py` |
| `CNN_CONFIDENCE_THRESHOLD` | `0.75` | **Calibrated** confidence needed to answer locally. Raise it to escalate more borderline cases; lower it to answer more offline |
| `CNN_MARGIN_THRESHOLD` | `0.10` | Minimum top-1 vs top-2 gap. Catches near-ties that a confidence bar alone misses |
| `CNN_REJECT_THRESHOLD` | `0.60` | Confidence needed to reject an image locally as "not a plant" |
| `CNN_USE_TTA` | `True` | Average over the image and its mirror (~2x inference, still well under 100ms) |
| `ENABLE_GRADCAM` | `True` | Attach a Grad-CAM heatmap to locally-diagnosed results |
| `GROQ_API_KEY` | *(optional)* | Free key from console.groq.com. Only used for escalated images; blank runs fully offline |
| `GROQ_VISION_MODEL` | `qwen/qwen3.6-27b` | Groq vision model (auto-resolves to an available one if this ID is retired) |
| `GROQ_AUDIO_MODEL` | `whisper-large-v3-turbo` | Groq ASR model for voice notes |
| `GROQ_TEXT_MODEL` | `llama-3.3-70b-versatile` | Text model (hosted path only — local diagnoses use the offline advice base) |
| `AI_TIMEOUT_SECONDS` | `30` | Timeout for AI calls |
| `AI_MAX_RETRIES` | `2` | Retries (also used for rate-limit backoff) |
| `ALLOWED_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` | CORS allow-list |
| `LOG_LEVEL` | `INFO` | Logging level |

> The two CNN thresholds encode a deliberate trade-off: a higher bar means fewer
> wrong local answers but more escalations (and more API calls). `scripts/evaluate_cnn.py`
> emits a threshold sweep measuring exactly that curve on held-out data, so the
> shipped value is an evidenced choice rather than a guess.
>
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
