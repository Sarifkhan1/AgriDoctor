# AI-4-Creativity-Project-Mohammad-Sarif-Khan-AgriDoctor 🌿🐄

> **An AI-Powered Multimodal Agricultural Health Assistant for Diagnosing Crop Diseases and Livestock Health Issues**

![Version](https://img.shields.io/badge/version-2.0.0-green)
![Python](https://img.shields.io/badge/Python-3.11+-blue)
![License](https://img.shields.io/badge/license-MIT-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688)
![AI](https://img.shields.io/badge/AI-Groq%20Llama%204%20Scout-orange)

---

## ⚡ v2.0 — Real AI diagnosis (Quick Start)

AgriDoctor now performs **genuine image-based diagnosis** with an open-source
vision-language model on **Groq** (Llama 4 Scout for vision, Whisper for voice).
It looks at the actual leaf pixels, identifies the crop, **rejects photos that
aren't one of the 6 supported crops** (or aren't a leaf), fuses in the farmer's
voice/text, and returns detailed, instant treatment advice.

> The full engineering plan lives in **[`docs/implementation/`](docs/implementation/)**.

```bash
# 1. Configure secrets (never commit .env)
cp .env.example .env
python -c "import secrets; print('SECRET_KEY=' + secrets.token_urlsafe(48))" >> .env
#   then add your free Groq key from https://console.groq.com  ->  GROQ_API_KEY=...

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

**Key endpoint:** `POST /api/analyze` (multipart: `image`, optional `audio`,
`crop_hint`, `onset_days`, `spread`, `notes`) → returns the full diagnosis in one
call. Result `kind` is one of: `diagnosis`, `healthy`, `unsupported_crop`,
`not_a_leaf`, `low_confidence`.

> **Heavy ML libs** (torch/whisper/opencv) are only needed for the offline
> training code in `src/` — install those with `pip install -r requirements-ml.txt`.
> The serving API does not need them.

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
- [Supported Crops & Livestock](#-supported-crops--livestock)
- [Technology Stack](#-technology-stack)
- [System Architecture](#-system-architecture)
- [Setup Instructions](#-setup-instructions)
  - [Prerequisites](#prerequisites)
  - [Option 1: Conda Environment Setup](#option-1-conda-environment-setup-recommended)
  - [Option 2: Docker Setup](#option-2-docker-setup)
  - [Option 3: Virtual Environment Setup](#option-3-virtual-environment-setup)
- [Running the Application](#-running-the-application)
- [Project Structure](#-project-structure)
- [API Documentation](#-api-documentation)
- [Model Training](#-model-training)
- [Screenshots](#-screenshots)
- [License & Disclaimer](#-license--disclaimer)

---

## 🌟 Project Overview

**AgriDoctor AI** is a comprehensive multimodal artificial intelligence system designed to assist farmers and agricultural professionals in diagnosing crop diseases and livestock health issues. The application leverages cutting-edge AI technologies including:

- **Computer Vision** - Using Vision Transformers (ViT) and Swin Transformers for image classification
- **Automatic Speech Recognition (ASR)** - Using OpenAI Whisper for voice-to-text conversion
- **Natural Language Understanding (NLU)** - For extracting symptoms and entities from text descriptions
- **Multimodal Fusion** - Combining visual and textual features for more accurate diagnosis

The system provides actionable advice including treatment recommendations, prevention strategies, and urgency levels based on the diagnosed condition.

---

## ✨ Key Features

| Feature                       | Description                                                                                        |
| ----------------------------- | -------------------------------------------------------------------------------------------------- |
| 🖼️ **Image Analysis**         | Upload photos of affected crops/animals for AI-powered disease detection                           |
| 🎤 **Voice Input**            | Describe symptoms using voice notes (supports multiple languages including English, Hindi, Nepali) |
| 🤖 **Multimodal Fusion**      | Advanced AI that combines visual and textual data for superior diagnostic accuracy                 |
| 💡 **Actionable Advice**      | Get detailed treatment recommendations, prevention tips, and urgency levels                        |
| 📱 **Responsive Design**      | Mobile-first design that works seamlessly on phones, tablets, and desktops                         |
| 🔒 **Secure Authentication**  | JWT-based authentication with secure password hashing                                              |
| 📊 **Case Management**        | Track and manage multiple diagnosis cases with history                                             |
| 🌐 **Multi-language Support** | Voice recognition in multiple languages for accessibility                                          |

---

## 🌾 Supported Crops & Livestock

### Crops

| Crop        | Diseases Detected                                             |
| ----------- | ------------------------------------------------------------- |
| 🍅 Tomato   | Early Blight, Late Blight, Leaf Mold, Septoria, Mosaic Virus  |
| 🥔 Potato   | Early Blight, Late Blight, Blackleg, Scab, Viral Diseases     |
| 🌾 Rice     | Blast, Brown Spot, Bacterial Blight, Tungro, Sheath Blight    |
| 🌽 Maize    | Northern Leaf Blight, Rust, Gray Leaf Spot, Smut              |
| 🌶️ Chili    | Anthracnose, Bacterial Wilt, Leaf Curl, Powdery Mildew        |
| 🥒 Cucumber | Powdery Mildew, Downy Mildew, Angular Leaf Spot, Mosaic Virus |
| 🫑 Pepper   | Bacterial Spot, Phytophthora Blight, Viral Diseases           |
| 🍆 Eggplant | Verticillium Wilt, Fruit Rot, Leaf Spot                       |

### Livestock

| Animal     | Health Issues Detected                                      |
| ---------- | ----------------------------------------------------------- |
| 🐄 Cattle  | Foot and Mouth Disease, Mastitis, Respiratory Issues        |
| 🐐 Goat    | Parasitic Infections, Respiratory Diseases, Skin Conditions |
| 🐑 Sheep   | Footrot, Parasites, Respiratory Infections                  |
| 🐔 Poultry | Newcastle Disease, Avian Influenza, Parasitic Infections    |

---

## 🛠️ Technology Stack

### Backend

- **Framework**: FastAPI (Python 3.11+)
- **Database**: SQLite (lightweight, no external setup required)
- **Authentication**: JWT with PBKDF2 password hashing
- **ASGI Server**: Uvicorn

### Frontend

- **HTML5/CSS3/JavaScript** - Modern responsive UI
- **No external framework dependencies** - Vanilla JS for maximum compatibility

### AI/ML Components

- **PyTorch** - Deep learning framework
- **TorchVision** - Computer vision models (ViT, Swin Transformer)
- **Transformers** - Hugging Face transformers for NLP
- **OpenAI Whisper** - Speech-to-text transcription
- **Scikit-learn** - Machine learning utilities

### Image & Audio Processing

- **Pillow** - Image processing
- **OpenCV** - Computer vision operations
- **Librosa** - Audio analysis
- **SoundFile** - Audio file handling

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                                │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │           Frontend (HTML/CSS/JavaScript)                     │   │
│  │    • Image Upload  • Voice Recording  • Results Display      │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         API LAYER                                   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │               FastAPI Backend (main.py)                      │   │
│  │    • Authentication  • Case Management  • Media Upload       │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                 │
            ┌────────────────────┼────────────────────┐
            ▼                    ▼                    ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│   Image Model    │  │    ASR Model     │  │    NLU Module    │
│   (ViT/Swin)     │  │   (Whisper)      │  │  (Entity Ext.)   │
└──────────────────┘  └──────────────────┘  └──────────────────┘
            │                    │                    │
            └────────────────────┼────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    MULTIMODAL FUSION LAYER                          │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │         Fusion Transformer (Cross-Attention)                 │   │
│  │    • Combines Image + Text Features  • Final Classification  │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       DATA LAYER                                    │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              SQLite Database (agridoctor.db)                 │   │
│  │    • Users  • Cases  • Media  • Results  • History           │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Setup Instructions

### Prerequisites

Before setting up the project, ensure you have the following installed:

- **Python 3.11 or higher** - [Download Python](https://www.python.org/downloads/)
- **Git** - [Download Git](https://git-scm.com/downloads)
- **Conda (Anaconda or Miniconda)** - [Download Miniconda](https://docs.conda.io/en/latest/miniconda.html) _(Recommended)_
- **Docker & Docker Compose** _(Optional, for containerized deployment)_

### Option 1: Conda Environment Setup (Recommended)

This is the recommended setup method for the best compatibility with ML dependencies.

#### Step 1: Clone the Repository

```bash
git clone https://github.com/Sarifkhan1/AI-4-Creativity-Project-Mohammad-Sarif-Khan-AgriDoctor.git
cd AI-4-Creativity-Project-Mohammad-Sarif-Khan-AgriDoctor
```

#### Step 2: Create Conda Environment

```bash
# Create a new conda environment with Python 3.11
conda create -n agridoctor python=3.11 -y

# Activate the environment
conda activate agridoctor
```

#### Step 3: Install PyTorch with CUDA Support (Optional, for GPU acceleration)

For **GPU Support** (if you have an NVIDIA GPU):

```bash
# For CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# OR for CUDA 12.1
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

For **CPU Only**:

```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
```

#### Step 4: Install Project Dependencies

```bash
# Install all required Python packages
pip install -r requirements.txt
```

#### Step 5: Install Additional Audio Dependencies

```bash
# Install ffmpeg for audio processing (required by Whisper)
conda install ffmpeg -c conda-forge -y
```

#### Step 6: Verify Installation

```bash
# Run verification script
python -c "import torch; import fastapi; import whisper; print('✅ All dependencies installed successfully!')"
```

---

### Option 2: Docker Setup

This is the easiest method if you have Docker installed.

#### Step 1: Clone the Repository

```bash
git clone https://github.com/Sarifkhan1/AI-4-Creativity-Project-Mohammad-Sarif-Khan-AgriDoctor.git
cd AI-4-Creativity-Project-Mohammad-Sarif-Khan-AgriDoctor
```

#### Step 2: Build and Run with Docker Compose

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

#### Step 3: Access the Application

- **Frontend**: http://localhost:80 or http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

---

### Option 3: Virtual Environment Setup

If you prefer not to use Conda, you can use Python's built-in venv.

#### Step 1: Clone the Repository

```bash
git clone https://github.com/Sarifkhan1/AI-4-Creativity-Project-Mohammad-Sarif-Khan-AgriDoctor.git
cd AI-4-Creativity-Project-Mohammad-Sarif-Khan-AgriDoctor
```

#### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate on macOS/Linux
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

#### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt
```

#### Step 4: Install System Dependencies

For **macOS**:

```bash
brew install ffmpeg
```

For **Ubuntu/Debian**:

```bash
sudo apt-get update
sudo apt-get install ffmpeg -y
```

For **Windows**:

- Download FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html)
- Add to system PATH

---

## 🏃 Running the Application

### Quick Start Script

The easiest way to run the application:

```bash
# Make the script executable (macOS/Linux)
chmod +x start.sh

# Run the application
./start.sh
```

### Manual Start

#### Terminal 1: Start Backend Server

```bash
# Activate your environment (conda or venv)
conda activate agridoctor  # or source venv/bin/activate

# Navigate to backend directory
cd backend

# Start the FastAPI server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### Terminal 2: Start Frontend Server

```bash
# Navigate to frontend directory
cd frontend

# Start simple HTTP server
python -m http.server 3000
```

### Access Points

| Service                | URL                         | Description                   |
| ---------------------- | --------------------------- | ----------------------------- |
| **Frontend**           | http://localhost:3000       | Main web application          |
| **Backend API**        | http://localhost:8000       | REST API endpoints            |
| **API Docs (Swagger)** | http://localhost:8000/docs  | Interactive API documentation |
| **API Docs (ReDoc)**   | http://localhost:8000/redoc | Alternative API documentation |

---

## 📁 Project Structure

```
AgriDoctor/
├── 📂 backend/
│   └── main.py                    # FastAPI application with all endpoints
│
├── 📂 frontend/
│   ├── index.html                 # Main HTML file
│   ├── 📂 css/
│   │   └── styles.css             # Application styling
│   └── 📂 js/
│       ├── api.js                 # API client for backend communication
│       └── app.js                 # Main application logic
│
├── 📂 src/
│   ├── 📂 models/
│   │   ├── train_image_model.py   # ViT/Swin image classifier training
│   │   └── train_multimodal.py    # Multimodal fusion transformer
│   ├── 📂 preprocessing/
│   │   ├── preprocess_images.py   # Image preprocessing pipeline
│   │   ├── asr_transcribe.py      # Speech-to-text transcription
│   │   └── text_nlu.py            # NLU entity extraction
│   ├── 📂 data/
│   └── 📂 utils/
│
├── 📂 tools/
│   ├── annotator_app.py           # Streamlit annotation tool
│   └── verify_auth.py             # Authentication verification script
│
├── 📂 data/
│   ├── 📂 schemas/                # JSON schemas for data validation
│   └── 📂 instruction_data/       # LLM training/fine-tuning data
│
├── 📂 config/
│   └── aug_config.yaml            # Data augmentation configuration
│
├── 📂 docs/
│   ├── PROJECT_SCOPE.md           # Project scope documentation
│   ├── CROP_DISEASE_TAXONOMY.md   # Disease classification taxonomy
│   ├── DATA_COLLECTION_PROTOCOL.md # Data collection guidelines
│   └── LABELING_GUIDELINES.md     # Annotation guidelines
│
├── 📄 requirements.txt            # Python dependencies
├── 📄 Dockerfile                  # Docker image configuration
├── 📄 docker-compose.yml          # Docker Compose configuration
├── 📄 nginx.conf                  # Nginx reverse proxy config
├── 📄 start.sh                    # Quick start script
├── 📄 LICENSE                     # MIT License
└── 📄 README.md                   # This file
```

---

## 📡 API Documentation

### Authentication Endpoints

| Method | Endpoint             | Description           | Auth Required |
| ------ | -------------------- | --------------------- | ------------- |
| POST   | `/api/auth/register` | Register new user     | ❌ No         |
| POST   | `/api/auth/login`    | Login and get JWT     | ❌ No         |
| GET    | `/api/auth/me`       | Get current user info | ✅ Yes        |

### Case Management Endpoints

| Method | Endpoint                       | Description               | Auth Required |
| ------ | ------------------------------ | ------------------------- | ------------- |
| POST   | `/api/cases`                   | Create new diagnosis case | ✅ Yes        |
| GET    | `/api/cases`                   | List user's cases         | ✅ Yes        |
| GET    | `/api/cases/{id}`              | Get case details          | ✅ Yes        |
| POST   | `/api/cases/{id}/media/image`  | Upload image              | ✅ Yes        |
| POST   | `/api/cases/{id}/media/speech` | Upload voice note         | ✅ Yes        |
| POST   | `/api/cases/{id}/run`          | Start AI analysis         | ✅ Yes        |
| GET    | `/api/cases/{id}/result`       | Get diagnosis result      | ✅ Yes        |

### Example API Usage

#### Register a New User

```bash
curl -X POST "http://localhost:8000/api/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "farmer@example.com",
    "password": "securepassword123",
    "full_name": "John Farmer"
  }'
```

#### Login

```bash
curl -X POST "http://localhost:8000/api/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "farmer@example.com",
    "password": "securepassword123"
  }'
```

#### Create a Case

```bash
curl -X POST "http://localhost:8000/api/cases" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "category": "crop",
    "crop_name": "tomato"
  }'
```

---

## 🧠 Model Training

### 1. Kaggle Dataset Downloader
We have added a utility script to automate downloading and mapping external plant disease datasets (PlantVillage & Rice Leaf Diseases) into standard disease labels.

Make sure you have your Kaggle API credentials set up at `~/.kaggle/kaggle.json`. Then, run:
```bash
# Download and organize both datasets automatically
python scripts/download_kaggle_data.py
```
This organizes the images under `data/dataset/train` and `data/dataset/val` mapped into unified folders and generates `data/dataset/labels.csv`.

### 2. Local CNN Model Training
To train the CNN model (ViT or Swin backbone) locally with advanced data augmentations and early stopping, run:
```bash
# Quick run with one convenience script
chmod +x scripts/train_cnn.sh
./scripts/train_cnn.sh
```
Or run the python script directly:
```bash
python src/models/train_image_model.py train \
    --labels data/dataset/labels.csv \
    --images data/dataset \
    --backbone vit_b_16 \
    --epochs 15 \
    --batch-size 32 \
    --learning-rate 0.0001 \
    --early-stop 5 \
    --use-folders
```
This saves the trained PyTorch checkpoint `best_model.pt` and a scripted TorchScript version `best_model_scripted.pt` under `data/models/`.

### 3. Multimodal Fusion Training

```bash
python src/models/train_multimodal.py \
    --labels data/labels.csv \
    --images data/images \
    --entities data/entities \
    --epochs 30 \
    --batch-size 16
```

### 4. Data Annotation Tool

```bash
# Run the Streamlit annotation tool
streamlit run tools/annotator_app.py

# Or with Docker
docker-compose --profile annotator up
```

---

## 🖼️ Screenshots

_(Add screenshots of your application here)_

| Feature           | Screenshot         |
| ----------------- | ------------------ |
| Login Page        | _(Add screenshot)_ |
| Dashboard         | _(Add screenshot)_ |
| Image Upload      | _(Add screenshot)_ |
| Diagnosis Results (Hybrid Badge) | _(Add screenshot)_ |
| Treatment Advice  | _(Add screenshot)_ |

---

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `SECRET_KEY` | *(Required)* | JWT signing secret key |
| `GROQ_API_KEY` | *(Required)* | Groq API Key |
| `GROQ_VISION_MODEL` | `meta-llama/llama-4-scout-17b-16e-instruct` | Groq Vision model |
| `GROQ_AUDIO_MODEL` | `whisper-large-v3-turbo` | Groq ASR Audio model |
| `GROQ_TEXT_MODEL` | `llama-3.3-70b-versatile` | Groq text model for advice |
| `AI_TIMEOUT_SECONDS` | `30` | Network request timeout for AI calls |
| `AI_MAX_RETRIES` | `2` | Number of retries for Groq API calls |
| `USE_LOCAL_CNN` | `True` | Toggle local CNN image classification engine |
| `CNN_MODEL_PATH` | `data/models/best_model.pt` | Path to trained CNN weights |
| `CNN_CONFIDENCE_THRESHOLD` | `0.80` | Confidence threshold to bypass cloud vision call |
| `LOG_LEVEL` | `INFO` | Logging level |
| `UPLOAD_DIR` | `./uploads` | Directory for uploaded files |
| `DATABASE_URL` | `sqlite:///./agridoctor.db` | Database connection URL |

### Creating a `.env` File

```bash
# Create .env file in the project root
cat > .env << EOF
SECRET_KEY=your-super-secret-key-here
GROQ_API_KEY=your-groq-api-key
GROQ_VISION_MODEL=meta-llama/llama-4-scout-17b-16e-instruct
GROQ_AUDIO_MODEL=whisper-large-v3-turbo
USE_LOCAL_CNN=True
CNN_MODEL_PATH=data/models/best_model.pt
CNN_CONFIDENCE_THRESHOLD=0.80
LOG_LEVEL=INFO
EOF
```

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=backend --cov-report=html

# Run specific test file
pytest tests/test_auth.py -v
```

---

## 📄 License & Disclaimer

### License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### Disclaimer

> ⚠️ **Important**: This application provides AI-generated guidance and is **NOT** a substitute for professional agricultural or veterinary diagnosis. Always consult local agricultural experts, veterinarians, and extension services for confirmation before applying any treatments. The creators of this application are not responsible for any decisions made based on the AI recommendations.

---

## 👨‍💻 Author

**Mohammad Sarif Khan**  
Student Number: 2238572  
AI-4-Creativity Project

---

## 🙏 Acknowledgements

- FastAPI team for the excellent web framework
- Hugging Face for transformer models
- OpenAI for the Whisper ASR model
- PyTorch team for the deep learning framework
- All contributors and testers

---

<div align="center">

**Made with ❤️ for farmers worldwide**

🌱 _Healthy Crops, Healthy Future_ 🌱

</div>
