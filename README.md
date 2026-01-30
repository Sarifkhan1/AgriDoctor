# AgriDoctor AI 🌿

An AI-powered multimodal agricultural health assistant for diagnosing crop diseases using images and voice descriptions.

![Version](https://img.shields.io/badge/version-1.0.0-green)
![License](https://img.shields.io/badge/license-MIT-blue)

## Features

- **🖼️ Image Analysis** - Upload photos of affected crops for AI disease detection
- **🎤 Voice Input** - Describe symptoms using voice notes (supports multiple languages)
- **🤖 Multimodal Fusion** - Combines visual and text data for accurate diagnosis
- **💡 Actionable Advice** - Get treatment recommendations and prevention tips
- **📱 Mobile-Friendly** - Responsive design works on any device
- **🔒 Secure** - JWT authentication and secure data handling

## Supported Crops

| Crop        | Diseases Detected                                            |
| ----------- | ------------------------------------------------------------ |
| 🍅 Tomato   | Early Blight, Late Blight, Leaf Mold, Septoria, Mosaic, etc. |
| 🥔 Potato   | Early Blight, Late Blight, Blackleg, Scab, Viral             |
| 🌾 Rice     | Blast, Brown Spot, Bacterial Blight, Tungro                  |
| 🌽 Maize    | Northern Leaf Blight, Rust, Gray Leaf Spot, Smut             |
| 🌶️ Chili    | Anthracnose, Bacterial Wilt, Leaf Curl, Powdery Mildew       |
| 🥒 Cucumber | Powdery Mildew, Downy Mildew, Angular Leaf Spot, Mosaic      |

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Or: Python 3.11+, Node.js (optional for frontend development)

### Using Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/agridoctor-ai.git
cd agridoctor-ai

# Start all services
docker-compose up -d

# Access the application
open http://localhost
```

### Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run backend
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Serve frontend (in another terminal)
cd frontend
python -m http.server 3000
```

## Project Structure

```
agridoctor-ai/
├── backend/
│   └── main.py              # FastAPI application
├── frontend/
│   ├── index.html           # Main HTML
│   ├── css/styles.css       # Styling
│   └── js/
│       ├── api.js           # API client
│       └── app.js           # Application logic
├── src/
│   ├── models/
│   │   ├── train_image_model.py    # ViT/Swin classifier
│   │   └── train_multimodal.py     # Fusion transformer
│   └── preprocessing/
│       ├── preprocess_images.py    # Image pipeline
│       ├── asr_transcribe.py       # Speech-to-text
│       └── text_nlu.py             # Entity extraction
├── tools/
│   └── annotator_app.py     # Streamlit labeling tool
├── data/
│   ├── schemas/             # JSON schemas
│   └── instruction_data/    # LLM training data
├── config/
│   └── aug_config.yaml      # Augmentation settings
├── docs/
│   ├── PROJECT_SCOPE.md     # MVP/V1 scope
│   ├── CROP_DISEASE_TAXONOMY.md
│   ├── DATA_COLLECTION_PROTOCOL.md
│   └── LABELING_GUIDELINES.md
├── docker-compose.yml
├── Dockerfile
├── nginx.conf
└── requirements.txt
```

## API Endpoints

| Method | Endpoint                       | Description               |
| ------ | ------------------------------ | ------------------------- |
| POST   | `/api/auth/register`           | Register new user         |
| POST   | `/api/auth/login`              | Login and get JWT         |
| POST   | `/api/cases`                   | Create new diagnosis case |
| POST   | `/api/cases/{id}/media/image`  | Upload image              |
| POST   | `/api/cases/{id}/media/speech` | Upload voice note         |
| POST   | `/api/cases/{id}/run`          | Start analysis            |
| GET    | `/api/cases/{id}/result`       | Get diagnosis result      |

Full API documentation: http://localhost:8000/docs

## Model Training

### Image Classifier

```bash
python src/models/train_image_model.py train \
    --labels data/labels.csv \
    --images data/images \
    --backbone vit_b_16 \
    --epochs 50 \
    --batch-size 32
```

### Multimodal Fusion

```bash
python src/models/train_multimodal.py \
    --labels data/labels.csv \
    --images data/images \
    --entities data/entities \
    --epochs 30
```

## Data Annotation

```bash
# Run the annotation tool
streamlit run tools/annotator_app.py

# Or with Docker
docker-compose --profile annotator up
```

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Frontend  │────▶│   Backend   │────▶│    Models   │
│  (JS/HTML)  │     │  (FastAPI)  │     │  (PyTorch)  │
└─────────────┘     └─────────────┘     └─────────────┘
                          │
                    ┌─────┴─────┐
                    │  SQLite   │
                    │  Database │
                    └───────────┘
```

## Configuration

Environment variables:

| Variable        | Default   | Description        |
| --------------- | --------- | ------------------ |
| `SECRET_KEY`    | `dev-key` | JWT signing key    |
| `WHISPER_MODEL` | `base`    | Whisper model size |
| `LOG_LEVEL`     | `INFO`    | Logging level      |

## Team

Developed as part of the AI Lab research project.

## License

MIT License - see LICENSE file for details.

## Disclaimer

⚠️ **Important**: This is AI-generated guidance, not professional agricultural or veterinary diagnosis. Always consult local agricultural experts for confirmation and before applying any treatments.
