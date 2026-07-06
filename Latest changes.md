# Latest Changes

Here is a summary of the latest updates and features integrated into the **AgriDoctor AI** application today:

## 1. Local CNN Inference Pipeline
- **Dataset Downloader & Organizer (`scripts/download_kaggle_data.py`)**: Developed an automated utility to download the public **PlantVillage** and **Rice Leaf Disease** datasets via the Kaggle API. The script automatically cleans, filters, and maps these folder labels to the unified AgriDoctor `DISEASE_LABELS` schema, splitting the images into `train` and `val` directories and outputting a structured `labels.csv`.
- **Training Script Enhancements (`src/models/train_image_model.py`)**:
  - Added folder-based dataset loading support (`CropDiseaseFolderDataset`).
  - Added advanced image augmentations for robust training (rotation, flip, color jitter, affine transforms).
  - Added Apple Silicon/MPS (`mps`) device acceleration.
  - Implemented early stopping to automatically halt training if validation F1 score does not improve.
  - Added model export to **TorchScript** (`best_model_scripted.pt`) for optimized runtime inference.
- **Convenience Pipeline Script (`scripts/train_cnn.sh`)**: Created a bash utility to execute the entire raw download-to-training pipeline in one step.

## 2. Hybrid AI Analyzer (`backend/ai/analyzer.py` & `cnn_predictor.py`)
- Created `CNNPredictor` to run inference locally using PyTorch. The model loading is performed lazily so the main API server starts instantly and runs without PyTorch dependencies if the model isn't trained yet.
- Integrated a **Hybrid Analysis Flow** in the orchestrator:
  - If a local trained CNN model exists, the image is passed to it first.
  - If the CNN returns a classification with confidence >= 80% (configurable), the local diagnosis is used. The system then makes a lightweight text-only query to the LLM solely to generate detailed, contextual treatment advice (significantly reducing API latency and token cost).
  - If the local CNN model has low confidence (< 80%) or is not found, the analyzer falls back to the full multimodal vision API pipeline.

## 3. UI and User Experience Updates
- **AI Source Badging**: Updated the frontend results UI (`frontend/js/ui.js` and `frontend/css/styles.css`) to display an active badge indicating the classification engine:
  - `🧠 Local AI` (for fast, local CNN inference results)
  - `☁️ Cloud AI` (for vision API fallback results)
- **CSS Improvements**: Added modern gradient backgrounds and custom animations for the new status badges.

## 4. Configuration and Environment Hardening
- Added new parameters to `backend/config.py`:
  - `cnn_model_path`: Path to local model weights (default: `data/models/best_model.pt`).
  - `cnn_confidence_threshold`: The minimum confidence score to accept local CNN results (default: `0.80`).
  - `use_local_cnn`: Toggle to enable/disable the local CNN engine entirely.
- Updated `.gitignore` to prevent tracking datasets, raw images, and model checkpoint binaries (`*.pt`, `*.ptl`).
- Updated `requirements.txt` to include `torch` and `torchvision` for local inference and `requirements-ml.txt` to include the `kaggle` package.
