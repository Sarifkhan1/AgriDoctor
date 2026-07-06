#!/usr/bin/env bash
# AgriDoctor — Download data from Kaggle and train the CNN model.
#
# Usage:
#   chmod +x scripts/train_cnn.sh
#   ./scripts/train_cnn.sh
#
# Prerequisites:
#   - pip install -r requirements-ml.txt
#   - pip install kaggle
#   - Place kaggle.json in ~/.kaggle/

set -euo pipefail
cd "$(dirname "$0")/.."

echo "================================================"
echo "  AgriDoctor CNN Training Pipeline"
echo "================================================"

# ---- Step 1: Download & organize data ----
echo ""
echo "📥 Step 1: Downloading and organizing Kaggle datasets..."
python scripts/download_kaggle_data.py

# ---- Step 2: Train the model ----
echo ""
echo "🧠 Step 2: Training ViT-B/16 image classifier..."
python src/models/train_image_model.py train \
    --labels data/dataset/labels.csv \
    --images data/dataset \
    --backbone vit_b_16 \
    --epochs 15 \
    --batch-size 32 \
    --lr 0.0001 \
    --output data/models

echo ""
echo "✅ Training complete!"
echo "   Model saved to: data/models/best_model.pt"
echo "   Metrics saved to: data/models/metrics.json"
echo ""
echo "🚀 To start the app with the trained model:"
echo "   uvicorn backend.main:app --reload --port 8000"
