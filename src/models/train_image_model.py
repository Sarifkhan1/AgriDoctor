"""
AgriDoctor AI - Vision Transformer Image Model
Baseline crop disease classifier using ViT/Swin encoder.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as T
from PIL import Image
import pandas as pd

try:
    from torchvision.models import vit_b_16, ViT_B_16_Weights
    from torchvision.models import swin_t, Swin_T_Weights
    HAS_PRETRAINED = True
except ImportError:
    HAS_PRETRAINED = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Disease Labels
# ============================================================================

DISEASE_LABELS = [
    # Tomato (10)
    "TOM_EARLY_BLIGHT", "TOM_LATE_BLIGHT", "TOM_LEAF_MOLD", "TOM_SEPTORIA",
    "TOM_SPIDER_MITES", "TOM_MOSAIC", "TOM_BACL_SPOT", "TOM_BLOSSOM_ROT",
    "TOM_HEALTHY", "TOM_UNKNOWN",
    # Potato (8)
    "POT_EARLY_BLIGHT", "POT_LATE_BLIGHT", "POT_BLACKLEG", "POT_SCAB",
    "POT_VIRAL", "POT_APHIDS", "POT_HEALTHY", "POT_UNKNOWN",
    # Rice (8)
    "RICE_BLAST", "RICE_BROWN_SPOT", "RICE_BACT_BLIGHT", "RICE_TUNGRO",
    "RICE_SHEATH_BLIGHT", "RICE_STEMB", "RICE_HEALTHY", "RICE_UNKNOWN",
    # Maize (7)
    "MAIZE_NLB", "MAIZE_RUST", "MAIZE_GLS", "MAIZE_SMUT",
    "MAIZE_BORER", "MAIZE_HEALTHY", "MAIZE_UNKNOWN",
    # Chili (7)
    "CHILI_ANTHRAC", "CHILI_BACT_WILT", "CHILI_LEAF_CURL", "CHILI_POWDERY",
    "CHILI_THRIPS", "CHILI_HEALTHY", "CHILI_UNKNOWN",
    # Cucumber (8)
    "CUC_POWDERY", "CUC_DOWNY", "CUC_ANGULAR", "CUC_MOSAIC",
    "CUC_ANTHRAC", "CUC_APHIDS", "CUC_HEALTHY", "CUC_UNKNOWN"
]

LABEL_TO_IDX = {label: idx for idx, label in enumerate(DISEASE_LABELS)}
IDX_TO_LABEL = {idx: label for idx, label in enumerate(DISEASE_LABELS)}
NUM_CLASSES = len(DISEASE_LABELS)


# ============================================================================
# Dataset
# ============================================================================

class CropDiseaseDataset(Dataset):
    """Dataset for crop disease classification."""
    
    def __init__(
        self,
        labels_csv: str,
        image_dir: str,
        transform: Optional[T.Compose] = None,
        mode: str = 'train'
    ):
        """
        Args:
            labels_csv: Path to labels.csv
            image_dir: Base directory for images
            transform: Image transforms
            mode: 'train' or 'val'
        """
        self.image_dir = Path(image_dir)
        self.mode = mode
        
        # Load labels
        self.df = pd.read_csv(labels_csv)
        
        # Filter valid entries
        self.df = self.df[self.df['primary_label'].isin(DISEASE_LABELS)]
        
        # Default transform
        if transform is None:
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        
        logger.info(f"Loaded {len(self.df)} samples for {mode}")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        
        # Load image
        image_path = row['image_path']
        if not Path(image_path).is_absolute():
            image_path = self.image_dir / image_path
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.warning(f"Error loading {image_path}: {e}")
            # Return a blank image
            image = Image.new('RGB', (224, 224), color='gray')
        
        if self.transform:
            image = self.transform(image)
        
        # Get labels
        primary_label = row['primary_label']
        label_idx = LABEL_TO_IDX.get(primary_label, 0)
        
        # Severity score
        severity = float(row.get('severity_score', 0.5))
        
        return {
            'image': image,
            'label': torch.tensor(label_idx, dtype=torch.long),
            'severity': torch.tensor(severity, dtype=torch.float32),
            'image_path': str(image_path)
        }


# ============================================================================
# Model
# ============================================================================

class CropDiseaseViT(nn.Module):
    """Vision Transformer for crop disease classification."""
    
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        backbone: str = 'vit_b_16',
        pretrained: bool = True,
        dropout: float = 0.1
    ):
        """
        Args:
            num_classes: Number of disease classes
            backbone: 'vit_b_16' or 'swin_t'
            pretrained: Use pretrained weights
            dropout: Dropout rate
        """
        super().__init__()
        
        self.backbone_name = backbone
        self.num_classes = num_classes
        
        # Load backbone
        if backbone == 'vit_b_16':
            if pretrained and HAS_PRETRAINED:
                self.backbone = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
            else:
                self.backbone = vit_b_16(weights=None)
            hidden_dim = 768
            # Replace classification head
            self.backbone.heads = nn.Identity()
            
        elif backbone == 'swin_t':
            if pretrained and HAS_PRETRAINED:
                self.backbone = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
            else:
                self.backbone = swin_t(weights=None)
            hidden_dim = 768
            # Replace classification head
            self.backbone.head = nn.Identity()
        
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        
        # Severity head (regression)
        self.severity_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Store feature dim for GradCAM
        self.hidden_dim = hidden_dim
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input images [B, 3, H, W]
            return_features: Whether to return intermediate features
            
        Returns:
            Dict with 'logits', 'severity', and optionally 'features'
        """
        # Extract features
        features = self.backbone(x)
        
        # Classification
        logits = self.classifier(features)
        
        # Severity regression
        severity = self.severity_head(features).squeeze(-1)
        
        output = {
            'logits': logits,
            'severity': severity
        }
        
        if return_features:
            output['features'] = features
        
        return output
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Extract attention weights for visualization."""
        if self.backbone_name != 'vit_b_16':
            return None
        
        # Hook to capture attention
        attention_weights = []
        
        def hook_fn(module, input, output):
            # ViT attention output
            attention_weights.append(output[1])
        
        # Register hooks on attention layers
        hooks = []
        for block in self.backbone.encoder.layers:
            hook = block.self_attention.register_forward_hook(hook_fn)
            hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            _ = self.backbone(x)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        if attention_weights:
            return torch.stack(attention_weights)
        return None


# ============================================================================
# Training
# ============================================================================

class Trainer:
    """Train and evaluate crop disease model."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda',
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        num_epochs: int = 50,
        output_dir: str = './outputs'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs,
            eta_min=1e-6
        )
        
        # Loss functions
        self.cls_criterion = nn.CrossEntropyLoss()
        self.sev_criterion = nn.MSELoss()
        
        # Metrics history
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [], 'val_f1': []
        }
        self.best_f1 = 0.0
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch in self.train_loader:
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            severity = batch['severity'].to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(images)
            
            # Classification loss
            cls_loss = self.cls_criterion(outputs['logits'], labels)
            
            # Severity loss
            sev_loss = self.sev_criterion(outputs['severity'], severity)
            
            # Combined loss
            loss = cls_loss + 0.3 * sev_loss
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Accuracy
            preds = outputs['logits'].argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        return {
            'loss': total_loss / len(self.train_loader),
            'accuracy': correct / total
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                severity = batch['severity'].to(self.device)
                
                outputs = self.model(images)
                
                cls_loss = self.cls_criterion(outputs['logits'], labels)
                sev_loss = self.sev_criterion(outputs['severity'], severity)
                loss = cls_loss + 0.3 * sev_loss
                
                total_loss += loss.item()
                
                preds = outputs['logits'].argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        accuracy = (all_preds == all_labels).mean()
        
        # Calculate per-class F1
        from sklearn.metrics import f1_score
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        return {
            'loss': total_loss / len(self.val_loader),
            'accuracy': accuracy,
            'f1': f1
        }
    
    def train(self):
        """Full training loop."""
        logger.info(f"Starting training for {self.num_epochs} epochs")
        
        for epoch in range(self.num_epochs):
            train_metrics = self.train_epoch()
            val_metrics = self.validate()
            
            self.scheduler.step()
            
            # Log
            logger.info(
                f"Epoch {epoch+1}/{self.num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f} - "
                f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}"
            )
            
            # Save best
            if val_metrics['f1'] > self.best_f1:
                self.best_f1 = val_metrics['f1']
                self.save_checkpoint('best_model.pt')
                logger.info(f"New best F1: {self.best_f1:.4f}")
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_f1'].append(val_metrics['f1'])
        
        # Save final
        self.save_checkpoint('final_model.pt')
        self.save_metrics()
        
        return self.history
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = self.output_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_f1': self.best_f1,
            'num_classes': self.model.num_classes,
            'backbone': self.model.backbone_name
        }, path)
        logger.info(f"Saved checkpoint: {path}")
    
    def save_metrics(self):
        """Save training history and metrics."""
        metrics_path = self.output_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump({
                'history': self.history,
                'best_f1': self.best_f1,
                'num_classes': self.model.num_classes,
                'trained_at': datetime.now().isoformat()
            }, f, indent=2)
        logger.info(f"Saved metrics: {metrics_path}")


# ============================================================================
# Inference
# ============================================================================

class Predictor:
    """Run inference with trained model."""
    
    def __init__(self, checkpoint_path: str, device: str = 'auto'):
        self.device = self._get_device(device)
        self.model = self._load_model(checkpoint_path)
        
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _get_device(self, device: str) -> str:
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def _load_model(self, path: str) -> nn.Module:
        checkpoint = torch.load(path, map_location=self.device)
        
        model = CropDiseaseViT(
            num_classes=checkpoint.get('num_classes', NUM_CLASSES),
            backbone=checkpoint.get('backbone', 'vit_b_16'),
            pretrained=False
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        logger.info(f"Loaded model from {path}")
        return model
    
    def predict(self, image_path: str) -> Dict:
        """
        Predict disease for a single image.
        
        Returns:
            Dict with predictions and confidence
        """
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_tensor, return_features=True)
        
        logits = outputs['logits'][0]
        probs = F.softmax(logits, dim=0)
        
        # Top-5 predictions
        top_k = min(5, NUM_CLASSES)
        top_probs, top_indices = probs.topk(top_k)
        
        predictions = []
        for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
            predictions.append({
                'label': IDX_TO_LABEL[idx],
                'confidence': float(prob)
            })
        
        return {
            'primary_label': predictions[0]['label'],
            'confidence': predictions[0]['confidence'],
            'severity': float(outputs['severity'][0].cpu()),
            'top_predictions': predictions
        }


def main():
    """CLI for training image model."""
    parser = argparse.ArgumentParser(description="Train crop disease image model")
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train model')
    train_parser.add_argument('--labels', required=True, help='Path to labels.csv')
    train_parser.add_argument('--images', required=True, help='Image directory')
    train_parser.add_argument('--output', default='./outputs', help='Output directory')
    train_parser.add_argument('--backbone', default='vit_b_16', choices=['vit_b_16', 'swin_t'])
    train_parser.add_argument('--epochs', type=int, default=50)
    train_parser.add_argument('--batch-size', type=int, default=32)
    train_parser.add_argument('--lr', type=float, default=1e-4)
    train_parser.add_argument('--device', default='auto')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Run inference')
    predict_parser.add_argument('--checkpoint', required=True, help='Model checkpoint')
    predict_parser.add_argument('--image', required=True, help='Image to classify')
    predict_parser.add_argument('--device', default='auto')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else 'cpu'
        
        # Create datasets
        train_dataset = CropDiseaseDataset(args.labels, args.images, mode='train')
        val_dataset = CropDiseaseDataset(args.labels, args.images, mode='val')
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        
        # Create model
        model = CropDiseaseViT(backbone=args.backbone, pretrained=True)
        
        # Train
        trainer = Trainer(
            model, train_loader, val_loader,
            device=device,
            learning_rate=args.lr,
            num_epochs=args.epochs,
            output_dir=args.output
        )
        trainer.train()
        
    elif args.command == 'predict':
        predictor = Predictor(args.checkpoint, args.device)
        result = predictor.predict(args.image)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
