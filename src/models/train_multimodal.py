"""
AgriDoctor AI - Multimodal Fusion Transformer
Combines image embedding + text embedding for enhanced disease classification.
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
    from transformers import AutoTokenizer, AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from torchvision.models import vit_b_16, ViT_B_16_Weights
    HAS_VIT = True
except ImportError:
    HAS_VIT = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Import labels from image model
from train_image_model import DISEASE_LABELS, LABEL_TO_IDX, IDX_TO_LABEL, NUM_CLASSES


# ============================================================================
# Dataset
# ============================================================================

class MultimodalDataset(Dataset):
    """Dataset for multimodal (image + text) classification."""
    
    def __init__(
        self,
        labels_csv: str,
        image_dir: str,
        entities_dir: str,
        tokenizer,
        image_transform: Optional[T.Compose] = None,
        max_text_length: int = 128,
        mode: str = 'train'
    ):
        """
        Args:
            labels_csv: Path to labels.csv
            image_dir: Base directory for images
            entities_dir: Directory with extracted entities JSON files
            tokenizer: HuggingFace tokenizer
            image_transform: Image transforms
            max_text_length: Maximum text sequence length
            mode: 'train' or 'val'
        """
        self.image_dir = Path(image_dir)
        self.entities_dir = Path(entities_dir)
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.mode = mode
        
        # Load labels
        self.df = pd.read_csv(labels_csv)
        self.df = self.df[self.df['primary_label'].isin(DISEASE_LABELS)]
        
        # Default image transform
        if image_transform is None:
            self.image_transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.image_transform = image_transform
        
        logger.info(f"Loaded {len(self.df)} multimodal samples for {mode}")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def _load_text(self, encounter_id: str) -> str:
        """Load text from entities file or generate from metadata."""
        entities_file = self.entities_dir / f"{encounter_id}_entities.json"
        
        if entities_file.exists():
            with open(entities_file, 'r') as f:
                entities = json.load(f)
            
            # Build structured text from entities
            parts = []
            if entities.get('crop_name'):
                parts.append(f"Crop: {entities['crop_name']}")
            if entities.get('symptoms'):
                parts.append(f"Symptoms: {', '.join(entities['symptoms'])}")
            if entities.get('affected_parts'):
                parts.append(f"Affected: {', '.join(entities['affected_parts'])}")
            if entities.get('duration_text'):
                parts.append(f"Duration: {entities['duration_text']}")
            if entities.get('spread_speed'):
                parts.append(f"Spread: {entities['spread_speed']}")
            if entities.get('weather_conditions'):
                parts.append(f"Weather: {', '.join(entities['weather_conditions'])}")
            
            return " | ".join(parts) if parts else "No additional information available."
        
        return "No text information available."
    
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
            image = Image.new('RGB', (224, 224), color='gray')
        
        if self.image_transform:
            image = self.image_transform(image)
        
        # Get encounter_id and load text
        encounter_id = row.get('encounter_id', Path(image_path).stem.split('_')[0])
        text = self._load_text(encounter_id)
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_text_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Labels
        primary_label = row['primary_label']
        label_idx = LABEL_TO_IDX.get(primary_label, 0)
        severity = float(row.get('severity_score', 0.5))
        
        return {
            'image': image,
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label_idx, dtype=torch.long),
            'severity': torch.tensor(severity, dtype=torch.float32),
            'text': text,
            'image_path': str(image_path)
        }


# ============================================================================
# Model Components
# ============================================================================

class ImageEncoder(nn.Module):
    """Image encoder using ViT."""
    
    def __init__(self, hidden_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        
        if HAS_VIT:
            self.backbone = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
            self.backbone.heads = nn.Identity()
        else:
            # Fallback to simple CNN
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(3, stride=2, padding=1),
                nn.Conv2d(64, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(256, hidden_dim)
            )
        
        self.projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.projection(features)


class TextEncoder(nn.Module):
    """Text encoder using transformer."""
    
    def __init__(
        self,
        model_name: str = 'distilbert-base-uncased',
        hidden_dim: int = 768,
        dropout: float = 0.1
    ):
        super().__init__()
        
        if HAS_TRANSFORMERS:
            self.transformer = AutoModel.from_pretrained(model_name)
            transformer_dim = self.transformer.config.hidden_size
        else:
            # Fallback to simple embedding
            self.transformer = None
            self.embedding = nn.Embedding(30522, 256)  # BERT vocab size
            self.rnn = nn.LSTM(256, 384, bidirectional=True, batch_first=True)
            transformer_dim = 768
        
        self.projection = nn.Sequential(
            nn.LayerNorm(transformer_dim),
            nn.Dropout(dropout),
            nn.Linear(transformer_dim, hidden_dim)
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        if self.transformer is not None:
            outputs = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            # Use [CLS] token
            features = outputs.last_hidden_state[:, 0, :]
        else:
            embedded = self.embedding(input_ids)
            output, _ = self.rnn(embedded)
            features = output.mean(dim=1)
        
        return self.projection(features)


class CrossModalAttention(nn.Module):
    """Cross-modal attention between image and text features."""
    
    def __init__(self, hidden_dim: int = 768, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.img_to_text = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.text_to_img = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.norm3 = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        img_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cross-modal attention.
        
        Args:
            img_features: [B, hidden_dim]
            text_features: [B, hidden_dim]
            
        Returns:
            Tuple of enhanced (img_features, text_features)
        """
        # Reshape for attention (add sequence dimension)
        img_seq = img_features.unsqueeze(1)  # [B, 1, D]
        text_seq = text_features.unsqueeze(1)  # [B, 1, D]
        
        # Image attends to text
        img_attn, _ = self.img_to_text(img_seq, text_seq, text_seq)
        img_enhanced = self.norm1(img_seq + img_attn)
        
        # Text attends to image
        text_attn, _ = self.text_to_img(text_seq, img_seq, img_seq)
        text_enhanced = self.norm2(text_seq + text_attn)
        
        # Squeeze back
        img_enhanced = img_enhanced.squeeze(1)
        text_enhanced = text_enhanced.squeeze(1)
        
        # Fusion via FFN
        fused = torch.cat([img_enhanced, text_enhanced], dim=-1)
        fused = self.ffn(fused)
        fused = self.norm3(fused)
        
        return fused, img_enhanced, text_enhanced


# ============================================================================
# Multimodal Model
# ============================================================================

class MultimodalFusionTransformer(nn.Module):
    """Multimodal transformer for image + text fusion."""
    
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        hidden_dim: int = 768,
        text_model: str = 'distilbert-base-uncased',
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # Encoders
        self.image_encoder = ImageEncoder(hidden_dim, dropout)
        self.text_encoder = TextEncoder(text_model, hidden_dim, dropout)
        
        # Cross-modal fusion
        self.cross_attention = CrossModalAttention(hidden_dim, num_heads=8, dropout=dropout)
        
        # Classification heads
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        
        # Severity head
        self.severity_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Modality importance weights (learnable)
        self.modality_weights = nn.Parameter(torch.ones(2) / 2)
    
    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            images: Input images [B, 3, H, W]
            input_ids: Text token IDs [B, seq_len]
            attention_mask: Text attention mask [B, seq_len]
            return_features: Return intermediate features
            
        Returns:
            Dict with predictions and optionally features
        """
        # Encode modalities
        img_features = self.image_encoder(images)
        text_features = self.text_encoder(input_ids, attention_mask)
        
        # Cross-modal fusion
        fused_features, img_enhanced, text_enhanced = self.cross_attention(
            img_features, text_features
        )
        
        # Classification
        logits = self.classifier(fused_features)
        
        # Severity
        severity = self.severity_head(fused_features).squeeze(-1)
        
        output = {
            'logits': logits,
            'severity': severity,
            'modality_weights': F.softmax(self.modality_weights, dim=0)
        }
        
        if return_features:
            output['fused_features'] = fused_features
            output['img_features'] = img_enhanced
            output['text_features'] = text_enhanced
        
        return output
    
    def forward_image_only(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Image-only forward pass for ablation."""
        img_features = self.image_encoder(images)
        logits = self.classifier(img_features)
        severity = self.severity_head(img_features).squeeze(-1)
        return {'logits': logits, 'severity': severity}
    
    def forward_text_only(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Text-only forward pass for ablation."""
        text_features = self.text_encoder(input_ids, attention_mask)
        logits = self.classifier(text_features)
        severity = self.severity_head(text_features).squeeze(-1)
        return {'logits': logits, 'severity': severity}


# ============================================================================
# Training
# ============================================================================

class MultimodalTrainer:
    """Train multimodal fusion model."""
    
    def __init__(
        self,
        model: MultimodalFusionTransformer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda',
        learning_rate: float = 5e-5,
        num_epochs: int = 30,
        output_dir: str = './outputs/multimodal'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimizer with different LR for encoders
        encoder_params = list(model.image_encoder.parameters()) + list(model.text_encoder.parameters())
        fusion_params = list(model.cross_attention.parameters()) + \
                       list(model.classifier.parameters()) + \
                       list(model.severity_head.parameters())
        
        self.optimizer = AdamW([
            {'params': encoder_params, 'lr': learning_rate * 0.1},
            {'params': fusion_params, 'lr': learning_rate}
        ], weight_decay=0.01)
        
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs)
        
        self.cls_criterion = nn.CrossEntropyLoss()
        self.sev_criterion = nn.MSELoss()
        
        self.history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
        self.best_f1 = 0.0
    
    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        
        for batch in self.train_loader:
            images = batch['image'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            severity = batch['severity'].to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(images, input_ids, attention_mask)
            
            cls_loss = self.cls_criterion(outputs['logits'], labels)
            sev_loss = self.sev_criterion(outputs['severity'], severity)
            
            loss = cls_loss + 0.3 * sev_loss
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_loader)
    
    def validate(self, mode: str = 'multimodal') -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                severity = batch['severity'].to(self.device)
                
                if mode == 'multimodal':
                    outputs = self.model(images, input_ids, attention_mask)
                elif mode == 'image_only':
                    outputs = self.model.forward_image_only(images)
                elif mode == 'text_only':
                    outputs = self.model.forward_text_only(input_ids, attention_mask)
                
                cls_loss = self.cls_criterion(outputs['logits'], labels)
                sev_loss = self.sev_criterion(outputs['severity'], severity)
                loss = cls_loss + 0.3 * sev_loss
                
                total_loss += loss.item()
                
                preds = outputs['logits'].argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        from sklearn.metrics import f1_score, accuracy_score
        
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        return {
            'loss': total_loss / len(self.val_loader),
            'accuracy': accuracy,
            'f1': f1
        }
    
    def run_ablation(self) -> Dict[str, Dict]:
        """Run ablation study comparing modalities."""
        logger.info("Running ablation study...")
        
        results = {
            'multimodal': self.validate('multimodal'),
            'image_only': self.validate('image_only'),
            'text_only': self.validate('text_only')
        }
        
        logger.info("Ablation Results:")
        for mode, metrics in results.items():
            logger.info(f"  {mode}: F1={metrics['f1']:.4f}, Acc={metrics['accuracy']:.4f}")
        
        # Save ablation results
        with open(self.output_dir / 'ablation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def train(self):
        logger.info(f"Training multimodal model for {self.num_epochs} epochs")
        
        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch()
            val_metrics = self.validate()
            
            self.scheduler.step()
            
            logger.info(
                f"Epoch {epoch+1}/{self.num_epochs} - "
                f"Train Loss: {train_loss:.4f} - "
                f"Val F1: {val_metrics['f1']:.4f}"
            )
            
            if val_metrics['f1'] > self.best_f1:
                self.best_f1 = val_metrics['f1']
                self.save_checkpoint('best_multimodal.pt')
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_f1'].append(val_metrics['f1'])
        
        # Final ablation
        self.run_ablation()
        
        # Save final
        self.save_checkpoint('final_multimodal.pt')
        self.save_metrics()
    
    def save_checkpoint(self, filename: str):
        path = self.output_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_f1': self.best_f1,
            'num_classes': self.model.num_classes,
            'hidden_dim': self.model.hidden_dim
        }, path)
    
    def save_metrics(self):
        with open(self.output_dir / 'metrics.json', 'w') as f:
            json.dump({
                'history': self.history,
                'best_f1': self.best_f1,
                'trained_at': datetime.now().isoformat()
            }, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Train multimodal fusion model")
    parser.add_argument('--labels', required=True, help='Path to labels.csv')
    parser.add_argument('--images', required=True, help='Image directory')
    parser.add_argument('--entities', required=True, help='Entities JSON directory')
    parser.add_argument('--output', default='./outputs/multimodal')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--device', default='auto')
    
    args = parser.parse_args()
    
    device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else 'cpu'
    
    # Tokenizer
    if HAS_TRANSFORMERS:
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    else:
        logger.error("transformers library required for text encoding")
        sys.exit(1)
    
    # Datasets
    train_dataset = MultimodalDataset(
        args.labels, args.images, args.entities, tokenizer, mode='train'
    )
    val_dataset = MultimodalDataset(
        args.labels, args.images, args.entities, tokenizer, mode='val'
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Model
    model = MultimodalFusionTransformer()
    
    # Train
    trainer = MultimodalTrainer(
        model, train_loader, val_loader,
        device=device,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        output_dir=args.output
    )
    trainer.train()


if __name__ == "__main__":
    main()
