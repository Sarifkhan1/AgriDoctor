"""
AgriDoctor AI - Image Preprocessing Pipeline
Resize, normalize, augment, and cache crop disease images.
"""

import os
import sys
import yaml
import argparse
import hashlib
import logging
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import numpy as np
from PIL import Image
import cv2
import torch
from torch import Tensor
import torchvision.transforms as T
import torchvision.transforms.functional as TF

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Preprocesses crop disease images for model training."""
    
    def __init__(self, config_path: str = None, seed: int = 42):
        """
        Initialize preprocessor with configuration.
        
        Args:
            config_path: Path to aug_config.yaml
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self.config = self._load_config(config_path)
        self._set_seed()
        
        # Initialize transforms
        self.preprocess_transform = self._build_preprocess_transform()
        self.train_transform = self._build_train_transform()
        self.val_transform = self._build_val_transform()
        
    def _load_config(self, config_path: str = None) -> dict:
        """Load configuration from YAML file."""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        
        # Default configuration
        return {
            'preprocessing': {
                'target_size': [224, 224],
                'normalize': {
                    'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]
                },
                'blur_detection': {'enabled': True, 'laplacian_threshold': 100},
                'color_constancy': {'enabled': True, 'method': 'gray_world'}
            },
            'train_augmentations': {
                'horizontal_flip': {'enabled': True, 'p': 0.5},
                'vertical_flip': {'enabled': True, 'p': 0.3},
                'rotation': {'enabled': True, 'degrees': [-30, 30], 'p': 0.5},
                'color_jitter': {'enabled': True, 'brightness': 0.2, 'contrast': 0.2, 
                                'saturation': 0.2, 'hue': 0.1, 'p': 0.5}
            },
            'seed': 42,
            'deterministic': True
        }
    
    def _set_seed(self):
        """Set random seeds for reproducibility."""
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
    
    def _build_preprocess_transform(self) -> T.Compose:
        """Build basic preprocessing transform."""
        config = self.config.get('preprocessing', {})
        target_size = config.get('target_size', [224, 224])
        normalize = config.get('normalize', {})
        
        return T.Compose([
            T.Resize(target_size),
            T.ToTensor(),
            T.Normalize(
                mean=normalize.get('mean', [0.485, 0.456, 0.406]),
                std=normalize.get('std', [0.229, 0.224, 0.225])
            )
        ])
    
    def _build_train_transform(self) -> T.Compose:
        """Build training augmentation pipeline."""
        config = self.config.get('preprocessing', {})
        aug_config = self.config.get('train_augmentations', {})
        target_size = config.get('target_size', [224, 224])
        normalize = config.get('normalize', {})
        
        transforms = []
        
        # Resize
        transforms.append(T.Resize(target_size))
        
        # Geometric transforms
        if aug_config.get('horizontal_flip', {}).get('enabled', True):
            p = aug_config['horizontal_flip'].get('p', 0.5)
            transforms.append(T.RandomHorizontalFlip(p=p))
        
        if aug_config.get('vertical_flip', {}).get('enabled', False):
            p = aug_config['vertical_flip'].get('p', 0.3)
            transforms.append(T.RandomVerticalFlip(p=p))
        
        if aug_config.get('rotation', {}).get('enabled', True):
            degrees = aug_config['rotation'].get('degrees', [-30, 30])
            transforms.append(T.RandomRotation(degrees=degrees))
        
        if aug_config.get('random_crop', {}).get('enabled', False):
            scale = aug_config['random_crop'].get('scale', [0.8, 1.0])
            ratio = aug_config['random_crop'].get('ratio', [0.9, 1.1])
            transforms.append(T.RandomResizedCrop(
                size=target_size,
                scale=scale,
                ratio=ratio
            ))
        
        if aug_config.get('affine', {}).get('enabled', False):
            transforms.append(T.RandomAffine(
                degrees=0,
                translate=tuple(aug_config['affine'].get('translate', [0.1, 0.1])),
                scale=tuple(aug_config['affine'].get('scale', [0.9, 1.1])),
                shear=aug_config['affine'].get('shear', [-10, 10])
            ))
        
        # Color transforms
        if aug_config.get('color_jitter', {}).get('enabled', True):
            cj = aug_config['color_jitter']
            transforms.append(T.ColorJitter(
                brightness=cj.get('brightness', 0.2),
                contrast=cj.get('contrast', 0.2),
                saturation=cj.get('saturation', 0.2),
                hue=cj.get('hue', 0.1)
            ))
        
        if aug_config.get('gaussian_blur', {}).get('enabled', False):
            gb = aug_config['gaussian_blur']
            transforms.append(T.GaussianBlur(
                kernel_size=gb.get('kernel_size', [3, 7]),
                sigma=gb.get('sigma', [0.1, 2.0])
            ))
        
        if aug_config.get('random_grayscale', {}).get('enabled', False):
            p = aug_config['random_grayscale'].get('p', 0.1)
            transforms.append(T.RandomGrayscale(p=p))
        
        # To tensor and normalize
        transforms.append(T.ToTensor())
        transforms.append(T.Normalize(
            mean=normalize.get('mean', [0.485, 0.456, 0.406]),
            std=normalize.get('std', [0.229, 0.224, 0.225])
        ))
        
        # Random erasing (after tensor)
        if aug_config.get('random_erasing', {}).get('enabled', False):
            re = aug_config['random_erasing']
            transforms.append(T.RandomErasing(
                p=re.get('p', 0.2),
                scale=tuple(re.get('scale', [0.02, 0.15])),
                ratio=tuple(re.get('ratio', [0.3, 3.3]))
            ))
        
        return T.Compose(transforms)
    
    def _build_val_transform(self) -> T.Compose:
        """Build validation transform (minimal augmentation)."""
        config = self.config.get('preprocessing', {})
        target_size = config.get('target_size', [224, 224])
        normalize = config.get('normalize', {})
        
        return T.Compose([
            T.Resize(target_size),
            T.CenterCrop(target_size),
            T.ToTensor(),
            T.Normalize(
                mean=normalize.get('mean', [0.485, 0.456, 0.406]),
                std=normalize.get('std', [0.229, 0.224, 0.225])
            )
        ])
    
    def detect_blur(self, image: np.ndarray) -> Tuple[bool, float]:
        """
        Detect if image is blurry using Laplacian variance.
        
        Returns:
            Tuple of (is_blurry, blur_score)
        """
        config = self.config.get('preprocessing', {}).get('blur_detection', {})
        if not config.get('enabled', True):
            return False, 0.0
        
        threshold = config.get('laplacian_threshold', 100)
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        is_blurry = laplacian_var < threshold
        
        return is_blurry, laplacian_var
    
    def apply_color_constancy(self, image: np.ndarray) -> np.ndarray:
        """
        Apply color constancy correction.
        
        Methods: gray_world, white_patch, max_rgb
        """
        config = self.config.get('preprocessing', {}).get('color_constancy', {})
        if not config.get('enabled', True):
            return image
        
        method = config.get('method', 'gray_world')
        image = image.astype(np.float32)
        
        if method == 'gray_world':
            # Gray World assumption
            avg = image.mean(axis=(0, 1))
            avg_gray = avg.mean()
            scale = avg_gray / (avg + 1e-6)
            corrected = image * scale
        
        elif method == 'white_patch':
            # White Patch assumption
            max_vals = image.max(axis=(0, 1))
            scale = 255.0 / (max_vals + 1e-6)
            corrected = image * scale
        
        elif method == 'max_rgb':
            # Max RGB
            max_rgb = image.max(axis=2, keepdims=True)
            corrected = image / (max_rgb + 1e-6) * 255
        
        else:
            corrected = image
        
        return np.clip(corrected, 0, 255).astype(np.uint8)
    
    def preprocess_image(
        self,
        image_path: str,
        mode: str = 'val',
        return_metadata: bool = False
    ) -> Tuple[Tensor, Optional[Dict]]:
        """
        Preprocess a single image.
        
        Args:
            image_path: Path to image file
            mode: 'train' or 'val' for different augmentations
            return_metadata: Whether to return preprocessing metadata
            
        Returns:
            Preprocessed tensor and optional metadata dict
        """
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            raise
        
        # Convert to numpy for preprocessing
        image_np = np.array(image)
        
        metadata = {}
        
        # Blur detection
        is_blurry, blur_score = self.detect_blur(image_np)
        metadata['is_blurry'] = is_blurry
        metadata['blur_score'] = blur_score
        
        if is_blurry:
            logger.warning(f"Image {image_path} is blurry (score: {blur_score:.2f})")
        
        # Color constancy
        image_np = self.apply_color_constancy(image_np)
        image = Image.fromarray(image_np)
        
        # Apply transforms
        if mode == 'train':
            tensor = self.train_transform(image)
        else:
            tensor = self.val_transform(image)
        
        metadata['original_size'] = list(Image.open(image_path).size)
        metadata['final_size'] = list(tensor.shape[1:])
        
        if return_metadata:
            return tensor, metadata
        return tensor, None
    
    def get_cache_path(self, image_path: str, cache_dir: str, mode: str) -> Path:
        """Generate cache path for processed tensor."""
        # Create hash of image path + mode for unique caching
        hash_input = f"{image_path}_{mode}_{self.seed}"
        hash_id = hashlib.md5(hash_input.encode()).hexdigest()[:12]
        
        filename = Path(image_path).stem
        cache_path = Path(cache_dir) / mode / f"{filename}_{hash_id}.pt"
        
        return cache_path
    
    def process_and_cache(
        self,
        image_path: str,
        cache_dir: str,
        mode: str = 'val',
        force: bool = False
    ) -> Path:
        """
        Process image and save to cache.
        
        Args:
            image_path: Path to source image
            cache_dir: Directory for cached tensors
            mode: 'train' or 'val'
            force: Overwrite existing cache
            
        Returns:
            Path to cached tensor file
        """
        cache_path = self.get_cache_path(image_path, cache_dir, mode)
        
        # Check cache
        if cache_path.exists() and not force:
            logger.debug(f"Using cached tensor: {cache_path}")
            return cache_path
        
        # Process
        tensor, metadata = self.preprocess_image(image_path, mode, return_metadata=True)
        
        # Save
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'tensor': tensor,
            'metadata': metadata,
            'source_path': image_path
        }, cache_path)
        
        logger.info(f"Cached: {cache_path}")
        return cache_path
    
    def process_directory(
        self,
        input_dir: str,
        cache_dir: str,
        mode: str = 'val',
        extensions: List[str] = None,
        force: bool = False
    ) -> List[Path]:
        """
        Process all images in a directory.
        
        Args:
            input_dir: Input directory with images
            cache_dir: Output cache directory
            mode: 'train' or 'val'
            extensions: Image file extensions to process
            force: Overwrite existing cache
            
        Returns:
            List of cached tensor paths
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.webp']
        
        input_path = Path(input_dir)
        cached_paths = []
        errors = []
        
        # Find all images
        image_files = []
        for ext in extensions:
            image_files.extend(input_path.rglob(f"*{ext}"))
            image_files.extend(input_path.rglob(f"*{ext.upper()}"))
        
        logger.info(f"Found {len(image_files)} images in {input_dir}")
        
        for i, image_path in enumerate(image_files):
            try:
                cache_path = self.process_and_cache(
                    str(image_path),
                    cache_dir,
                    mode,
                    force
                )
                cached_paths.append(cache_path)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1}/{len(image_files)} images")
                    
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                errors.append((str(image_path), str(e)))
        
        logger.info(f"Successfully processed {len(cached_paths)} images")
        if errors:
            logger.warning(f"Failed to process {len(errors)} images")
        
        return cached_paths


def main():
    """CLI for image preprocessing."""
    parser = argparse.ArgumentParser(description="Preprocess crop disease images")
    parser.add_argument("--input", "-i", required=True, help="Input image or directory")
    parser.add_argument("--output", "-o", default="./cache", help="Output cache directory")
    parser.add_argument("--config", "-c", help="Path to aug_config.yaml")
    parser.add_argument("--mode", "-m", choices=['train', 'val'], default='val')
    parser.add_argument("--force", "-f", action="store_true", help="Force reprocess")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = ImagePreprocessor(
        config_path=args.config,
        seed=args.seed
    )
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single image
        cache_path = preprocessor.process_and_cache(
            str(input_path),
            args.output,
            args.mode,
            args.force
        )
        print(f"Cached tensor: {cache_path}")
        
    elif input_path.is_dir():
        # Directory of images
        cached_paths = preprocessor.process_directory(
            str(input_path),
            args.output,
            args.mode,
            force=args.force
        )
        print(f"Processed {len(cached_paths)} images")
    
    else:
        print(f"Error: {args.input} does not exist")
        sys.exit(1)


if __name__ == "__main__":
    main()
