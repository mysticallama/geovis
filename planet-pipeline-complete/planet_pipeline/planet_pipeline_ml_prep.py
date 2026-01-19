"""
Machine Learning Data Preparation Module

Prepare Planet imagery for ML workflows including PyTorch, TensorFlow, and scikit-learn.
Supports image chipping, data augmentation, and train/val/test splitting.
"""

import numpy as np
import rasterio
from rasterio.windows import Window
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import logging
from dataclasses import dataclass
import random

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for ML dataset."""
    chip_size: int = 256
    overlap: int = 32
    normalize: bool = True
    augment: bool = False
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    random_seed: int = 42


class MLDataPrep:
    """Prepare imagery for machine learning models."""
    
    def __init__(self, storage_dir: Path):
        """
        Initialize ML data prep.
        
        Args:
            storage_dir: Root storage directory
        """
        self.storage_dir = Path(storage_dir)
    
    def prepare_dataset(
        self,
        aois: List[str],
        storage,
        model_type: str = "pytorch",
        output_format: str = "chips",
        chip_size: int = 256,
        overlap: int = 32,
        splits: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        normalize: bool = True,
        augment: bool = False,
        label_file: Optional[str] = None
    ) -> Path:
        """
        Prepare complete ML dataset.
        
        Args:
            aois: List of AOI names to include
            storage: Storage manager instance
            model_type: Target framework ("pytorch", "tensorflow", "sklearn")
            output_format: "chips", "patches", or "full"
            chip_size: Size of image chips
            overlap: Overlap between chips
            splits: (train, val, test) split ratios
            normalize: Apply normalization
            augment: Apply data augmentation
            label_file: Optional labels file
        
        Returns:
            Path to prepared dataset
        """
        dataset_name = f"{model_type}_{output_format}_{chip_size}"
        dataset_dir = storage.get_ml_dataset_dir(dataset_name)
        
        logger.info(f"Preparing {model_type} dataset at {dataset_dir}")
        
        # Collect all imagery files
        all_files = []
        for aoi_name in aois:
            files = storage.get_imagery_files(aoi_name)
            all_files.extend(files)
        
        logger.info(f"Found {len(all_files)} imagery files")
        
        # Load labels if provided
        labels = self._load_labels(label_file) if label_file else None
        
        # Split data
        train_files, val_files, test_files = self._split_data(
            all_files, splits, seed=42
        )
        
        # Process each split
        for split_name, split_files in [
            ("train", train_files),
            ("val", val_files),
            ("test", test_files)
        ]:
            split_dir = dataset_dir / split_name
            
            if output_format == "chips":
                self._create_chips(
                    files=split_files,
                    output_dir=split_dir,
                    chip_size=chip_size,
                    overlap=overlap,
                    normalize=normalize,
                    augment=(augment and split_name == "train"),
                    labels=labels
                )
            elif output_format == "patches":
                self._create_random_patches(
                    files=split_files,
                    output_dir=split_dir,
                    patch_size=chip_size,
                    num_patches_per_image=100,
                    normalize=normalize,
                    labels=labels
                )
            else:  # full
                self._prepare_full_images(
                    files=split_files,
                    output_dir=split_dir,
                    normalize=normalize,
                    labels=labels
                )
        
        # Create dataset metadata
        self._save_dataset_metadata(
            dataset_dir=dataset_dir,
            config={
                "model_type": model_type,
                "output_format": output_format,
                "chip_size": chip_size,
                "overlap": overlap,
                "normalize": normalize,
                "augment": augment,
                "splits": splits,
                "num_train": len(train_files),
                "num_val": len(val_files),
                "num_test": len(test_files),
                "aois": aois
            }
        )
        
        # Create framework-specific loaders
        if model_type == "pytorch":
            self._create_pytorch_loader(dataset_dir)
        elif model_type == "tensorflow":
            self._create_tensorflow_loader(dataset_dir)
        
        logger.info(f"Dataset preparation complete: {dataset_dir}")
        return dataset_dir
    
    def _split_data(
        self,
        files: List[Path],
        splits: Tuple[float, float, float],
        seed: int = 42
    ) -> Tuple[List[Path], List[Path], List[Path]]:
        """
        Split data into train/val/test sets.
        
        Args:
            files: List of file paths
            splits: (train, val, test) ratios
            seed: Random seed
        
        Returns:
            Tuple of (train_files, val_files, test_files)
        """
        random.seed(seed)
        files_shuffled = files.copy()
        random.shuffle(files_shuffled)
        
        train_ratio, val_ratio, test_ratio = splits
        n_total = len(files_shuffled)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_files = files_shuffled[:n_train]
        val_files = files_shuffled[n_train:n_train + n_val]
        test_files = files_shuffled[n_train + n_val:]
        
        logger.info(f"Split: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
        return train_files, val_files, test_files
    
    def _create_chips(
        self,
        files: List[Path],
        output_dir: Path,
        chip_size: int,
        overlap: int,
        normalize: bool,
        augment: bool,
        labels: Optional[Dict]
    ) -> None:
        """
        Create image chips from full imagery.
        
        Args:
            files: Input imagery files
            output_dir: Output directory
            chip_size: Size of chips
            overlap: Overlap between chips
            normalize: Apply normalization
            augment: Apply augmentation
            labels: Optional labels dict
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        chip_idx = 0
        
        for file_path in files:
            with rasterio.open(file_path) as src:
                height = src.height
                width = src.width
                
                stride = chip_size - overlap
                
                # Generate chip windows
                for y in range(0, height - chip_size + 1, stride):
                    for x in range(0, width - chip_size + 1, stride):
                        window = Window(x, y, chip_size, chip_size)
                        chip = src.read(window=window)
                        
                        # Skip chips with too many nodata values
                        if np.sum(chip == 0) / chip.size > 0.5:
                            continue
                        
                        # Normalize if requested
                        if normalize:
                            chip = self._normalize_chip(chip)
                        
                        # Save chip
                        chip_name = f"chip_{chip_idx:06d}.npy"
                        np.save(output_dir / chip_name, chip)
                        
                        # Save augmented versions if requested
                        if augment:
                            self._save_augmented_chips(
                                chip, output_dir, chip_idx
                            )
                        
                        chip_idx += 1
        
        logger.info(f"Created {chip_idx} chips in {output_dir}")
    
    def _create_random_patches(
        self,
        files: List[Path],
        output_dir: Path,
        patch_size: int,
        num_patches_per_image: int,
        normalize: bool,
        labels: Optional[Dict]
    ) -> None:
        """
        Create random patches from imagery.
        
        Args:
            files: Input imagery files
            output_dir: Output directory
            patch_size: Size of patches
            num_patches_per_image: Number of patches per image
            normalize: Apply normalization
            labels: Optional labels dict
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        patch_idx = 0
        
        for file_path in files:
            with rasterio.open(file_path) as src:
                height = src.height
                width = src.width
                
                for _ in range(num_patches_per_image):
                    # Random position
                    x = random.randint(0, max(0, width - patch_size))
                    y = random.randint(0, max(0, height - patch_size))
                    
                    window = Window(x, y, patch_size, patch_size)
                    patch = src.read(window=window)
                    
                    # Skip invalid patches
                    if np.sum(patch == 0) / patch.size > 0.3:
                        continue
                    
                    if normalize:
                        patch = self._normalize_chip(patch)
                    
                    patch_name = f"patch_{patch_idx:06d}.npy"
                    np.save(output_dir / patch_name, patch)
                    patch_idx += 1
        
        logger.info(f"Created {patch_idx} random patches in {output_dir}")
    
    def _prepare_full_images(
        self,
        files: List[Path],
        output_dir: Path,
        normalize: bool,
        labels: Optional[Dict]
    ) -> None:
        """
        Prepare full images for ML.
        
        Args:
            files: Input imagery files
            output_dir: Output directory
            normalize: Apply normalization
            labels: Optional labels dict
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, file_path in enumerate(files):
            with rasterio.open(file_path) as src:
                image = src.read()
            
            if normalize:
                image = self._normalize_chip(image)
            
            output_name = f"image_{idx:04d}.npy"
            np.save(output_dir / output_name, image)
        
        logger.info(f"Prepared {len(files)} full images in {output_dir}")
    
    def _normalize_chip(self, chip: np.ndarray) -> np.ndarray:
        """
        Normalize chip values.
        
        Args:
            chip: Input chip array
        
        Returns:
            Normalized chip
        """
        # Per-band normalization using 2-98 percentile
        normalized = np.zeros_like(chip, dtype=np.float32)
        
        for band_idx in range(chip.shape[0]):
            band = chip[band_idx].astype(np.float32)
            
            if np.sum(band > 0) == 0:
                continue
            
            p2 = np.percentile(band[band > 0], 2)
            p98 = np.percentile(band[band > 0], 98)
            
            if p98 > p2:
                normalized[band_idx] = np.clip(
                    (band - p2) / (p98 - p2), 0, 1
                )
            else:
                normalized[band_idx] = band
        
        return normalized
    
    def _save_augmented_chips(
        self,
        chip: np.ndarray,
        output_dir: Path,
        base_idx: int
    ) -> None:
        """
        Save augmented versions of chip.
        
        Args:
            chip: Input chip
            output_dir: Output directory
            base_idx: Base index for naming
        """
        augmentations = [
            ("flip_lr", np.fliplr),
            ("flip_ud", np.flipud),
            ("rot90", lambda x: np.rot90(x, k=1, axes=(1, 2))),
            ("rot180", lambda x: np.rot90(x, k=2, axes=(1, 2))),
            ("rot270", lambda x: np.rot90(x, k=3, axes=(1, 2)))
        ]
        
        for aug_name, aug_func in augmentations:
            augmented = aug_func(chip)
            output_name = f"chip_{base_idx:06d}_{aug_name}.npy"
            np.save(output_dir / output_name, augmented)
    
    def _load_labels(self, label_file: str) -> Dict:
        """
        Load labels from file.
        
        Args:
            label_file: Path to labels file (JSON or CSV)
        
        Returns:
            Dict of labels
        """
        label_path = Path(label_file)
        
        if label_path.suffix == ".json":
            with open(label_path, 'r') as f:
                return json.load(f)
        elif label_path.suffix == ".csv":
            import csv
            labels = {}
            with open(label_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    labels[row['id']] = row
            return labels
        
        return {}
    
    def _save_dataset_metadata(
        self,
        dataset_dir: Path,
        config: Dict
    ) -> None:
        """
        Save dataset metadata.
        
        Args:
            dataset_dir: Dataset directory
            config: Configuration dict
        """
        metadata_path = dataset_dir / "dataset_info.json"
        
        with open(metadata_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved dataset metadata to {metadata_path}")
    
    def _create_pytorch_loader(self, dataset_dir: Path) -> None:
        """
        Create PyTorch dataset loader script.
        
        Args:
            dataset_dir: Dataset directory
        """
        loader_code = '''
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path

class PlanetDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.files = sorted(list(self.data_dir.glob("*.npy")))
        self.transform = transform
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # Load chip
        chip = np.load(self.files[idx])
        chip = torch.from_numpy(chip).float()
        
        if self.transform:
            chip = self.transform(chip)
        
        # Placeholder label (replace with actual labels)
        label = 0
        
        return chip, label

# Usage example:
# train_dataset = PlanetDataset("train")
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
'''
        
        with open(dataset_dir / "pytorch_loader.py", 'w') as f:
            f.write(loader_code)
        
        logger.info("Created PyTorch loader script")
    
    def _create_tensorflow_loader(self, dataset_dir: Path) -> None:
        """
        Create TensorFlow dataset loader script.
        
        Args:
            dataset_dir: Dataset directory
        """
        loader_code = '''
import tensorflow as tf
import numpy as np
from pathlib import Path

def load_chip(file_path):
    chip = np.load(file_path.numpy())
    return chip.astype(np.float32)

def create_dataset(data_dir, batch_size=32):
    data_dir = Path(data_dir)
    files = sorted(list(data_dir.glob("*.npy")))
    file_paths = [str(f) for f in files]
    
    dataset = tf.data.Dataset.from_tensor_slices(file_paths)
    
    def load_wrapper(file_path):
        chip = tf.py_function(load_chip, [file_path], tf.float32)
        # Placeholder label
        label = 0
        return chip, label
    
    dataset = dataset.map(load_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

# Usage example:
# train_dataset = create_dataset("train", batch_size=32)
'''
        
        with open(dataset_dir / "tensorflow_loader.py", 'w') as f:
            f.write(loader_code)
        
        logger.info("Created TensorFlow loader script")
