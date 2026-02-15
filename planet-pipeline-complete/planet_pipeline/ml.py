"""
Machine Learning data preparation module.

Prepare imagery for ML workflows: chipping, splitting, augmentation, and framework loaders.
Supports PyTorch, TensorFlow, and scikit-learn pipelines.
"""

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import rasterio
from rasterio.windows import Window

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DatasetConfig:
    """ML dataset configuration."""
    chip_size: int = 256
    overlap: int = 32
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    normalize: bool = True
    augment: bool = False
    seed: int = 42


# =============================================================================
# Core Functions
# =============================================================================

def normalize_chip(chip: np.ndarray, method: str = "percentile") -> np.ndarray:
    """
    Normalize chip per-band.

    Args:
        chip: Array (bands, height, width)
        method: "percentile" (2-98), "minmax", or "zscore"

    Returns:
        Normalized array
    """
    result = np.zeros_like(chip, dtype=np.float32)

    for i in range(chip.shape[0]):
        band = chip[i].astype(np.float32)
        valid = band[band > 0] if np.any(band > 0) else band.flatten()

        if len(valid) == 0:
            continue

        if method == "percentile":
            p2, p98 = np.percentile(valid, [2, 98])
            if p98 > p2:
                result[i] = np.clip((band - p2) / (p98 - p2), 0, 1)
        elif method == "minmax":
            vmin, vmax = valid.min(), valid.max()
            if vmax > vmin:
                result[i] = (band - vmin) / (vmax - vmin)
        elif method == "zscore":
            mean, std = valid.mean(), valid.std()
            if std > 0:
                result[i] = (band - mean) / std

    return result


def augment_chip(chip: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    """
    Generate augmented versions of a chip.

    Returns:
        List of (name, augmented_chip) tuples
    """
    return [
        ("flip_lr", np.flip(chip, axis=2)),
        ("flip_ud", np.flip(chip, axis=1)),
        ("rot90", np.rot90(chip, k=1, axes=(1, 2))),
        ("rot180", np.rot90(chip, k=2, axes=(1, 2))),
        ("rot270", np.rot90(chip, k=3, axes=(1, 2))),
    ]


def split_files(
    files: List[Path],
    splits: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42
) -> Tuple[List[Path], List[Path], List[Path]]:
    """Split files into train/val/test sets."""
    random.seed(seed)
    shuffled = files.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * splits[0])
    n_val = int(n * splits[1])

    return shuffled[:n_train], shuffled[n_train:n_train+n_val], shuffled[n_train+n_val:]


# =============================================================================
# Dataset Creator
# =============================================================================

class DatasetCreator:
    """Create ML-ready datasets from satellite imagery."""

    def __init__(self, config: Optional[DatasetConfig] = None):
        self.config = config or DatasetConfig()

    def create_chips(
        self,
        imagery_files: List[Path],
        output_dir: Union[str, Path],
        labels_dir: Optional[Union[str, Path]] = None
    ) -> Path:
        """
        Create tiled chips from imagery.

        Args:
            imagery_files: List of input GeoTIFF files
            output_dir: Output directory for dataset
            labels_dir: Optional directory containing label masks

        Returns:
            Path to dataset directory
        """
        output_dir = Path(output_dir)

        # Split files
        train_files, val_files, test_files = split_files(
            imagery_files,
            (self.config.train_split, self.config.val_split, self.config.test_split),
            self.config.seed
        )

        # Process each split
        for split_name, files in [("train", train_files), ("val", val_files), ("test", test_files)]:
            split_dir = output_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)

            chip_idx = 0
            for file_path in files:
                chips = self._extract_chips(file_path)

                for chip in chips:
                    if self._is_valid_chip(chip):
                        if self.config.normalize:
                            chip = normalize_chip(chip)

                        np.save(split_dir / f"chip_{chip_idx:06d}.npy", chip)
                        chip_idx += 1

                        # Augment training data
                        if self.config.augment and split_name == "train":
                            for aug_name, aug_chip in augment_chip(chip):
                                np.save(split_dir / f"chip_{chip_idx:06d}_{aug_name}.npy", aug_chip)
                                chip_idx += 1

            logger.info(f"{split_name}: {chip_idx} chips")

        # Save metadata
        self._save_metadata(output_dir, len(train_files), len(val_files), len(test_files))

        # Create loaders
        self._create_pytorch_loader(output_dir)
        self._create_tensorflow_loader(output_dir)

        logger.info(f"Dataset created: {output_dir}")
        return output_dir

    def create_patches(
        self,
        imagery_files: List[Path],
        output_dir: Union[str, Path],
        patches_per_image: int = 100
    ) -> Path:
        """Create random patches from imagery."""
        output_dir = Path(output_dir)
        train_files, val_files, test_files = split_files(
            imagery_files,
            (self.config.train_split, self.config.val_split, self.config.test_split),
            self.config.seed
        )

        for split_name, files in [("train", train_files), ("val", val_files), ("test", test_files)]:
            split_dir = output_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)

            patch_idx = 0
            for file_path in files:
                patches = self._extract_random_patches(file_path, patches_per_image)

                for patch in patches:
                    if self._is_valid_chip(patch, threshold=0.3):
                        if self.config.normalize:
                            patch = normalize_chip(patch)
                        np.save(split_dir / f"patch_{patch_idx:06d}.npy", patch)
                        patch_idx += 1

            logger.info(f"{split_name}: {patch_idx} patches")

        self._save_metadata(output_dir, len(train_files), len(val_files), len(test_files))
        self._create_pytorch_loader(output_dir)
        return output_dir

    def _extract_chips(self, file_path: Path) -> List[np.ndarray]:
        """Extract overlapping chips from an image."""
        chips = []
        stride = self.config.chip_size - self.config.overlap

        with rasterio.open(file_path) as src:
            h, w = src.height, src.width

            for y in range(0, h - self.config.chip_size + 1, stride):
                for x in range(0, w - self.config.chip_size + 1, stride):
                    window = Window(x, y, self.config.chip_size, self.config.chip_size)
                    chip = src.read(window=window)
                    chips.append(chip)

        return chips

    def _extract_random_patches(self, file_path: Path, n: int) -> List[np.ndarray]:
        """Extract random patches from an image."""
        patches = []

        with rasterio.open(file_path) as src:
            h, w = src.height, src.width

            for _ in range(n):
                x = random.randint(0, max(0, w - self.config.chip_size))
                y = random.randint(0, max(0, h - self.config.chip_size))
                window = Window(x, y, self.config.chip_size, self.config.chip_size)
                patches.append(src.read(window=window))

        return patches

    def _is_valid_chip(self, chip: np.ndarray, threshold: float = 0.5) -> bool:
        """Check if chip has enough valid pixels."""
        return np.sum(chip == 0) / chip.size < threshold

    def _save_metadata(self, output_dir: Path, n_train: int, n_val: int, n_test: int):
        """Save dataset metadata."""
        metadata = {
            "chip_size": self.config.chip_size,
            "overlap": self.config.overlap,
            "normalize": self.config.normalize,
            "augment": self.config.augment,
            "splits": {
                "train": n_train,
                "val": n_val,
                "test": n_test
            }
        }
        with open(output_dir / "dataset_info.json", 'w') as f:
            json.dump(metadata, f, indent=2)

    def _create_pytorch_loader(self, output_dir: Path):
        """Create PyTorch DataLoader script."""
        code = '''"""PyTorch DataLoader for Planet imagery chips."""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path


class PlanetDataset(Dataset):
    """PyTorch Dataset for Planet imagery chips."""

    def __init__(self, data_dir: str, transform=None):
        self.files = sorted(Path(data_dir).glob("*.npy"))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        chip = np.load(self.files[idx])
        chip = torch.from_numpy(chip).float()

        if self.transform:
            chip = self.transform(chip)

        # Replace with actual label loading
        label = 0
        return chip, label


def create_loaders(data_dir: str, batch_size: int = 32, num_workers: int = 4):
    """Create train/val/test DataLoaders."""
    loaders = {}
    for split in ["train", "val", "test"]:
        split_dir = Path(data_dir) / split
        if split_dir.exists():
            dataset = PlanetDataset(str(split_dir))
            loaders[split] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split == "train"),
                num_workers=num_workers
            )
    return loaders


# Usage:
# loaders = create_loaders("./dataset")
# for batch, labels in loaders["train"]:
#     ...
'''
        with open(output_dir / "pytorch_loader.py", 'w') as f:
            f.write(code)

    def _create_tensorflow_loader(self, output_dir: Path):
        """Create TensorFlow dataset script."""
        code = '''"""TensorFlow Dataset for Planet imagery chips."""
import tensorflow as tf
import numpy as np
from pathlib import Path


def load_chip(file_path):
    """Load a single chip."""
    chip = np.load(file_path.numpy().decode())
    return chip.astype(np.float32)


def create_dataset(data_dir: str, batch_size: int = 32):
    """Create TensorFlow dataset from chips directory."""
    files = sorted([str(f) for f in Path(data_dir).glob("*.npy")])

    dataset = tf.data.Dataset.from_tensor_slices(files)

    def load_fn(path):
        chip = tf.py_function(load_chip, [path], tf.float32)
        label = 0  # Replace with actual label
        return chip, label

    dataset = dataset.map(load_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def create_datasets(data_dir: str, batch_size: int = 32):
    """Create train/val/test datasets."""
    datasets = {}
    for split in ["train", "val", "test"]:
        split_dir = Path(data_dir) / split
        if split_dir.exists():
            datasets[split] = create_dataset(str(split_dir), batch_size)
    return datasets


# Usage:
# datasets = create_datasets("./dataset")
# for batch, labels in datasets["train"]:
#     ...
'''
        with open(output_dir / "tensorflow_loader.py", 'w') as f:
            f.write(code)


# =============================================================================
# Label Integration
# =============================================================================

class LabelManager:
    """Manage labels for ML datasets."""

    @staticmethod
    def load_yolo_labels(label_dir: Union[str, Path]) -> Dict[str, List[List[float]]]:
        """Load YOLO format labels."""
        labels = {}
        for label_file in Path(label_dir).glob("*.txt"):
            boxes = []
            with open(label_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        boxes.append([float(x) for x in parts])
            labels[label_file.stem] = boxes
        return labels

    @staticmethod
    def load_mask_labels(label_dir: Union[str, Path]) -> Dict[str, Path]:
        """Load segmentation mask paths."""
        masks = {}
        for mask_file in Path(label_dir).glob("*.tif"):
            masks[mask_file.stem] = mask_file
        for mask_file in Path(label_dir).glob("*.npy"):
            masks[mask_file.stem] = mask_file
        return masks

    @staticmethod
    def create_chip_with_label(
        imagery_path: Path,
        label_path: Path,
        window: Window
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract aligned chip and label."""
        with rasterio.open(imagery_path) as src:
            chip = src.read(window=window)

        # Load label (supports GeoTIFF or numpy)
        if label_path.suffix == ".npy":
            label = np.load(label_path)
            # Extract window from label
            label = label[
                int(window.row_off):int(window.row_off + window.height),
                int(window.col_off):int(window.col_off + window.width)
            ]
        else:
            with rasterio.open(label_path) as src:
                label = src.read(1, window=window)

        return chip, label


# =============================================================================
# Convenience Functions
# =============================================================================

def prepare_dataset(
    imagery_files: List[Path],
    output_dir: Union[str, Path],
    chip_size: int = 256,
    normalize: bool = True,
    augment: bool = False
) -> Path:
    """
    Prepare ML dataset from imagery files.

    Args:
        imagery_files: List of input GeoTIFF files
        output_dir: Output directory
        chip_size: Size of chips in pixels
        normalize: Apply normalization
        augment: Apply augmentation to training data

    Returns:
        Path to dataset directory
    """
    config = DatasetConfig(chip_size=chip_size, normalize=normalize, augment=augment)
    creator = DatasetCreator(config)
    return creator.create_chips(imagery_files, output_dir)


def prepare_detection_dataset(
    imagery_dir: Union[str, Path],
    labels_dir: Union[str, Path],
    output_dir: Union[str, Path],
    chip_size: int = 640
) -> Path:
    """
    Prepare object detection dataset (YOLOv8 compatible).

    Args:
        imagery_dir: Directory with imagery files
        labels_dir: Directory with YOLO format labels
        output_dir: Output directory

    Returns:
        Path to dataset directory
    """
    output_dir = Path(output_dir)

    # Create YOLO directory structure
    for split in ["train", "val", "test"]:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    imagery_files = sorted(Path(imagery_dir).glob("*.tif"))
    train_files, val_files, test_files = split_files(imagery_files)

    # Copy files to splits
    import shutil
    for split_name, files in [("train", train_files), ("val", val_files), ("test", test_files)]:
        for img_file in files:
            # Copy image
            shutil.copy(img_file, output_dir / split_name / "images" / img_file.name)

            # Copy label if exists
            label_file = Path(labels_dir) / f"{img_file.stem}.txt"
            if label_file.exists():
                shutil.copy(label_file, output_dir / split_name / "labels" / label_file.name)

    # Create data.yaml for YOLOv8
    yaml_content = f"""
path: {output_dir.absolute()}
train: train/images
val: val/images
test: test/images

names:
  0: object
"""
    with open(output_dir / "data.yaml", 'w') as f:
        f.write(yaml_content)

    logger.info(f"Detection dataset created: {output_dir}")
    return output_dir


def prepare_segmentation_dataset(
    imagery_files: List[Path],
    mask_files: List[Path],
    output_dir: Union[str, Path],
    chip_size: int = 256
) -> Path:
    """
    Prepare semantic segmentation dataset.

    Args:
        imagery_files: List of imagery files
        mask_files: List of corresponding mask files
        output_dir: Output directory
        chip_size: Chip size

    Returns:
        Path to dataset directory
    """
    output_dir = Path(output_dir)
    config = DatasetConfig(chip_size=chip_size)

    # Pair imagery with masks
    paired = list(zip(imagery_files, mask_files))
    random.seed(config.seed)
    random.shuffle(paired)

    n = len(paired)
    n_train = int(n * config.train_split)
    n_val = int(n * config.val_split)

    splits = {
        "train": paired[:n_train],
        "val": paired[n_train:n_train+n_val],
        "test": paired[n_train+n_val:]
    }

    for split_name, pairs in splits.items():
        img_dir = output_dir / split_name / "images"
        mask_dir = output_dir / split_name / "masks"
        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        chip_idx = 0
        stride = chip_size - config.overlap

        for img_path, mask_path in pairs:
            with rasterio.open(img_path) as src:
                h, w = src.height, src.width

                for y in range(0, h - chip_size + 1, stride):
                    for x in range(0, w - chip_size + 1, stride):
                        window = Window(x, y, chip_size, chip_size)

                        # Read image chip
                        img_chip = src.read(window=window)
                        if np.sum(img_chip == 0) / img_chip.size > 0.5:
                            continue

                        # Read mask chip
                        with rasterio.open(mask_path) as mask_src:
                            mask_chip = mask_src.read(1, window=window)

                        if config.normalize:
                            img_chip = normalize_chip(img_chip)

                        np.save(img_dir / f"chip_{chip_idx:06d}.npy", img_chip)
                        np.save(mask_dir / f"chip_{chip_idx:06d}.npy", mask_chip)
                        chip_idx += 1

        logger.info(f"{split_name}: {chip_idx} chip pairs")

    logger.info(f"Segmentation dataset created: {output_dir}")
    return output_dir
