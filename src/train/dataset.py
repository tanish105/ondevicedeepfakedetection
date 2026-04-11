from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from src.common.constants import PROJECT_ROOT
from src.common.csv_io import read_rows
from src.common.schemas import FrameManifestRow


class FrameDataset(Dataset):
    """Dataset for frame-level training from manifests."""

    def __init__(
        self,
        task_id: str,
        split: str,
        manifest_path: Path | None = None,
        image_size: int = 224,
        augment: bool = False,
    ) -> None:
        """
        Args:
            task_id: Task ID (e.g., 'mobilenetv2_df_vs_real')
            split: 'train', 'val', or 'test'
            manifest_path: Path to manifest CSV. If None, uses aggregate manifest.
            image_size: Target image size (224x224 for MobileNetV2)
            augment: Whether to apply augmentation (train only)
        """
        self.task_id = task_id
        self.split = split
        self.image_size = image_size
        self.augment = augment

        # Load manifest
        if manifest_path is None:
            from src.data.ffpp_preprocess import aggregate_manifest_output_path

            manifest_path = aggregate_manifest_output_path(task_id)

        rows = read_rows(manifest_path, FrameManifestRow)
        self.rows = [row for row in rows if row.split == split and row.face_found == 1]

        # ImageNet normalization
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        # Augmentation (train only)
        self.augment_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.rows[idx]
        image_path = PROJECT_ROOT / row.image_path

        if not image_path.exists():
            raise FileNotFoundError(f"Missing image: {image_path}")

        img = Image.open(image_path).convert("RGB")

        # Apply augmentation if training
        if self.augment:
            img = self.augment_transform(img)
        else:
            img = self.transform(img)

        label = row.binary_label
        return img, label


def create_dataloaders(
    task_id: str,
    batch_size: int = 32,
    num_workers: int = 0,
    augment: bool = True,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train, val, and test dataloaders."""
    train_dataset = FrameDataset(task_id, "train", augment=augment and True)
    val_dataset = FrameDataset(task_id, "val", augment=False)
    test_dataset = FrameDataset(task_id, "test", augment=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader
