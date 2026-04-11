from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm


@dataclass(frozen=True)
class TrainingConfig:
    task_id: str
    device: str = "auto"
    num_epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    patience: int = 5
    checkpoint_dir: Path = Path("artifacts/checkpoints")


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    loss: float
    accuracy: float
    auc: float


class ModelTrainer:
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        runtime: object,
    ) -> None:
        self.model = model
        self.config = config
        self.runtime = runtime
        self.device = runtime.resolved_device
        self.model.to(self.device)

    def train_epoch(self, train_loader) -> EpochMetrics:
        """Train for one epoch."""
        self.model.train()
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        total_loss = 0.0
        all_preds = []
        all_labels = []

        with tqdm(train_loader, desc="Training", unit="batch") as pbar:
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.float().unsqueeze(1).to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                all_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
                all_labels.extend(labels.detach().cpu().numpy())

                pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(train_loader)
        all_preds = np.array(all_preds).flatten()
        all_labels = np.array(all_labels).flatten()
        accuracy = np.mean((all_preds > 0.5) == all_labels)
        auc = roc_auc_score(all_labels, all_preds)

        return EpochMetrics(epoch=0, loss=avg_loss, accuracy=accuracy, auc=auc)

    def validate(self, val_loader, epoch: int) -> EpochMetrics:
        """Validate on validation set."""
        self.model.eval()
        criterion = nn.BCEWithLogitsLoss()

        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch} Validation", unit="batch"):
                images = images.to(self.device)
                labels = labels.float().unsqueeze(1).to(self.device)

                outputs = self.model(images)
                loss = criterion(outputs, labels)

                total_loss += loss.item()
                all_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
                all_labels.extend(labels.detach().cpu().numpy())

        avg_loss = total_loss / len(val_loader)
        all_preds = np.array(all_preds).flatten()
        all_labels = np.array(all_labels).flatten()
        accuracy = np.mean((all_preds > 0.5) == all_labels)
        auc = roc_auc_score(all_labels, all_preds)

        return EpochMetrics(epoch=epoch, loss=avg_loss, accuracy=accuracy, auc=auc)

    def train(
        self,
        train_loader,
        val_loader,
    ) -> dict:
        """Full training loop with early stopping."""
        best_auc = 0.0
        patience_counter = 0
        checkpoint_path = (
            self.config.checkpoint_dir / f"{self.config.task_id}_best.pth"
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
        criterion = nn.BCEWithLogitsLoss()

        train_history = []
        val_history = []

        for epoch in range(self.config.num_epochs):
            print(f"\n=== Epoch {epoch + 1}/{self.config.num_epochs} ===")

            # Train
            self.model.train()
            total_loss = 0.0
            all_preds = []
            all_labels = []

            with tqdm(train_loader, desc="Training", unit="batch") as pbar:
                for images, labels in pbar:
                    images = images.to(self.device)
                    labels = labels.float().unsqueeze(1).to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)

                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    all_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
                    all_labels.extend(labels.detach().cpu().numpy())

                    pbar.set_postfix({"loss": loss.item()})

            avg_train_loss = total_loss / len(train_loader)
            all_preds_np = np.array(all_preds).flatten()
            all_labels_np = np.array(all_labels).flatten()
            train_accuracy = np.mean((all_preds_np > 0.5) == all_labels_np)
            train_auc = roc_auc_score(all_labels_np, all_preds_np)
            train_history.append(
                EpochMetrics(epoch=epoch, loss=avg_train_loss, accuracy=train_accuracy, auc=train_auc)
            )
            print(
                f"Train Loss: {avg_train_loss:.4f} | Accuracy: {train_accuracy:.4f} | AUC: {train_auc:.4f}"
            )

            # Validate
            self.model.eval()
            total_loss = 0.0
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc=f"Validation", unit="batch"):
                    images = images.to(self.device)
                    labels = labels.float().unsqueeze(1).to(self.device)

                    outputs = self.model(images)
                    loss = criterion(outputs, labels)

                    total_loss += loss.item()
                    all_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
                    all_labels.extend(labels.detach().cpu().numpy())

            avg_val_loss = total_loss / len(val_loader)
            all_preds_np = np.array(all_preds).flatten()
            all_labels_np = np.array(all_labels).flatten()
            val_accuracy = np.mean((all_preds_np > 0.5) == all_labels_np)
            val_auc = roc_auc_score(all_labels_np, all_preds_np)
            val_history.append(
                EpochMetrics(epoch=epoch, loss=avg_val_loss, accuracy=val_accuracy, auc=val_auc)
            )
            print(
                f"Val Loss: {avg_val_loss:.4f} | Accuracy: {val_accuracy:.4f} | AUC: {val_auc:.4f}"
            )

            # Learning rate scheduling
            scheduler.step(val_auc)

            # Early stopping and checkpointing
            if val_auc > best_auc:
                best_auc = val_auc
                patience_counter = 0
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"✓ New best AUC: {best_auc:.4f} | Saved checkpoint: {checkpoint_path}")
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    print(f"\nEarly stopping after {epoch + 1} epochs.")
                    break

        # Load best model
        self.model.load_state_dict(torch.load(checkpoint_path))
        print(f"\nLoaded best model from {checkpoint_path}")

        return {
            "task_id": self.config.task_id,
            "best_auc": best_auc,
            "final_epoch": epoch,
            "checkpoint_path": str(checkpoint_path),
            "train_history": train_history,
            "val_history": val_history,
        }
