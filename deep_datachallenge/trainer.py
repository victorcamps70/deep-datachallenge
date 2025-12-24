"""
Trainer générique réutilisable pour tous les modèles de segmentation
"""

import torch
import torch.nn as nn
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np

from deep_datachallenge.metrics import MetricsTracker


class SegmentationTrainer:
    """
    Trainer générique pour l'entraînement de modèles de segmentation

    Avantages:
    - Réutilisable pour tous les modèles (U-Net, FCN, DeepLabV3+, etc.)
    - Gestion automatique des poids de classes
    - Logging complet des métriques
    - Early stopping basé sur la métrique de validation
    - Sauvegarde du meilleur modèle
    """

    def __init__(self, model, device, lr=1e-3, class_weights=None):
        """
        Initialiser le trainer

        Args:
            model (nn.Module): Le modèle à entraîner
            device (torch.device): Device (CPU ou GPU)
            lr (float): Learning rate
            class_weights (torch.Tensor): Poids des classes pour l'équilibrage
        """

        self.model = model.to(device)
        self.device = device
        self.lr = lr

        # Loss avec poids pour équilibrer les classes
        if class_weights is not None:
            class_weights = class_weights.to(device)

        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Optimizer
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)

        # Historique
        self.history = {
            "train_loss": [],
            "train_iou": [],
            "val_loss": [],
            "val_iou": [],
        }

        # Pour l'early stopping
        self.best_val_iou = -1
        self.patience_counter = 0

    def train_epoch(self, train_loader):
        """
        Entraîner une époque

        Args:
            train_loader (DataLoader): DataLoader d'entraînement

        Returns:
            dict: Métriques d'entraînement
        """

        self.model.train()
        tracker = MetricsTracker()

        with tqdm(train_loader, desc="Training") as pbar:
            for batch in pbar:
                # Charger les données
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)

                # Forward pass
                self.optimizer.zero_grad()
                logits = self.model(images)
                loss = self.criterion(logits, masks)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                # Mettre à jour les métriques
                with torch.no_grad():
                    pred = torch.argmax(logits, dim=1)
                    tracker.update(pred, masks, loss.item())

                pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        metrics = tracker.get_metrics()
        return metrics

    def validate(self, val_loader):
        """
        Valider sur l'ensemble de validation

        Args:
            val_loader (DataLoader): DataLoader de validation

        Returns:
            dict: Métriques de validation
        """

        self.model.eval()
        tracker = MetricsTracker()

        with torch.no_grad():
            with tqdm(val_loader, desc="Validation") as pbar:
                for batch in pbar:
                    images = batch["image"].to(self.device)
                    masks = batch["mask"].to(self.device)

                    logits = self.model(images)
                    loss = self.criterion(logits, masks)

                    pred = torch.argmax(logits, dim=1)
                    tracker.update(pred, masks, loss.item())

                    pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        metrics = tracker.get_metrics()
        return metrics

    def save_checkpoint(self, save_dir, model_name, epoch):
        """
        Sauvegarder l'état complet du training (checkpoint)

        Args:
            save_dir (Path): Répertoire de sauvegarde
            model_name (str): Nom du modèle
            epoch (int): Numéro d'époque actuelle
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = save_dir / f"{model_name}_checkpoint.pt"

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "history": self.history,
            "best_val_iou": self.best_val_iou,
            "patience_counter": self.patience_counter,
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"✓ Checkpoint sauvegardé: {checkpoint_path} (époque {epoch+1})")

    def load_checkpoint(self, save_dir, model_name):
        """
        Charger un checkpoint pour reprendre l'entraînement

        Args:
            save_dir (Path): Répertoire contenant le checkpoint
            model_name (str): Nom du modèle

        Returns:
            int: Numéro de l'époque à partir de laquelle reprendre (ou -1 si pas de checkpoint)
        """
        save_dir = Path(save_dir)
        checkpoint_path = save_dir / f"{model_name}_checkpoint.pt"

        if not checkpoint_path.exists():
            print(f"Aucun checkpoint trouvé: {checkpoint_path}")
            return -1

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.history = checkpoint["history"]
        self.best_val_iou = checkpoint["best_val_iou"]
        self.patience_counter = checkpoint["patience_counter"]

        epoch_start = checkpoint["epoch"]
        print(f"✓ Checkpoint chargé: {checkpoint_path}")
        print(f"  Reprise à partir de l'époque {epoch_start + 1}")
        print(f"  Meilleur IoU actuel: {self.best_val_iou:.4f}")

        return epoch_start

    def fit(
        self,
        train_loader,
        val_loader,
        epochs=50,
        early_stopping_patience=10,
        save_dir=None,
        model_name="model",
        resume=False,
    ):
        """
        Entraîner le modèle pour plusieurs époque avec early stopping

        Args:
            train_loader (DataLoader): DataLoader d'entraînement
            val_loader (DataLoader): DataLoader de validation
            epochs (int): Nombre d'époque
            early_stopping_patience (int): Nombre d'époque sans amélioration avant stop
            save_dir (Path): Répertoire pour sauvegarder le modèle
            model_name (str): Nom du modèle pour la sauvegarde
            resume (bool): Si True, reprendre depuis un checkpoint existant

        Returns:
            dict: Historique complet de l'entraînement
        """

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        # Charger un checkpoint si resume=True
        start_epoch = 0
        if resume and save_dir:
            start_epoch = self.load_checkpoint(save_dir, model_name)
            if start_epoch >= 0:
                start_epoch += 1  # Reprendre à partir de l'époque suivante
            else:
                start_epoch = 0

        print("\n" + "=" * 70)
        print(f"ENTRAÎNEMENT DU MODÈLE: {model_name}")
        if resume and start_epoch > 0:
            print(f"MODE REPRISE - À partir de l'époque {start_epoch + 1}")
        print("=" * 70)

        for epoch in range(start_epoch, epochs):
            print(f"\n[Epoch {epoch+1}/{epochs}]")

            # Entraîner
            train_metrics = self.train_epoch(train_loader)

            # Valider
            val_metrics = self.validate(val_loader)

            # Afficher les résultats
            print(f"Train Loss: {train_metrics['loss']:.6f} | IoU: {train_metrics['iou_mean']:.4f}")
            print(f"Val Loss:   {val_metrics['loss']:.6f} | IoU: {val_metrics['iou_mean']:.4f}")

            # Enregistrer dans l'historique
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_iou"].append(train_metrics["iou_mean"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_iou"].append(val_metrics["iou_mean"])

            # Early stopping
            if val_metrics["iou_mean"] > self.best_val_iou:
                self.best_val_iou = val_metrics["iou_mean"]
                self.patience_counter = 0

                # Sauvegarder le meilleur modèle
                if save_dir:
                    save_path = save_dir / f"{model_name}_best.pt"
                    torch.save(self.model.state_dict(), save_path)
                    print(f"✓ Meilleur modèle sauvegardé: {save_path}")
            else:
                self.patience_counter += 1

                if self.patience_counter >= early_stopping_patience:
                    print(
                        f"\nEarly stopping: pas d'amélioration depuis {early_stopping_patience} époque"
                    )
                    break

            # Sauvegarder un checkpoint tous les 5 epochs pour la reprise
            if save_dir and (epoch + 1) % 5 == 0:
                self.save_checkpoint(save_dir, model_name, epoch)

            # Mettre à jour le learning rate
            self.scheduler.step()

        # Sauvegarder l'historique
        if save_dir:
            history_path = save_dir / f"{model_name}_history.json"
            with open(history_path, "w") as f:
                json.dump(self.history, f, indent=2)
            print(f"Historique sauvegardé: {history_path}")

        print("\n" + "=" * 70)
        print("ENTRAÎNEMENT TERMINÉ")
        print("=" * 70)

        return self.history

    def load_best_model(self, save_dir, model_name):
        """
        Charger le meilleur modèle sauvegardé

        Args:
            save_dir (Path): Répertoire contenant le modèle
            model_name (str): Nom du modèle
        """

        save_dir = Path(save_dir)
        save_path = save_dir / f"{model_name}_best.pt"

        if save_path.exists():
            self.model.load_state_dict(torch.load(save_path, map_location=self.device))
            print(f"Modèle chargé: {save_path}")
        else:
            print(f"Fichier non trouvé: {save_path}")

    def predict(self, image_tensor):
        """
        Effectuer une prédiction sur une image

        Args:
            image_tensor (torch.Tensor): Image [1, H, W] ou [B, H, W]

        Returns:
            torch.Tensor: Masque prédit [H, W] ou [B, H, W]
        """

        self.model.eval()

        with torch.no_grad():
            if len(image_tensor.shape) == 3:
                image_tensor = image_tensor.unsqueeze(1)  # [B, H, W] → [B, 1, H, W]

            image_tensor = image_tensor.to(self.device)
            logits = self.model(image_tensor)
            pred = torch.argmax(logits, dim=1)

        return pred


if __name__ == "__main__":
    print("=" * 70)
    print("TEST TRAINER")
    print("=" * 70)

    # Imports
    import pandas as pd
    from pathlib import Path
    from deep_datachallenge.models.unet import UNet
    from deep_datachallenge.dataset import create_dataloaders
    from deep_datachallenge.preprocessing import ImagePreprocessor

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Charger les données
    data_dir = Path("data")
    y_train_file = data_dir / "y_train_labels" / "Y_train_T9NrBYo.csv"
    x_train_dir = data_dir / "x_train_images"

    y_train = pd.read_csv(y_train_file, index_col=0)
    preprocessor = ImagePreprocessor(target_size=(160, 160))

    # Créer les DataLoaders
    train_loader, val_loader, class_weights = create_dataloaders(
        y_train, x_train_dir, preprocessor, batch_size=8, augment_train=True
    )

    # Créer le modèle
    model = UNet(in_channels=1, out_channels=3, depth=4)

    # Créer le trainer
    trainer = SegmentationTrainer(model, device, lr=1e-3, class_weights=class_weights)

    # Entraîner pour 2 époque (test rapide)
    history = trainer.fit(
        train_loader,
        val_loader,
        epochs=2,
        early_stopping_patience=1,
        save_dir="checkpoints",
        model_name="unet_test",
    )

    print("\nTrainer test réussi!")
