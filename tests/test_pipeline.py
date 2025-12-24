"""
Tests pytest pour le pipeline complet
Exécution: pytest test_pipeline.py -v
"""

import pytest
import torch
import pandas as pd
from pathlib import Path
import numpy as np


class TestImports:
    """Tests d'imports des modules"""

    def test_import_models(self):
        """Vérifier l'import des modèles"""
        from deep_datachallenge.models.unet import UNet

        assert UNet is not None

    def test_import_dataset(self):
        """Vérifier l'import des données"""
        from deep_datachallenge.dataset import create_dataloaders
        from deep_datachallenge.preprocessing import ImagePreprocessor

        assert create_dataloaders is not None
        assert ImagePreprocessor is not None

    def test_import_trainer(self):
        """Vérifier l'import du trainer"""
        from deep_datachallenge.trainer import SegmentationTrainer
        from deep_datachallenge.metrics import MetricsTracker

        assert SegmentationTrainer is not None
        assert MetricsTracker is not None


class TestDataLoading:
    """Tests de chargement des données"""

    def test_csv_exists(self):
        """Vérifier que le fichier CSV existe"""
        y_train_file = Path("data/y_train_labels/Y_train_T9NrBYo.csv")
        assert y_train_file.exists(), f"Fichier non trouvé: {y_train_file}"

    def test_images_dir_exists(self):
        """Vérifier que le répertoire images existe"""
        x_train_dir = Path("data/x_train_images")
        assert x_train_dir.exists(), f"Répertoire non trouvé: {x_train_dir}"

    def test_load_csv(self):
        """Charger et vérifier le CSV"""
        y_train_file = Path("data/y_train_labels/Y_train_T9NrBYo.csv")
        y_train = pd.read_csv(y_train_file, index_col=0)

        assert len(y_train) == 4410, "Mauvais nombre de patches"
        assert y_train.shape[1] > 0, "Pas de colonnes dans le CSV"

    def test_preprocessor_creation(self):
        """Créer un preprocessor"""
        from deep_datachallenge.preprocessing import ImagePreprocessor

        preprocessor = ImagePreprocessor(target_size=(160, 160))
        assert preprocessor is not None
        assert preprocessor.target_size == (160, 160)


class TestDataLoaders:
    """Tests des DataLoaders PyTorch"""

    @pytest.fixture
    def data_setup(self):
        """Fixture pour préparer les données"""
        from deep_datachallenge.preprocessing import ImagePreprocessor

        y_train_file = Path("data/y_train_labels/Y_train_T9NrBYo.csv")
        x_train_dir = Path("data/x_train_images")
        y_train = pd.read_csv(y_train_file, index_col=0)
        preprocessor = ImagePreprocessor(target_size=(160, 160))

        return y_train, x_train_dir, preprocessor

    def test_create_dataloaders(self, data_setup):
        """Créer les dataloaders"""
        from deep_datachallenge.dataset import create_dataloaders

        y_train, x_train_dir, preprocessor = data_setup

        train_loader, val_loader, class_weights = create_dataloaders(
            y_train, x_train_dir, preprocessor, batch_size=4, augment_train=False
        )

        assert train_loader is not None
        assert val_loader is not None
        assert class_weights is not None

    def test_dataloaders_sizes(self, data_setup):
        """Vérifier les tailles des dataloaders"""
        from deep_datachallenge.dataset import create_dataloaders

        y_train, x_train_dir, preprocessor = data_setup

        train_loader, val_loader, class_weights = create_dataloaders(
            y_train, x_train_dir, preprocessor, batch_size=4, augment_train=False
        )

        # 3526 train / 884 val
        assert len(train_loader) > 0, "Train loader vide"
        assert len(val_loader) > 0, "Val loader vide"

    def test_batch_shapes(self, data_setup):
        """Vérifier les shapes des batches"""
        from deep_datachallenge.dataset import create_dataloaders

        y_train, x_train_dir, preprocessor = data_setup

        train_loader, val_loader, class_weights = create_dataloaders(
            y_train, x_train_dir, preprocessor, batch_size=4, augment_train=False
        )

        batch = next(iter(train_loader))

        # Vérifier les shapes
        assert batch["image"].shape == (
            4,
            1,
            160,
            160,
        ), f"Shape image invalide: {batch['image'].shape}"
        assert batch["mask"].shape == (4, 160, 160), f"Shape mask invalide: {batch['mask'].shape}"
        assert len(batch["patch_name"]) == 4, "Nombre de noms de patches incorrect"

    def test_class_weights(self, data_setup):
        """Vérifier les poids des classes"""
        from deep_datachallenge.dataset import create_dataloaders

        y_train, x_train_dir, preprocessor = data_setup

        train_loader, val_loader, class_weights = create_dataloaders(
            y_train, x_train_dir, preprocessor, batch_size=4, augment_train=False
        )

        # 3 classes
        assert class_weights.shape[0] == 3, f"Mauvais nombre de poids: {class_weights.shape}"
        # Les poids doivent être positifs
        assert torch.all(class_weights > 0), "Poids négatifs détectés"
        # La somme doit être 3 (normalisée)
        assert torch.isclose(
            torch.sum(class_weights), torch.tensor(3.0), atol=0.1
        ), f"Somme des poids incorrecte: {torch.sum(class_weights)}"


class TestModels:
    """Tests des modèles"""

    @pytest.fixture
    def device(self):
        """Device (CPU ou GPU)"""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_unet_creation(self, device):
        """Créer un U-Net"""
        from deep_datachallenge.models import UNet

        model = UNet(in_channels=1, out_channels=3, depth=4)
        model = model.to(device)

        assert model is not None
        assert next(model.parameters()).device == device

    def test_unet_forward(self, device):
        """Test forward pass U-Net"""
        from deep_datachallenge.models import UNet

        model = UNet(in_channels=1, out_channels=3, depth=4)
        model = model.to(device)

        dummy_input = torch.randn(2, 1, 160, 160).to(device)
        output = model(dummy_input)

        assert output.shape == (2, 3, 160, 160), f"Shape output incorrecte: {output.shape}"

    def test_unet_params(self, device):
        """Vérifier le nombre de paramètres U-Net"""
        from deep_datachallenge.models import UNet

        model = UNet(in_channels=1, out_channels=3, depth=4)
        params = sum(p.numel() for p in model.parameters())

        # Devrait être entre 1M et 5M
        assert 1_000_000 < params < 50_000_000, f"Nombre de params étrange: {params:,}"


class TestMetrics:
    """Tests des métriques IoU"""

    def test_compute_iou(self):
        """Tester compute_iou"""
        from deep_datachallenge.metrics import compute_iou

        pred = torch.randint(0, 3, (4, 160, 160))
        target = torch.randint(0, 3, (4, 160, 160))

        iou_per_class = compute_iou(pred, target)

        assert iou_per_class.shape[0] == 3, "Mauvais nombre de classes"
        assert torch.all(iou_per_class >= 0), "IoU négatif"
        assert torch.all(iou_per_class <= 1), "IoU > 1"

    def test_metrics_tracker(self):
        """Tester MetricsTracker"""
        from deep_datachallenge.metrics import MetricsTracker

        tracker = MetricsTracker()

        # Ajouter quelques batch
        for _ in range(3):
            pred = torch.randint(0, 3, (4, 160, 160))
            target = torch.randint(0, 3, (4, 160, 160))
            tracker.update(pred, target, loss=0.5)

        metrics = tracker.get_metrics()

        assert "loss" in metrics, "Loss manquante"
        assert "iou_mean" in metrics, "IoU mean manquante"
        assert "iou_class_0" in metrics, "IoU class_0 manquante"


class TestTrainer:
    """Tests du trainer (le plus long)"""

    @pytest.fixture
    def device(self):
        """Device (CPU ou GPU)"""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @pytest.fixture
    def trainer_setup(self, device):
        """Fixture pour préparer trainer"""
        from deep_datachallenge.models import UNet
        from deep_datachallenge.dataset import create_dataloaders
        from deep_datachallenge.preprocessing import ImagePreprocessor
        from deep_datachallenge.trainer import SegmentationTrainer

        # Charger données
        y_train_file = Path("data/y_train_labels/Y_train_T9NrBYo.csv")
        x_train_dir = Path("data/x_train_images")
        y_train = pd.read_csv(y_train_file, index_col=0)
        preprocessor = ImagePreprocessor(target_size=(160, 160))

        # Mini dataloaders
        train_loader, val_loader, class_weights = create_dataloaders(
            y_train, x_train_dir, preprocessor, batch_size=8, augment_train=False
        )

        # Modèle
        model = UNet(in_channels=1, out_channels=3, depth=4)
        model = model.to(device)

        # Trainer
        trainer = SegmentationTrainer(model, device, lr=1e-3, class_weights=class_weights)

        return trainer, train_loader, val_loader

    def test_trainer_creation(self, trainer_setup):
        """Créer un trainer"""
        trainer, _, _ = trainer_setup
        assert trainer is not None
        assert trainer.optimizer is not None
        assert trainer.criterion is not None

    def test_trainer_train_epoch(self, trainer_setup):
        """Test une epoch d'entraînement"""
        trainer, train_loader, _ = trainer_setup

        metrics = trainer.train_epoch(train_loader)

        assert "loss" in metrics, "Loss manquante"
        assert "iou_mean" in metrics, "IoU mean manquante"
        assert metrics["loss"] > 0, "Loss invalide"

    def test_trainer_validate(self, trainer_setup):
        """Test validation"""
        trainer, _, val_loader = trainer_setup

        metrics = trainer.validate(val_loader)

        assert "loss" in metrics, "Loss manquante"
        assert "iou_mean" in metrics, "IoU mean manquante"

    def test_trainer_fit_short(self, trainer_setup):
        """Test full fit avec 1 epoch seulement"""
        trainer, train_loader, val_loader = trainer_setup

        history = trainer.fit(
            train_loader,
            val_loader,
            epochs=1,
            early_stopping_patience=10,
            save_dir=Path("checkpoints_test"),
            model_name="test_unet",
        )

        assert "train_loss" in history, "Train loss manquante"
        assert "val_loss" in history, "Val loss manquante"
        assert len(history["train_loss"]) == 1, "Mauvais nombre d'epochs"
