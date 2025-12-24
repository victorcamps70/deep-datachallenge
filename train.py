"""
Script d'entraînement principal pour comparer différents modèles
"""

import torch
import pandas as pd
from pathlib import Path
import json

from deep_datachallenge.models.unet import UNet
from deep_datachallenge.dataset import create_dataloaders
from deep_datachallenge.preprocessing import ImagePreprocessor
from deep_datachallenge.trainer import SegmentationTrainer


def train_model(
    model,
    model_name,
    train_loader,
    val_loader,
    class_weights,
    device,
    epochs=50,
    lr=1e-3,
    save_dir=None,
):
    """
    Entraîner un modèle

    Args:
        model (nn.Module): Le modèle à entraîner
        model_name (str): Nom du modèle
        train_loader, val_loader: DataLoaders
        class_weights: Poids des classes
        device: CPU ou GPU
        epochs: Nombre d'époque
        lr: Learning rate
        save_dir: Répertoire pour sauvegarder

    Returns:
        dict: Résultats et historique
    """

    print(f"\n{'='*70}")
    print(f"ENTRAÎNEMENT: {model_name}")
    print(f"{'='*70}")

    trainer = SegmentationTrainer(model, device, lr=lr, class_weights=class_weights)

    history = trainer.fit(
        train_loader,
        val_loader,
        epochs=epochs,
        early_stopping_patience=10,
        save_dir=save_dir,
        model_name=model_name,
    )

    results = {
        "model_name": model_name,
        "best_val_iou": trainer.best_val_iou,
        "history": history,
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "final_train_iou": history["train_iou"][-1],
        "final_val_iou": history["val_iou"][-1],
    }

    return results


def main():
    """Script principal d'entraînement"""

    # Configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32
    EPOCHS = 50
    LR = 1e-3
    SAVE_DIR = Path("checkpoints")

    print(f"\n{'='*70}")
    print("CONFIGURATION")
    print(f"{'='*70}")
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: {LR}")
    print(f"Save directory: {SAVE_DIR}\n")

    # Créer le répertoire de sauvegarde
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # Charger les données
    print(f"{'='*70}")
    print("CHARGEMENT DES DONNÉES")
    print(f"{'='*70}")

    data_dir = Path("data")
    y_train_file = data_dir / "y_train_labels" / "Y_train_T9NrBYo.csv"
    x_train_dir = data_dir / "x_train_images"

    y_train = pd.read_csv(y_train_file, index_col=0)
    preprocessor = ImagePreprocessor(target_size=(160, 160))

    # Créer les DataLoaders
    train_loader, val_loader, class_weights = create_dataloaders(
        y_train, x_train_dir, preprocessor, batch_size=BATCH_SIZE, augment_train=True
    )

    # Déplacer les poids sur le device
    class_weights = class_weights.to(DEVICE)

    # Entraîner les modèles
    results = {}

    # Modèle 1: U-Net
    model_unet = UNet(in_channels=1, out_channels=3, depth=4)
    results["unet"] = train_model(
        model_unet,
        "unet",
        train_loader,
        val_loader,
        class_weights,
        DEVICE,
        epochs=EPOCHS,
        lr=LR,
        save_dir=SAVE_DIR,
    )

    # Afficher la comparaison
    print(f"\n{'='*70}")
    print("COMPARAISON DES MODÈLES")
    print(f"{'='*70}\n")

    for model_name, result in results.items():
        print(f"Modèle: {result['model_name']}")
        print(f"  • Best Val IoU:    {result['best_val_iou']:.4f}")
        print(f"  • Final Train Loss: {result['final_train_loss']:.6f}")
        print(f"  • Final Val Loss:   {result['final_val_loss']:.6f}")
        print(f"  • Final Train IoU:  {result['final_train_iou']:.4f}")
        print(f"  • Final Val IoU:    {result['final_val_iou']:.4f}\n")

    # Sauvegarder les résultats
    results_summary = {
        name: {
            "best_val_iou": result["best_val_iou"],
            "final_train_loss": result["final_train_loss"],
            "final_val_loss": result["final_val_loss"],
            "final_train_iou": result["final_train_iou"],
            "final_val_iou": result["final_val_iou"],
        }
        for name, result in results.items()
    }

    results_path = SAVE_DIR / "results_summary.json"
    with open(results_path, "w") as f:
        json.dump(results_summary, f, indent=2)

    print(f"✓ Résultats sauvegardés: {results_path}")


if __name__ == "__main__":
    main()
