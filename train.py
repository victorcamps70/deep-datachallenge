"""
Script d'entraînement principal pour comparer différents modèles
Supporte la reprise d'entraînement avec --resume
"""

import torch
import pandas as pd
from pathlib import Path
import json
import argparse

from deep_datachallenge.models.unet import UNet
from deep_datachallenge.models.unet_pretrained import (
    create_unet_pretrained,
    freeze_encoder,
    unfreeze_encoder,
)
from deep_datachallenge.dataset import create_dataloaders
from deep_datachallenge.preprocessing import ImagePreprocessor
from deep_datachallenge.trainer import SegmentationTrainer
from deep_datachallenge.losses import FocalLoss, DiceLoss, CombinedLoss


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
    resume=False,
    loss_type="crossentropy",
    checkpoint_path=None,
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
        resume (bool): Si True, reprendre depuis un checkpoint
        loss_type (str): Type de loss ('crossentropy', 'focal', 'dice', 'combined')
        checkpoint_path (str): Chemin vers un checkpoint à charger avant entraînement

    Returns:
        dict: Résultats et historique
    """

    print(f"\n{'='*70}")
    print(f"ENTRAÎNEMENT: {model_name}")
    if resume or checkpoint_path:
        print("MODE: REPRISE D'ENTRAÎNEMENT")
    if checkpoint_path:
        print(f"Chargement checkpoint: {checkpoint_path}")
    loss_names = {
        "crossentropy": "CrossEntropyLoss",
        "focal": "Focal Loss (gamma=2.0)",
        "dice": "Dice Loss",
        "combined": "Combined Loss (CE + Dice)",
    }
    print(f"Loss: {loss_names.get(loss_type, 'Unknown')}")
    print(f"{'='*70}")

    trainer = SegmentationTrainer(
        model, device, lr=lr, class_weights=class_weights, loss_type=loss_type
    )

    # Charger un checkpoint avant entraînement si fourni
    if checkpoint_path:
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint)
            print(f"✓ Checkpoint chargé: {checkpoint_path}\n")
        else:
            print(f"⚠ Checkpoint non trouvé: {checkpoint_path}\n")

    history = trainer.fit(
        train_loader,
        val_loader,
        epochs=epochs,
        early_stopping_patience=10,
        save_dir=save_dir,
        model_name=model_name,
        resume=resume,
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

    # Parser arguments
    parser = argparse.ArgumentParser(description="Entraîner le modèle U-Net")
    parser.add_argument("--epochs", type=int, default=50, help="Nombre d'époque (défaut: 50)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (défaut: 32)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (défaut: 1e-3)")
    parser.add_argument("--resume", action="store_true", help="Reprendre depuis un checkpoint")
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Chemin vers un checkpoint à charger (ex: checkpoints/unet_pretrained_resnet34_frozen_best.pt)",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="crossentropy",
        choices=["crossentropy", "focal", "dice", "combined"],
        help="Type de loss function à utiliser (défaut: crossentropy)",
    )
    parser.add_argument(
        "--focal",
        action="store_true",
        help="(DEPRECATED) Utiliser Focal Loss - utilisez --loss focal à la place",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Utiliser UNet avec encodeur ResNet pré-entraîné (défaut: UNet custom)",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="resnet34",
        help="Encodeur à utiliser (resnet18, resnet34, resnet50, etc.) si --pretrained",
    )
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Geler l'encodeur pré-entraîné pendant l'entraînement",
    )
    args = parser.parse_args()

    # Configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LR = args.lr
    SAVE_DIR = Path("checkpoints")
    RESUME = args.resume
    CHECKPOINT_PATH = args.checkpoint_path
    LOSS_TYPE = args.loss if args.loss else ("focal" if args.focal else "crossentropy")
    USE_PRETRAINED = args.pretrained
    ENCODER_NAME = args.encoder
    FREEZE_ENCODER = args.freeze_encoder

    print(f"\n{'='*70}")
    print("CONFIGURATION")
    print(f"{'='*70}")
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning rate: {LR}")
    loss_names = {
        "crossentropy": "CrossEntropyLoss (Baseline)",
        "focal": "Focal Loss (gamma=2.0)",
        "dice": "Dice Loss",
        "combined": "Combined Loss (CrossEntropy + Dice)",
    }
    print(f"Loss Function: {loss_names.get(LOSS_TYPE, 'Unknown')}")
    if USE_PRETRAINED:
        print(f"Model: UNet with pretrained {ENCODER_NAME} encoder")
        print(f"Encoder frozen: {FREEZE_ENCODER}")
    else:
        print("Model: Custom UNet (no pretraining)")
    if CHECKPOINT_PATH:
        print(f"Checkpoint to load: {CHECKPOINT_PATH}")
    print(f"Save directory: {SAVE_DIR}")
    print(f"Mode: {'REPRISE' if RESUME else 'NOUVEAU'}")
    if CHECKPOINT_PATH:
        print(f"(Chargement depuis: {CHECKPOINT_PATH})")
    print()

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

    # Créer le modèle
    if USE_PRETRAINED:
        print(f"{'='*70}")
        print("CRÉATION DU MODÈLE")
        print(f"{'='*70}")
        print(f"Téléchargement du modèle pré-entraîné {ENCODER_NAME}...")
        model = create_unet_pretrained(encoder_name=ENCODER_NAME, encoder_weights="imagenet")

        if FREEZE_ENCODER:
            freeze_encoder(model)

        model_name = f"unet_pretrained_{ENCODER_NAME}"
        if FREEZE_ENCODER:
            model_name += "_frozen"
    else:
        print(f"{'='*70}")
        print("CRÉATION DU MODÈLE")
        print(f"{'='*70}")
        model = UNet(in_channels=1, out_channels=3, depth=4)
        model_name = "unet"

    print(f"✓ Modèle créé\n")

    # Entraîner le modèle
    results[model_name] = train_model(
        model,
        model_name,
        train_loader,
        val_loader,
        class_weights,
        DEVICE,
        epochs=EPOCHS,
        lr=LR,
        save_dir=SAVE_DIR,
        resume=RESUME,
        loss_type=LOSS_TYPE,
        checkpoint_path=CHECKPOINT_PATH,
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
