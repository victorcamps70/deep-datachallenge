"""
Script de prédiction : génère y_test.csv à partir du modèle entraîné
"""

import torch
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

from deep_datachallenge.models import UNet
from deep_datachallenge.preprocessing import ImagePreprocessor


def load_test_patch_names(x_test_dir):
    """
    Charger les noms des patches de test

    Args:
        x_test_dir (Path): Répertoire contenant les images de test

    Returns:
        list: Liste des noms de patches
    """
    x_test_dir = Path(x_test_dir)
    patch_files = sorted(x_test_dir.glob("*.npy"))
    patch_names = [f.stem for f in patch_files]
    return patch_names


def predict_on_test_set(
    model,
    x_test_dir,
    preprocessor,
    device,
    batch_size=32,
):
    """
    Générer les prédictions sur le jeu de test avec redimensionnement post-traitement

    Args:
        model: Modèle entraîné
        x_test_dir: Répertoire des images de test
        preprocessor: ImagePreprocessor
        device: CPU ou GPU
        batch_size: Taille des batches

    Returns:
        dict: {patch_name: prediction_flattened_resized_à_taille_originale}
    """

    x_test_dir = Path(x_test_dir)
    patch_names = load_test_patch_names(x_test_dir)
    predictions = {}

    # Charger les tailles originales
    original_sizes = {}
    for patch_name in patch_names:
        image_path = x_test_dir / f"{patch_name}.npy"
        image = np.load(image_path)
        original_sizes[patch_name] = image.shape  # (H, W)

    print(f"\nGénération des prédictions sur {len(patch_names)} patches de test...")

    model.eval()

    with torch.no_grad():
        for i in tqdm(range(0, len(patch_names), batch_size)):
            batch_names = patch_names[i : i + batch_size]

            # Charger et prétraiter le batch
            images = []
            for patch_name in batch_names:
                image_path = x_test_dir / f"{patch_name}.npy"
                image = np.load(image_path)

                # Prétraiter (pas de mask pour le test)
                image_processed = preprocessor.full_preprocessing(
                    image, mask=None, normalize=True, fill_missing=True
                )

                images.append(image_processed)

            # Convertir en tensor
            images_tensor = (
                torch.from_numpy(np.array(images)).float().unsqueeze(1)
            )  # [B, 1, 160, 160]
            images_tensor = images_tensor.to(device)

            # Prédire
            logits = model(images_tensor)  # [B, 3, 160, 160]
            preds = torch.argmax(logits, dim=1)  # [B, 160, 160]

            # Redimensionner à la taille originale avec OpenCV (cohérence avec preprocessing)
            for j, patch_name in enumerate(batch_names):
                pred_np = preds[j].cpu().numpy().astype(np.uint8)  # [160, 160]

                # Redimensionner de 160x160 à 160x272 avec INTER_NEAREST
                pred_resized = cv2.resize(
                    pred_np,
                    (272, 160),  # OpenCV : (width, height) = (272, 160)
                    interpolation=cv2.INTER_NEAREST,
                )  # [160, 272]

                pred_flat = pred_resized.flatten()
                predictions[patch_name] = pred_flat

    return predictions


def create_y_test_csv(predictions, output_path):
    """
    Créer le CSV y_test au même format que y_train
    (redimensionné à la taille originale 160x272 = 43520 pixels)

    Args:
        predictions (dict): {patch_name: prediction_flattened_43520}
        output_path (Path): Chemin de sortie
    """

    output_path = Path(output_path)

    # Créer un DataFrame
    data = {patch_name: pred for patch_name, pred in predictions.items()}
    df = pd.DataFrame(data).T  # Transposer pour avoir patches en index

    # Renommer les colonnes (0, 1, 2, ..., 43519) - 160 × 272 = 43520
    df.columns = list(range(df.shape[1]))

    # Sauvegarder
    df.to_csv(output_path)

    print(f"✓ Fichier y_test sauvegardé: {output_path}")
    print(f"  Shape: {df.shape} (160 × 272 = {160 * 272} pixels)")
    print(f"  Aperçu:")
    print(df.head())


def main():
    """Générer les prédictions et créer y_test.csv"""

    # Configuration
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CHECKPOINT_PATH = Path("checkpoints/unet_best.pt")
    X_TEST_DIR = Path("data/x_test_images")
    OUTPUT_PATH = Path("data/y_test_predictions.csv")
    BATCH_SIZE = 32

    print(f"\n{'='*70}")
    print("PRÉDICTION SUR LE JEU DE TEST")
    print(f"{'='*70}")
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Test images: {X_TEST_DIR}")

    # Vérifier que le checkpoint existe
    if not CHECKPOINT_PATH.exists():
        print(f"❌ Erreur: {CHECKPOINT_PATH} n'existe pas!")
        print("   Veuillez d'abord entraîner le modèle avec: python train.py")
        return

    # Charger le modèle
    print(f"\n{'='*70}")
    print("CHARGEMENT DU MODÈLE")
    print(f"{'='*70}")

    model = UNet(in_channels=1, out_channels=3, depth=4)
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    model = model.to(DEVICE)
    print(f"✓ Modèle chargé depuis {CHECKPOINT_PATH}")

    # Créer le preprocessor
    preprocessor = ImagePreprocessor(target_size=(160, 160))

    # Générer les prédictions
    predictions = predict_on_test_set(
        model, X_TEST_DIR, preprocessor, DEVICE, batch_size=BATCH_SIZE
    )

    # Créer le CSV
    print(f"\n{'='*70}")
    print("CRÉATION DU CSV")
    print(f"{'='*70}")

    create_y_test_csv(predictions, OUTPUT_PATH)

    print(f"\n✓ Pipeline de prédiction complété !")


if __name__ == "__main__":
    main()
