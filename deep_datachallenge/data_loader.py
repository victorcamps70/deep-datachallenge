"""
Script pour charger les images et labels du datachallenge
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


class DataLoader:
    """Classe pour charger et traiter les données du datachallenge"""

    def __init__(self, data_dir="data"):
        """
        Initialise le DataLoader

        Args:
            data_dir (str): Chemin vers le répertoire des données
        """
        self.data_dir = Path(data_dir)
        self.x_train_dir = self.data_dir / "x_train_images"
        self.x_test_dir = self.data_dir / "x_test_images"
        self.y_train_file = self.data_dir / "y_train_labels" / "Y_train_T9NrBYo.csv"

        # Charger les labels d'entraînement
        print("Chargement des labels d'entraînement...")
        self.y_train = pd.read_csv(self.y_train_file, index_col=0)
        print(f"{len(self.y_train)} patches d'entraînement chargés")

    def load_image(self, patch_name, phase="train"):
        """
        Charge une image .npy

        Args:
            patch_name (str): Nom du patch (ex: 'well_12_section_1_patch_0')
            phase (str): 'train' ou 'test'

        Returns:
            np.array: Image chargée
        """
        if phase == "train":
            image_path = self.x_train_dir / f"{patch_name}.npy"
        else:
            image_path = self.x_test_dir / f"{patch_name}.npy"

        if not image_path.exists():
            raise FileNotFoundError(f"Image non trouvée: {image_path}")

        image = np.load(image_path)
        return image

    def load_mask(self, patch_name):
        """
        Charge le masque (label) pour un patch et le reconstruit

        Args:
            patch_name (str): Nom du patch

        Returns:
            np.array: Masque de segmentation reshapé (160, hauteur)
        """
        if patch_name not in self.y_train.index:
            raise ValueError(f"Patch non trouvé dans les labels: {patch_name}")

        # Récupérer la ligne du CSV
        mask_values = self.y_train.loc[patch_name].values

        # Enlever les valeurs de padding (-1)
        mask_filtered = np.array([v for v in mask_values if v != -1])

        # Redimensionner en (160, -1) pour obtenir les vraies dimensions
        mask = mask_filtered.reshape(160, -1)

        return mask

    def load_pair(self, patch_name, phase="train"):
        """
        Charge une image et son masque correspondant

        Args:
            patch_name (str): Nom du patch
            phase (str): 'train' ou 'test'

        Returns:
            tuple: (image, mask) ou (image, None) si phase='test'
        """
        image = self.load_image(patch_name, phase=phase)

        if phase == "train":
            mask = self.load_mask(patch_name)
            return image, mask
        else:
            return image, None

    def visualize_pair(self, patch_name, phase="train"):
        """
        Affiche l'image et le masque côte à côte

        Args:
            patch_name (str): Nom du patch
            phase (str): 'train' ou 'test'
        """
        image, mask = self.load_pair(patch_name, phase=phase)

        if phase == "train" and mask is not None:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # Afficher l'image
            im1 = axes[0].imshow(image, cmap="gray")
            axes[0].set_title(f"Image: {patch_name}\nShape: {image.shape}")
            axes[0].set_xlabel("X (pixels)")
            axes[0].set_ylabel("Y (pixels)")
            plt.colorbar(im1, ax=axes[0])

            # Afficher le masque
            im2 = axes[1].imshow(mask, cmap="viridis")
            axes[1].set_title(f"Masque (Label)\nShape: {mask.shape}")
            axes[1].set_xlabel("X (pixels)")
            axes[1].set_ylabel("Y (pixels)")
            plt.colorbar(im2, ax=axes[1])
        else:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            im = ax.imshow(image, cmap="gray")
            ax.set_title(f"Image (Test): {patch_name}\nShape: {image.shape}")
            ax.set_xlabel("X (pixels)")
            ax.set_ylabel("Y (pixels)")
            plt.colorbar(im, ax=ax)

        plt.tight_layout()
        plt.show()

    def get_random_patch(self):
        """
        Retourne le nom d'un patch aléatoire du dataset d'entraînement

        Returns:
            str: Nom du patch
        """
        return np.random.choice(self.y_train.index)

    def get_all_patches(self):
        """
        Retourne la liste de tous les patches du dataset d'entraînement

        Returns:
            list: Liste des noms de patches
        """
        return list(self.y_train.index)


# Exemple d'utilisation
if __name__ == "__main__":
    # Initialiser le DataLoader
    loader = DataLoader(data_dir="data")

    # Obtenir un patch aléatoire
    patch_name = loader.get_random_patch()
    print(f"\nPatch aléatoire sélectionné: {patch_name}")

    # Charger l'image et le masque
    print(f"\nChargement de l'image et du masque...")
    image, mask = loader.load_pair(patch_name, phase="train")

    print(f"Image shape: {image.shape}")
    if mask is not None:
        print(f"Mask shape: {mask.shape}")
        print(f"Image - Min: {image.min()}, Max: {image.max()}, Mean: {image.mean():.2f}")
        print(f"Mask - Min: {mask.min()}, Max: {mask.max()}, Classes: {np.unique(mask)}")
    else:
        print(f"Image - Min: {image.min()}, Max: {image.max()}, Mean: {image.mean():.2f}")
        print("Mask: None (test set)")

    # Visualiser
    print(f"\nAffichage de l'image et du masque...")
    loader.visualize_pair(patch_name, phase="train")
