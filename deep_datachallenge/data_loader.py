"""
Script pour charger les images et labels du datachallenge
"""

import numpy as np
import pandas as pd
from pathlib import Path


class DataLoader:
    """Charge les images et labels pour l'entraînement"""

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
        self.y_train = pd.read_csv(self.y_train_file, index_col=0)

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

    def get_all_patches(self):
        """
        Retourne la liste de tous les patches du dataset d'entraînement

        Returns:
            list: Liste des noms de patches
        """
        return list(self.y_train.index)
