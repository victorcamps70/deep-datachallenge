"""
Module de preprocessing pour standardiser les dimensions des images et gérer le déséquilibre des classes
"""

import numpy as np
import cv2
from pathlib import Path
import warnings


class ImagePreprocessor:
    """Gère les opérations de preprocessing des images et masques"""

    def __init__(self, target_size=(160, 160)):
        """
        Initialise le preprocessor

        Args:
            target_size (tuple): Dimensions cibles de l'image (hauteur, largeur)
        """
        self.target_size = target_size

    def resize_image(self, image, interpolation=cv2.INTER_LINEAR):
        """
        Redimensionne l'image aux dimensions cibles

        Args:
            image (np.ndarray): Image d'entrée
            interpolation (int): Méthode d'interpolation OpenCV
                - cv2.INTER_LINEAR: Bon pour les images générales (par défaut)
                - cv2.INTER_NEAREST: Plus rapide, pour les labels
                - cv2.INTER_CUBIC: Plus lent mais plus lisse

        Returns:
            np.ndarray: Image redimensionnée
        """
        if image.shape[:2] == self.target_size:
            return image

        # Utiliser OpenCV pour redimensionner
        resized = cv2.resize(
            image, (self.target_size[1], self.target_size[0]), interpolation=interpolation
        )

        return resized

    def resize_mask(self, mask):
        """
        Redimensionne le masque de segmentation aux dimensions cibles en utilisant
        l'interpolation NEAREST pour préserver les valeurs des classes

        Args:
            mask (np.ndarray): Masque d'entrée

        Returns:
            np.ndarray: Masque redimensionné avec classes préservées
        """
        if mask.shape[:2] == self.target_size:
            return mask

        # Utiliser l'interpolation NEAREST pour préserver les labels des classes
        resized = cv2.resize(
            mask.astype(np.float32),
            (self.target_size[1], self.target_size[0]),
            interpolation=cv2.INTER_NEAREST,
        )

        # Convertir au type de données original
        resized = resized.astype(mask.dtype)

        return resized

    def preprocess_pair(self, image, mask):
        """
        Prétraite l'image et le masque ensemble

        Args:
            image (np.ndarray): Image d'entrée
            mask (np.ndarray): Masque d'entrée

        Returns:
            tuple: (image_redimensionnée, masque_redimensionné)
        """
        image_resized = self.resize_image(image, interpolation=cv2.INTER_LINEAR)
        mask_resized = self.resize_mask(mask)

        return image_resized, mask_resized

    def normalize_image(self, image):
        """
        Normalise l'image à la plage [0, 1] en utilisant le scaling min-max

        Args:
            image (np.ndarray): Image d'entrée

        Returns:
            np.ndarray: Image normalisée (float32)
        """
        image_float = image.astype(np.float32)

        # Normalisation min-max par patch
        img_min = image_float.min()
        img_max = image_float.max()

        if img_max - img_min > 0:
            normalized = (image_float - img_min) / (img_max - img_min)
        else:
            # Traiter le cas où toutes les valeurs sont identiques
            normalized = np.zeros_like(image_float)

        return normalized

    def fill_missing_values(self, image, fill_value=0):
        """
        Remplace les valeurs manquantes (NaN) par une valeur spécifiée

        Args:
            image (np.ndarray): Image d'entrée
            fill_value (float): Valeur de remplissage pour les NaN (par défaut 0)

        Returns:
            np.ndarray: Image avec les NaN remplacées
        """
        if np.isnan(image).any():
            # Créer une copie pour ne pas modifier l'original
            image_filled = image.copy()
            # Remplacer les NaN par la valeur spécifiée
            image_filled[np.isnan(image_filled)] = fill_value
            return image_filled
        return image

    def full_preprocessing(self, image, mask, normalize=True, fill_missing=True, fill_value=0):
        """
        Applique le pipeline de preprocessing complet: redimensionnement → remplissage NaN (si nécessaire) → normalisation

        Args:
            image (np.ndarray): Image d'entrée
            mask (np.ndarray): Masque d'entrée
            normalize (bool): Normaliser l'image ou non
            fill_missing (bool): Remplacer les valeurs manquantes (NaN) après redimensionnement si trouvées
            fill_value (float): Valeur de remplissage pour les NaN

        Returns:
            tuple: (image_traitée, masque_traité)
        """
        # Étape 1: Redimensionner
        image_resized, mask_resized = self.preprocess_pair(image, mask)

        # Étape 2: Remplir les NaN SEULEMENT s'ils sont trouvés après redimensionnement
        if fill_missing and np.isnan(image_resized).any():
            image_resized = self.fill_missing_values(image_resized, fill_value=fill_value)

        # Étape 3: Normaliser
        if normalize:
            image_resized = self.normalize_image(image_resized)

        return image_resized, mask_resized


# Fonctions utilitaires
def resize_batch(images, masks=None, target_size=(160, 160)):
    """
    Redimensionne un batch d'images et masques optionnels

    Args:
        images (list): Liste des arrays d'images
        masks (list): Liste des arrays de masques (optionnel)
        target_size (tuple): Dimensions cibles

    Returns:
        tuple: (images_redimensionnées, masques_redimensionnés) ou seulement images_redimensionnées
    """
    preprocessor = ImagePreprocessor(target_size=target_size)

    resized_images = [preprocessor.resize_image(img) for img in images]

    if masks is not None:
        resized_masks = [preprocessor.resize_mask(m) for m in masks]
        return resized_images, resized_masks

    return resized_images


def check_image_sizes(images, masks=None):
    """
    Vérifie et rapporte les tailles des images/masques dans un batch

    Args:
        images (list): Liste des arrays d'images
        masks (list): Liste des arrays de masques (optionnel)

    Returns:
        dict: Dictionnaire avec les statistiques de taille
    """
    image_sizes = {}
    for img in images:
        size_key = f"{img.shape[0]}x{img.shape[1]}"
        image_sizes[size_key] = image_sizes.get(size_key, 0) + 1

    result = {"images": image_sizes}

    if masks is not None:
        mask_sizes = {}
        for mask in masks:
            size_key = f"{mask.shape[0]}x{mask.shape[1]}"
            mask_sizes[size_key] = mask_sizes.get(size_key, 0) + 1
        result["masks"] = mask_sizes

    return result


if __name__ == "__main__":
    # Exemple d'utilisation
    preprocessor = ImagePreprocessor(target_size=(160, 160))

    # Créer des images factices pour tester
    print("Test du ImagePreprocessor...")

    # Test 1: Redimensionner image 160x160 (ne devrait pas changer)
    img_160 = np.random.rand(160, 160)
    resized = preprocessor.resize_image(img_160)
    print(f"Test 1 - image 160x160: {img_160.shape} → {resized.shape}")

    # Test 2: Redimensionner image 160x272 (devrait devenir 160x160)
    img_272 = np.random.rand(160, 272)
    resized = preprocessor.resize_image(img_272)
    print(f"Test 2 - image 160x272: {img_272.shape} → {resized.shape}")

    # Test 3: Redimensionner le masque
    mask_272 = np.random.randint(0, 3, (160, 272))
    resized_mask = preprocessor.resize_mask(mask_272)
    print(f"Test 3 - masque 160x272: {mask_272.shape} → {resized_mask.shape}")
    print(f"         Valeurs uniques préservées: {np.unique(mask_272) == np.unique(resized_mask)}")

    # Test 4: Preprocessing complet
    img = np.random.randint(0, 256, (160, 272), dtype=np.uint8)
    mask = np.random.randint(0, 3, (160, 272), dtype=np.uint8)
    img_proc, mask_proc = preprocessor.full_preprocessing(img, mask)
    print(
        f"Test 4 - Preprocessing complet: image {img.shape} → {img_proc.shape}, "
        + f"masque {mask.shape} → {mask_proc.shape}"
    )
