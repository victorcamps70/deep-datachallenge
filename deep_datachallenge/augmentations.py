"""
Augmentations personnalisées pour la segmentation d'images ultrasoniques
Inclut: elastic deformation, Gaussian noise, etc.
"""

import torch
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter, map_coordinates


class ElasticDeformation:
    """
    Applique une déformation élastique aléatoire à l'image
    Utile pour augmenter les données d'images médicales

    Args:
        alpha (float): Magnitude de la déformation
        sigma (float): Largeur du filtre Gaussien
        p (float): Probabilité d'appliquer la transformation
    """

    def __init__(self, alpha=30, sigma=5, p=0.5):
        self.alpha = alpha
        self.sigma = sigma
        self.p = p

    def __call__(self, img_tensor):
        """
        Args:
            img_tensor (torch.Tensor): Image tensor de shape [1, H, W]

        Returns:
            torch.Tensor: Image déformée
        """
        if np.random.rand() > self.p:
            return img_tensor

        # Convertir en numpy
        img = img_tensor.numpy()  # [1, H, W]

        # Appliquer la déformation élastique
        img_deformed = self.elastic_transform(img[0])  # [H, W]

        # Reconvertir en tensor
        return torch.from_numpy(img_deformed).unsqueeze(0)

    def elastic_transform(self, image, alpha=None, sigma=None, random_state=None):
        """
        Déformation élastique d'une image 2D

        Args:
            image (np.ndarray): Image en échelle de gris [H, W]
            alpha (float): Magnitude de la déformation
            sigma (float): Largeur du filtre Gaussien
            random_state (int): Seed pour reproductibilité

        Returns:
            np.ndarray: Image déformée
        """
        if alpha is None:
            alpha = self.alpha
        if sigma is None:
            sigma = self.sigma

        if random_state is not None:
            np.random.seed(random_state)

        height, width = image.shape

        # Générer des cartes de déplacement aléatoires
        dx = (
            gaussian_filter((np.random.rand(height, width) * 2 - 1), sigma, mode="constant", cval=0)
            * alpha
        )

        dy = (
            gaussian_filter((np.random.rand(height, width) * 2 - 1), sigma, mode="constant", cval=0)
            * alpha
        )

        # Créer les coordonnées
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

        # Appliquer la transformation
        distorted_image = map_coordinates(image, indices, order=1, cval=0.0).reshape(image.shape)

        return distorted_image.astype(image.dtype)


class GaussianNoise:
    """
    Ajoute un bruit Gaussien aléatoire à l'image

    Args:
        std (float): Écart-type du bruit
        p (float): Probabilité d'appliquer la transformation
    """

    def __init__(self, std=0.02, p=0.3):
        self.std = std
        self.p = p

    def __call__(self, img_tensor):
        """
        Args:
            img_tensor (torch.Tensor): Image tensor de shape [1, H, W]

        Returns:
            torch.Tensor: Image avec bruit ajouté
        """
        if np.random.rand() > self.p:
            return img_tensor

        # Ajouter du bruit Gaussien
        noise = torch.randn_like(img_tensor) * self.std
        img_noisy = torch.clamp(img_tensor + noise, 0, 1)  # Garder dans [0, 1]

        return img_noisy
