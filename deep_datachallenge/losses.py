"""
Fonctions de loss personnalisées pour la segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss pour l'apprentissage avec déséquilibre de classes

    La Focal Loss est idéale pour les problèmes où:
    - Les classes sont très déséquilibrées (ex: TIE = 5.81% vs Background = 63%)
    - Il y a beaucoup d'exemples faciles et peu d'exemples difficiles

    Elle pénalise davantage les prédictions incorrectes (hard negatives) et
    réduit le poids des exemples faciles bien classifiés.

    Paper: "Focal Loss for Dense Object Detection"
    https://arxiv.org/abs/1708.02002

    Args:
        alpha (list/tensor): Poids des classes (balance), shape [num_classes]
                            Exemple: [0.25, 1.0, 4.0] pour 3 classes
        gamma (float): Facteur de concentration (concentration sur hard examples)
                      gamma=0 → CrossEntropyLoss normal
                      gamma=2 → Focal Loss standard
        reduction (str): 'mean' ou 'none'
    """

    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Calculer la Focal Loss

        Args:
            inputs (torch.Tensor): Logits du modèle, shape [N, C, H, W]
                                  où C = nombre de classes
            targets (torch.Tensor): Labels de classe, shape [N, H, W]
                                   contient les indices 0, 1, ..., C-1

        Returns:
            torch.Tensor: Loss scalaire
        """
        # Vérifier les dimensions
        if inputs.dim() == 4 and targets.dim() == 3:
            # Cas segmentation: [N, C, H, W] et [N, H, W]
            N, C, H, W = inputs.shape
            # Reshape pour traiter comme classification multi-classe
            inputs = inputs.permute(0, 2, 3, 1).contiguous().view(-1, C)
            targets = targets.view(-1)

        # Appliquer softmax pour obtenir les probabilités
        p = F.softmax(inputs, dim=-1)

        # Cross entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")

        # Récupérer la probabilité de la classe vraie
        p_t = p.gather(1, targets.view(-1, 1)).squeeze(1)

        # Focal term: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Focal Loss = -focal_weight * ce_loss
        focal_loss = focal_weight * ce_loss

        # Appliquer les poids alpha si fournis (class weighting)
        if self.alpha is not None:
            if isinstance(self.alpha, (list, tuple)):
                alpha_t = torch.tensor(self.alpha, device=inputs.device, dtype=inputs.dtype)
            else:
                alpha_t = self.alpha

            # alpha_t shape: [C], récupérer alpha pour chaque sample
            alpha_t = alpha_t.gather(0, targets)
            focal_loss = alpha_t * focal_loss

        # Réduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class CombinedLoss(nn.Module):
    """
    Loss combinée: Focal Loss + Dice Loss
    Utile pour augmenter la robustesse

    Args:
        alpha (list): Poids des classes pour Focal Loss
        gamma (float): Facteur de concentration pour Focal Loss
        focal_weight (float): Poids de Focal Loss dans la combinaison
        dice_weight (float): Poids de Dice Loss dans la combinaison
    """

    def __init__(self, alpha=None, gamma=2.0, focal_weight=1.0, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, inputs, targets):
        focal = self.focal_loss(inputs, targets)
        # Dice Loss peut être ajouté ici si nécessaire
        return self.focal_weight * focal
