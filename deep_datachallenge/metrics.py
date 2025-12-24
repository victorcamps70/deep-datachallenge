"""
Métriques pour l'évaluation de la segmentation - IoU uniquement
"""

import torch
import numpy as np


def compute_iou(pred, target, num_classes=3):
    """
    Calcule l'Intersection over Union (IoU) pour chaque classe

    Args:
        pred (torch.Tensor): Prédictions du modèle [B, H, W] (class indices)
        target (torch.Tensor): Masques de référence [B, H, W] (class indices)
        num_classes (int): Nombre de classes

    Returns:
        torch.Tensor: IoU pour chaque classe [num_classes]
    """

    iou_per_class = []

    for cls in range(num_classes):
        # Créer des masks binaires
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()

        # Calculer intersection et union
        intersection = (pred_cls * target_cls).sum()
        union = (pred_cls + target_cls).sum() - intersection

        # Éviter la division par zéro
        iou = intersection / (union + 1e-6)
        iou_per_class.append(iou.item())

    return torch.tensor(iou_per_class)


class MetricsTracker:
    """
    Tracker pour accumuler les métriques IoU pendant l'entraînement/validation
    """

    def __init__(self, num_classes=3):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        """Réinitialiser les métriques"""
        self.iou_per_class = {f"class_{i}": [] for i in range(self.num_classes)}
        self.losses = []

    def update(self, pred, target, loss=None):
        """
        Mettre à jour les métriques avec un nouveau batch

        Args:
            pred (torch.Tensor): Prédictions [B, H, W] ou [B, C, H, W]
            target (torch.Tensor): Masques de référence [B, H, W]
            loss (float): Perte (optionnel)
        """

        # Si pred est logits, prendre argmax
        if len(pred.shape) == 4:
            pred = torch.argmax(pred, dim=1)

        # Déplacer sur CPU
        if pred.is_cuda:
            pred = pred.cpu()
        if target.is_cuda:
            target = target.cpu()

        # Calculer IoU pour chaque classe
        iou = compute_iou(pred, target, self.num_classes)

        for cls in range(self.num_classes):
            self.iou_per_class[f"class_{cls}"].append(iou[cls].item())

        # Loss
        if loss is not None:
            self.losses.append(loss)

    def get_metrics(self):
        """
        Obtenir les métriques moyennes

        Returns:
            dict: Métriques moyennes pour toutes les classes
        """

        metrics = {}

        # IoU moyen par classe
        for cls in range(self.num_classes):
            avg_iou = np.mean(self.iou_per_class[f"class_{cls}"])
            metrics[f"iou_class_{cls}"] = avg_iou

        metrics["iou_mean"] = np.mean([np.mean(v) for v in self.iou_per_class.values()])

        # Loss moyen
        if self.losses:
            metrics["loss"] = np.mean(self.losses)

        return metrics

    def __repr__(self):
        """Affichage des métriques"""
        metrics = self.get_metrics()

        lines = []
        lines.append("=" * 60)
        lines.append("MÉTRIQUES IoU")
        lines.append("=" * 60)

        for key, value in sorted(metrics.items()):
            lines.append(f"{key:25s}: {value:.6f}")

        return "\n".join(lines)


if __name__ == "__main__":
    print("=" * 70)
    print("TEST DES MÉTRIQUES IoU")
    print("=" * 70)

    # Créer des tensors de test
    batch_size = 4
    height, width = 160, 160
    num_classes = 3

    # Prédictions aléatoires
    pred = torch.randint(0, num_classes, (batch_size, height, width))

    # Masques de référence aléatoires
    target = torch.randint(0, num_classes, (batch_size, height, width))

    # Calculer IoU
    iou = compute_iou(pred, target, num_classes)
    print(f"\nIoU par classe: {iou}")
    print(f"IoU moyen: {iou.mean():.6f}")

    # Test du tracker
    print("\n" + "=" * 70)
    print("TEST METRICS TRACKER")
    print("=" * 70)

    tracker = MetricsTracker(num_classes=3)

    # Simuler 5 batches
    for i in range(5):
        pred = torch.randint(0, 3, (4, 160, 160))
        target = torch.randint(0, 3, (4, 160, 160))
        loss = np.random.random()

        tracker.update(pred, target, loss)

    print(tracker)
