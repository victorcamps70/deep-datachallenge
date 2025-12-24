"""
Dataset PyTorch pour la segmentation d'images ultrasoniques
"""

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class WellSegmentationDataset(Dataset):
    """
    Dataset PyTorch pour charger les images et masques d'entraînement

    Args:
        image_dir (Path): Répertoire contenant les images .npy
        labels_df (pd.DataFrame): DataFrame avec les labels (index = noms de patches)
        preprocessor: Instance de ImagePreprocessor pour prétraiter les images
        augment (bool): Appliquer l'augmentation de données (par défaut False)
        augment_transform: Transformations PyTorch à appliquer (rotation, flip, etc.)
    """

    def __init__(self, image_dir, labels_df, preprocessor, augment=False, augment_transform=None):
        self.image_dir = Path(image_dir)
        self.labels_df = labels_df
        self.preprocessor = preprocessor
        self.patch_names = labels_df.index.tolist()
        self.augment = augment
        self.augment_transform = augment_transform

    def __len__(self):
        return len(self.patch_names)

    def __getitem__(self, idx):
        patch_name = self.patch_names[idx]

        # Charger l'image
        image_path = self.image_dir / f"{patch_name}.npy"
        image = np.load(image_path)

        # Charger le masque
        label_values = np.array([v for v in self.labels_df.loc[patch_name] if v != -1])
        mask = label_values.reshape(160, -1)

        # Appliquer le preprocessing
        image_processed, mask_processed = self.preprocessor.full_preprocessing(
            image, mask, normalize=True, fill_missing=True
        )

        # Convertir en tensors PyTorch
        image_tensor = torch.from_numpy(image_processed).float().unsqueeze(0)  # [1, 160, 160]
        mask_tensor = torch.from_numpy(mask_processed).long()  # [160, 160]

        # Appliquer l'augmentation si nécessaire
        if self.augment and self.augment_transform is not None:
            # Les augmentations de torchvision s'attendent à PIL ou tensors float [0,1]
            # Notre image_tensor est déjà float [0,1], et mask_tensor int [0,2]
            image_tensor = self.augment_transform(image_tensor)

        return {"image": image_tensor, "mask": mask_tensor, "patch_name": patch_name}


def get_train_val_split(labels_df, train_ratio=0.8, stratify_by_well=True, random_state=42):
    """
    Divise les données en ensembles d'entraînement et validation

    Args:
        labels_df (pd.DataFrame): DataFrame avec les labels
        train_ratio (float): Proportion d'entraînement (par défaut 0.8)
        stratify_by_well (bool): Stratifier par numéro de puits pour éviter la fuite de données
        random_state (int): Seed pour reproductibilité

    Returns:
        train_indices (list): Indices pour l'entraînement
        val_indices (list): Indices pour la validation
    """

    np.random.seed(random_state)

    if stratify_by_well:
        # Grouper par numéro de puits
        well_groups = {}
        for i, patch_name in enumerate(labels_df.index):
            well_num = patch_name.split("_")[1]  # Extraire "X" de "well_X"
            if well_num not in well_groups:
                well_groups[well_num] = []
            well_groups[well_num].append(i)

        # Diviser chaque groupe puits
        train_indices = []
        val_indices = []

        for well_num, indices in well_groups.items():
            np.random.shuffle(indices)
            split_point = int(len(indices) * train_ratio)
            train_indices.extend(indices[:split_point])
            val_indices.extend(indices[split_point:])

        print(f"Split stratifié par puits:")
        print(f"  • Train: {len(train_indices)} patches")
        print(f"  • Val: {len(val_indices)} patches")
    else:
        # Split aléatoire simple
        indices = np.arange(len(labels_df))
        np.random.shuffle(indices)
        split_point = int(len(indices) * train_ratio)
        train_indices = indices[:split_point].tolist()
        val_indices = indices[split_point:].tolist()

    return train_indices, val_indices


def compute_class_weights(labels_df):
    """
    Calcule les poids des classes pour équilibrer l'entraînement
    Utilise la formule: weight = 1 / (proportion de la classe)

    Args:
        labels_df (pd.DataFrame): DataFrame avec les labels

    Returns:
        torch.Tensor: Poids pour chaque classe [w0, w1, w2]
    """

    class_counts = {0: 0, 1: 0, 2: 0}
    total_pixels = 0

    print("\nCalcul des poids des classes...")

    for i, patch_name in enumerate(labels_df.index):
        if (i + 1) % 500 == 0:
            print(f"  Traitement {i+1}/{len(labels_df)}...", end="\r")

        label_values = np.array([v for v in labels_df.loc[patch_name] if v != -1])

        for class_id in [0, 1, 2]:
            count = np.sum(label_values == class_id)
            class_counts[class_id] += count

        total_pixels += len(label_values)

    print(f"  Traitement {len(labels_df)}/{len(labels_df)}... Fait!     \n")

    # Calculer les proportions et les poids
    print("Distribution des classes:")
    weights = []
    for class_id in [0, 1, 2]:
        proportion = class_counts[class_id] / total_pixels
        weight = 1.0 / proportion if proportion > 0 else 1.0
        weights.append(weight)

        class_names = {0: "Arrière-plan", 1: "Casing", 2: "TIE"}
        print(f"  • {class_names[class_id]}: {proportion*100:.2f}% → poids = {weight:.4f}")

    # Normaliser pour que la somme = 3 (optionnel mais utile)
    total_weight = sum(weights)
    weights = [w / total_weight * 3 for w in weights]

    print(f"\nPoids normalisés: {[f'{w:.4f}' for w in weights]}")

    return torch.tensor(weights, dtype=torch.float32)


def create_dataloaders(
    labels_df,
    image_dir,
    preprocessor,
    batch_size=32,
    num_workers=0,
    augment_train=True,
    random_state=42,
):
    """
    Crée les DataLoaders pour l'entraînement et la validation

    Args:
        labels_df (pd.DataFrame): DataFrame avec les labels
        image_dir (Path): Répertoire des images
        preprocessor: Instance de ImagePreprocessor
        batch_size (int): Taille des batches
        num_workers (int): Nombre de workers pour le chargement (0 = main process)
        augment_train (bool): Appliquer l'augmentation aux données d'entraînement
        random_state (int): Seed pour reproductibilité

    Returns:
        train_loader (DataLoader): DataLoader pour l'entraînement
        val_loader (DataLoader): DataLoader pour la validation
        class_weights (torch.Tensor): Poids pour l'équilibrage des classes
    """

    # Diviser les données
    train_indices, val_indices = get_train_val_split(
        labels_df, train_ratio=0.8, stratify_by_well=True, random_state=random_state
    )

    # Augmentation pour l'entraînement
    augment_transform = None
    if augment_train:
        augment_transform = transforms.Compose(
            [
                transforms.RandomRotation(degrees=10),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

    # Créer les datasets
    train_labels = labels_df.iloc[train_indices]
    val_labels = labels_df.iloc[val_indices]

    train_dataset = WellSegmentationDataset(
        image_dir, train_labels, preprocessor, augment=True, augment_transform=augment_transform
    )

    val_dataset = WellSegmentationDataset(
        image_dir, val_labels, preprocessor, augment=False, augment_transform=None
    )

    # Créer les DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    # Calculer les poids des classes
    class_weights = compute_class_weights(labels_df)

    print(f"\nDataLoaders créés:")
    print(f"  • Train: {len(train_loader)} batches de {batch_size}")
    print(f"  • Val: {len(val_loader)} batches de {batch_size}\n")

    return train_loader, val_loader, class_weights


if __name__ == "__main__":
    # Test du dataset
    import pandas as pd
    from deep_datachallenge.preprocessing import ImagePreprocessor

    print("=" * 70)
    print("TEST DATASET")
    print("=" * 70)

    # Charger les données
    data_dir = Path("data")
    y_train_file = data_dir / "y_train_labels" / "Y_train_T9NrBYo.csv"
    x_train_dir = data_dir / "x_train_images"

    y_train = pd.read_csv(y_train_file, index_col=0)
    preprocessor = ImagePreprocessor(target_size=(160, 160))

    # Créer les DataLoaders
    train_loader, val_loader, class_weights = create_dataloaders(
        y_train, x_train_dir, preprocessor, batch_size=8, augment_train=True
    )

    # Tester le chargement
    batch = next(iter(train_loader))

    print("Premier batch:")
    print(f"  • Images: {batch['image'].shape}")
    print(f"  • Masques: {batch['mask'].shape}")
    print(f"  • Noms de patches: {batch['patch_name'][:3]}")
    print(f"\nDataset test réussi!")
