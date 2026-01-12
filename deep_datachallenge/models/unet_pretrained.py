"""
Architecture U-Net avec encodeur ResNet pré-entraîné sur ImageNet

Avantages du pretraining:
- L'encodeur ResNet a déjà appris des features générales (textures, formes, etc.)
- Entraînement plus rapide
- Meilleurs résultats avec moins de données
- Meilleure généralisation
"""

import segmentation_models_pytorch as smp


def create_unet_pretrained(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=1,
    classes=3,
    activation=None,
):
    """
    Créer un UNet avec encodeur pré-entraîné sur ImageNet

    Args:
        encoder_name (str): Nom de l'encodeur (resnet18, resnet34, resnet50, etc.)
        encoder_weights (str): Poids pré-entraînés ('imagenet' ou None)
        in_channels (int): Nombre de canaux d'entrée (1 pour grayscale)
        classes (int): Nombre de classes de sortie (3 pour BG, Casing, TIE)
        activation (str): Fonction d'activation finale (None pour logits bruts)

    Returns:
        torch.nn.Module: Modèle UNet pré-entraîné
    """
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=activation,
    )
    return model


def freeze_encoder(model):
    """
    Geler tous les paramètres de l'encodeur
    Utile pour la phase 1 d'entraînement où on veut garder les features pré-apprises

    Args:
        model: Modèle UNet de smp
    """
    for param in model.encoder.parameters():
        param.requires_grad = False
    print("Encodeur gelé (pas d'entraînement)")


def unfreeze_encoder(model):
    """
    Dégeler tous les paramètres de l'encodeur
    Utile pour la phase 2 et 3 d'entraînement (fine-tuning)

    Args:
        model: Modèle UNet de smp
    """
    for param in model.encoder.parameters():
        param.requires_grad = True
    print("Encodeur dégélé (entraînement activé)")


def freeze_encoder_body_unfreeze_head(model):
    """
    Geler le corps de l'encodeur mais dégeler la tête (dernières couches)
    Approche intermédiaire pour fine-tuning progressif

    Args:
        model: Modèle UNet de smp
    """
    # Geler tout d'abord
    for param in model.encoder.parameters():
        param.requires_grad = False

    # Dégeler les dernières couches (stage4 = dernière)
    if hasattr(model.encoder, "layer4"):
        for param in model.encoder.layer4.parameters():
            param.requires_grad = True
    print("Encodeur partiellement dégélé (dernières couches)")
