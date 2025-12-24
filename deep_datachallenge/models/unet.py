"""
Architecture U-Net pour la segmentation sémantique
"""

import torch
import torch.nn as nn
from deep_datachallenge.models.conv_block import ConvBlock


class UNet(nn.Module):
    """
    Architecture U-Net pour la segmentation sémantique

    Structure:
    - Encoder: 4 niveaux de downsampling avec skip connections
    - Bottleneck: 1 bloc au plus bas niveau
    - Decoder: 4 niveaux de upsampling avec concatenation des skip connections

    Args:
        in_channels (int): Nombre de canaux d'entrée (1 pour images en niveaux de gris)
        out_channels (int): Nombre de classes de sortie (3 pour BG, Casing, TIE)
        depth (int): Profondeur du réseau (nombre de niveaux). Par défaut 4.
        base_channels (int): Nombre de filtres de base. Par défaut 64.
    """

    def __init__(self, in_channels=1, out_channels=3, depth=4, base_channels=64):
        super(UNet, self).__init__()

        self.depth = depth
        self.base_channels = base_channels

        # ENCODER (chemin descendant)
        self.encoders = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        in_ch = in_channels
        out_ch = base_channels

        for i in range(depth):
            self.encoders.append(ConvBlock(in_ch, out_ch))
            in_ch = out_ch
            out_ch = out_ch * 2

        # BOTTLENECK (goulot d'étranglement)
        self.bottleneck = ConvBlock(in_ch, out_ch)

        # DECODER (chemin montant)
        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()

        in_ch = out_ch
        for i in range(depth):
            out_ch = in_ch // 2
            self.upconvs.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2))
            # Les skip connections doublent le nombre de canaux d'entrée
            self.decoders.append(ConvBlock(in_ch, out_ch))
            in_ch = out_ch

        # OUTPUT
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # ENCODER avec sauvegarde des skip connections
        skip_connections = []

        for encoder in self.encoders:
            x = encoder(x)
            skip_connections.append(x)
            x = self.pool(x)

        # BOTTLENECK
        x = self.bottleneck(x)

        # DECODER avec skip connections
        skip_connections = skip_connections[::-1]  # Inverser pour l'ordre montant

        for i, (upconv, decoder) in enumerate(zip(self.upconvs, self.decoders)):
            x = upconv(x)

            # Concatener la skip connection (dimensionner si nécessaire)
            skip = skip_connections[i]
            if x.shape != skip.shape:
                # Adapter les dimensions si nécessaire
                skip = skip[:, :, : x.shape[2], : x.shape[3]]

            x = torch.cat([x, skip], dim=1)
            x = decoder(x)

        # OUTPUT
        x = self.final_conv(x)

        return x


if __name__ == "__main__":
    # Test du modèle
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("TEST U-NET")
    print("=" * 70)

    model = UNet(in_channels=1, out_channels=3, depth=4).to(device)

    # Compter les paramètres
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Nombre de paramètres U-Net: {total_params:,}")

    # Test forward pass
    x = torch.randn(4, 1, 160, 160).to(device)  # Batch de 4 images
    output = model(x)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"✓ U-Net test réussi!\n")
