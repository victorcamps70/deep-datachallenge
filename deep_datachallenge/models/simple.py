"""
Modèle simple de segmentation (baseline pour comparaison avec U-Net)
"""

import torch.nn as nn


class SimpleSegmentationNet(nn.Module):
    """
    Réseau de segmentation simple (FCN-style) pour baseline

    Architecture:
    - 4 blocs Conv+ReLU+MaxPool (encoder)
    - 4 blocs ConvTranspose (decoder)
    - Pas de skip connections (contrairement à U-Net)

    Args:
        in_channels (int): Nombre de canaux d'entrée
        out_channels (int): Nombre de classes de sortie
    """

    def __init__(self, in_channels=1, out_channels=3):
        super(SimpleSegmentationNet, self).__init__()

        # ENCODER
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.enc4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # BOTTLENECK
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # DECODER
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # OUTPUT
        self.final = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # ENCODER
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # BOTTLENECK
        b = self.bottleneck(e4)

        # DECODER (sans skip connections)
        d4 = self.dec4(b)
        d3 = self.dec3(d4)
        d2 = self.dec2(d3)
        d1 = self.dec1(d2)

        # OUTPUT
        out = self.final(d1)

        return out


if __name__ == "__main__":
    # Test du modèle
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("TEST SIMPLE SEGMENTATION NET")
    print("=" * 70)

    model = SimpleSegmentationNet(in_channels=1, out_channels=3).to(device)

    # Compter les paramètres
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Nombre de paramètres: {total_params:,}")

    # Test forward pass
    x = torch.randn(4, 1, 160, 160).to(device)
    output = model(x)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"✓ SimpleSegmentationNet test réussi!\n")
