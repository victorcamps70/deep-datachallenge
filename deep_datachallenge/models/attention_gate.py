"""
Attention Gates pour U-Net
Basé sur "Attention U-Net: Learning Where to Look for the Pancreas"
(Ozan Oktay et al., 2018)

Les attention gates permettent au réseau de se concentrer sur les régions
pertinentes en pondérant les skip connections dynamiquement.
"""

import torch
import torch.nn as nn


class AttentionGate(nn.Module):
    """
    Attention Gate pour U-Net

    Permet au réseau d'apprendre quelles parties des skip connections
    sont pertinentes étant donné le signal "gating" du decoder.

    Args:
        in_channels (int): Nombre de canaux de la skip connection
        gating_channels (int): Nombre de canaux du signal de gating
        inter_channels (int): Nombre de canaux internes (bottleneck)
    """

    def __init__(self, in_channels, gating_channels, inter_channels=None):
        super(AttentionGate, self).__init__()

        if inter_channels is None:
            inter_channels = in_channels // 2

        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels

        # Branches de convolution pour le signal x (skip connection)
        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
        )

        # Branches de convolution pour le signal g (gating signal)
        self.conv_g = nn.Sequential(
            nn.Conv2d(gating_channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
        )

        # Couche de fusion et attention
        self.relu = nn.ReLU(inplace=True)
        self.conv_psi = nn.Sequential(
            nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),  # Sigmoid pour obtenir des poids entre 0 et 1
        )

    def forward(self, x, g):
        """
        Forward pass

        Args:
            x (torch.Tensor): Skip connection (B, in_channels, H, W)
            g (torch.Tensor): Gating signal du decoder (B, gating_channels, H, W)
                             NOTE: Doit avoir les mêmes dimensions spatiales que x

        Returns:
            torch.Tensor: Skip connection pondérée (B, in_channels, H, W)
        """

        # Traiter x et g séparément
        x_conv = self.conv_x(x)  # (B, inter_channels, H, W)
        g_conv = self.conv_g(g)  # (B, inter_channels, H, W)

        # Fusion additive
        psi_input = self.relu(x_conv + g_conv)  # (B, inter_channels, H, W)

        # Générer les poids d'attention
        psi = self.conv_psi(psi_input)  # (B, 1, H, W)

        # Appliquer l'attention: multiplier x par les poids
        out = x * psi  # Broadcasting: (B, in_channels, H, W) * (B, 1, H, W)

        return out


class ChannelAttention(nn.Module):
    """
    Attention au niveau des canaux (Squeeze-and-Excitation)
    Alternative plus légère aux AttentionGates

    Args:
        channels (int): Nombre de canaux
        reduction (int): Facteur de réduction (défaut: 16)
    """

    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): (B, C, H, W)

        Returns:
            torch.Tensor: (B, C, H, W) avec attention appliquée
        """
        b, c, _, _ = x.size()
        se = self.avg_pool(x)  # (B, C, 1, 1)
        se = self.fc(se)  # (B, C, 1, 1)
        return x * se  # Broadcasting


if __name__ == "__main__":
    # Test des modules
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("TEST ATTENTION GATES")
    print("=" * 70)

    # Test AttentionGate
    print("\n✓ Testing AttentionGate...")
    gate = AttentionGate(in_channels=64, gating_channels=128, inter_channels=32).to(device)

    x = torch.randn(4, 64, 32, 32).to(device)  # Skip connection
    g = torch.randn(4, 128, 32, 32).to(device)  # Gating signal

    out = gate(x, g)
    print(f"  Input (skip connection): {x.shape}")
    print(f"  Input (gating signal):   {g.shape}")
    print(f"  Output (gated):          {out.shape}")
    print(f"  ✓ Output shape correct!")

    # Compter les paramètres
    params = sum(p.numel() for p in gate.parameters() if p.requires_grad)
    print(f"  Parameters: {params:,}")

    # Test ChannelAttention
    print("\n✓ Testing ChannelAttention...")
    se = ChannelAttention(channels=64).to(device)
    x = torch.randn(4, 64, 32, 32).to(device)
    out = se(x)
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")
    print(f"  ✓ Output shape correct!")

    params = sum(p.numel() for p in se.parameters() if p.requires_grad)
    print(f"  Parameters: {params:,}")

    print("\n" + "=" * 70)
    print("✓ All tests passed!")
    print("=" * 70 + "\n")
