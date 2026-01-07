"""
Architecture U-Net avec Attention Gates pour la segmentation sémantique
"""

import torch
import torch.nn as nn
from deep_datachallenge.models.conv_block import ConvBlock
from deep_datachallenge.models.attention_gate import AttentionGate


class UNet(nn.Module):
    """
    Architecture U-Net avec Attention Gates pour la segmentation sémantique

    Structure:
    - Encoder: 4 niveaux de downsampling avec skip connections
    - Bottleneck: 1 bloc au plus bas niveau
    - Decoder: 4 niveaux de upsampling avec Attention Gates sur les skip connections

    Les Attention Gates permettent au réseau d'apprendre quelles parties
    des skip connections sont pertinentes étant donné le signal du decoder.

    Args:
        in_channels (int): Nombre de canaux d'entrée (1 pour images en niveaux de gris)
        out_channels (int): Nombre de classes de sortie (3 pour BG, Casing, TIE)
        depth (int): Profondeur du réseau (nombre de niveaux). Par défaut 4.
        base_channels (int): Nombre de filtres de base. Par défaut 64.
        use_attention (bool): Utiliser les Attention Gates. Par défaut True.
    """

    def __init__(
        self, in_channels=1, out_channels=3, depth=4, base_channels=64, use_attention=True
    ):
        super(UNet, self).__init__()

        self.depth = depth
        self.base_channels = base_channels
        self.use_attention = use_attention

        # ENCODER (chemin descendant)
        self.encoders = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        in_ch = in_channels
        out_ch = base_channels

        for i in range(depth):
            self.encoders.append(ConvBlock(in_ch, out_ch, dropout_p=0.0))
            in_ch = out_ch
            out_ch = out_ch * 2

        # BOTTLENECK (goulot d'étranglement)
        self.bottleneck = ConvBlock(in_ch, out_ch, dropout_p=0.0)

        # DECODER (chemin montant)
        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        self.attention_gates = nn.ModuleList()

        in_ch = out_ch
        for i in range(depth):
            out_ch = in_ch // 2
            self.upconvs.append(nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2))

            # Attention Gate: g (upsampled) et skip connection
            # Après upconv, on a out_ch canaux
            # Skip connection a out_ch canaux (même niveau)
            if use_attention:
                self.attention_gates.append(
                    AttentionGate(
                        in_channels=out_ch,  # Skip connection channels
                        gating_channels=out_ch,  # Upsampled signal channels
                        inter_channels=out_ch // 2,
                    )
                )

            # Les skip connections doublent le nombre de canaux d'entrée (attention gate + concat)
            self.decoders.append(ConvBlock(in_ch, out_ch, dropout_p=0.0))
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

        # DECODER avec skip connections et Attention Gates
        skip_connections = skip_connections[::-1]  # Inverser pour l'ordre montant

        for i, (upconv, decoder) in enumerate(zip(self.upconvs, self.decoders)):
            x = upconv(x)

            # Récupérer la skip connection
            skip = skip_connections[i]
            if x.shape != skip.shape:
                # Adapter les dimensions si nécessaire
                skip = skip[:, :, : x.shape[2], : x.shape[3]]

            # Appliquer l'Attention Gate (si activé)
            if self.use_attention and self.attention_gates is not None:
                skip = self.attention_gates[i](skip, x)

            x = torch.cat([x, skip], dim=1)
            x = decoder(x)

        # OUTPUT
        x = self.final_conv(x)

        return x


if __name__ == "__main__":
    # Test du modèle
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("TEST U-NET AVEC ATTENTION GATES")
    print("=" * 70)

    # Test sans attention gates
    print("\n✓ U-Net BASELINE (sans Attention Gates)")
    model_baseline = UNet(in_channels=1, out_channels=3, depth=4, use_attention=False).to(device)
    total_params_baseline = sum(p.numel() for p in model_baseline.parameters() if p.requires_grad)
    print(f"  Nombre de paramètres: {total_params_baseline:,}")

    x = torch.randn(4, 1, 160, 160).to(device)
    output = model_baseline(x)
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {output.shape}")

    # Test avec attention gates
    print("\n✓ U-Net ATTENTION (avec Attention Gates)")
    model_attention = UNet(in_channels=1, out_channels=3, depth=4, use_attention=True).to(device)
    total_params_attention = sum(p.numel() for p in model_attention.parameters() if p.requires_grad)
    print(f"  Nombre de paramètres: {total_params_attention:,}")

    output = model_attention(x)
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {output.shape}")

    # Comparaison
    print("\n" + "=" * 70)
    print("COMPARAISON")
    print("=" * 70)
    print(f"Baseline:  {total_params_baseline:,} params")
    print(f"Attention: {total_params_attention:,} params")
    print(
        f"Overhead:  +{total_params_attention - total_params_baseline:,} params "
        f"(+{((total_params_attention - total_params_baseline) / total_params_baseline * 100):.2f}%)"
    )
    print(f"\n✓ U-Net test réussi!\n")
