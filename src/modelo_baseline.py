"""
modelo_baseline.py
--------------------
Clasificador directo end-to-end (baseline, ablacion E2): recibe las 5 vistas de un
mismo fruto y las "funde" en una sola representación antes de clasificar, en vez de
pasar por segmentacion + atributos + regla comercial.

Arquitectura: Multi-View CNN (inspirada en Su et al., 2015 - MVCNN).
    1. Un backbone CNN (ResNet18/34, preentrenado en ImageNet) CON PESOS COMPARTIDOS
       procesa cada una de las 5 vistas por separado -> 5 vectores de features.
    2. Los 5 vectores se funden en uno solo por fruto con max-pooling (o
       promedio) a traves de las vistas: esta es la operacion que "convierte 5
       imagenes en una sola" representacion.
    3. Ese vector fusionado pasa por una cabeza totalmente conectada que produce
       la categoria final (5 clases: Extra, Primera, Segunda, Tercera, Fabrica).

Se entrena con las mismas 3 semillas (0, 1, 2) y la misma particion 70-15-15 que el
pipeline propuesto, para que la comparacion (McNemar / bootstrap pareado) sea valida.
"""

import torch
import torch.nn as nn
import torchvision.models as models

N_VISTAS = 5
N_CLASES = 5  # Extra, Primera, Segunda, Tercera, Fabrica


class MultiViewCNNBaseline(nn.Module):
    def __init__(self, n_clases: int = N_CLASES, pooling: str = "max"):
        super().__init__()
        assert pooling in {"max", "mean"}
        self.pooling = pooling

        # Backbone compartido: los 5 vistas pasan por LA MISMA red (pesos compartidos).
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()  # se queda solo con el vector de features
        self.backbone = backbone

        self.clasificador = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, n_clases),
        )

    def forward(self, vistas: torch.Tensor) -> torch.Tensor:
        """
        vistas: tensor de forma (batch, N_VISTAS, 3, H, W) -- las 5 imagenes del fruto.
        """
        b, v, c, h, w = vistas.shape
        assert v == N_VISTAS, f"Se esperaban {N_VISTAS} vistas, llegaron {v}"

        # Aplana (batch, vistas) para pasar todas las imagenes por el mismo backbone
        # en un solo forward, y luego separa de vuelta por fruto.
        vistas_flat = vistas.view(b * v, c, h, w)
        features_flat = self.backbone(vistas_flat)                # (b*v, feature_dim)
        features = features_flat.view(b, v, self.feature_dim)     # (b, v, feature_dim)

        # Fusion entre vistas: esto es lo que convierte "5 imagenes" en "1 fruto".
        if self.pooling == "max":
            fruto_feature, _ = torch.max(features, dim=1)         # (b, feature_dim)
        else:
            fruto_feature = torch.mean(features, dim=1)           # (b, feature_dim)

        return self.clasificador(fruto_feature)                   # (b, n_clases)


if __name__ == "__main__":
    modelo = MultiViewCNNBaseline()
    entrada_dummy = torch.randn(2, N_VISTAS, 3, 224, 224)  # 2 frutos de ejemplo
    salida = modelo(entrada_dummy)
    print("Salida (logits por clase):", salida.shape)  # esperado: (2, 5)
