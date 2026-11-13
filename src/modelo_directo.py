"""
modelo_directo.py
------------------
Clasificador DIRECTO end-to-end de aprendizaje profundo (baseline fuerte de H3 / ablación E2), Plan de tesis v4 sección 4.7.1:

    * Backbone ResNet-50 preentrenado en ImageNet con PESOS COMPARTIDOS: procesa cada una de las 5 vistas por separado.
    * Cada vista produce sus logits de las 5 categorías; la decisión del fruto es el PROMEDIO DE LOS LOGITS de las 5 vistas.
    * Recibe exactamente las mismas 5 imágenes que el sistema híbrido, con las mismas particiones, y NO recibe como entrada el
      calibre medido con vernier ni las etiquetas intermedias del panel (color, severidad, máscaras).
      El híbrido usa supervisión adicional (contornos y máscaras de defecto) que éste no usa: esa diferencia forma parte de la
      arquitectura comparada y se declara como limitación al interpretar H3.

(Antes: MVCNN con ResNet18 y max-pooling de features. Se reemplazó para ajustarse al plan v4.)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from src import config as C

N_VISTAS = len(C.VISTAS)            # 5
N_CLASES = len(C.CATEGORIAS)        # 5


class DirectoCincoVistas(nn.Module):
    def __init__(self, n_clases: int = N_CLASES, preentrenado: bool = True, dropout: float = 0.3):
        super().__init__()
        w = models.ResNet50_Weights.IMAGENET1K_V2 if preentrenado else None
        base = models.resnet50(weights=w)
        self.dim = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base                                   # compartido entre las 5 vistas
        self.cabeza = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.dim, n_clases))

    def logits_por_vista(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, V, 3, H, W) -> (B, V, K)."""
        b, v, c, h, w = x.shape
        f = self.backbone(x.reshape(b * v, c, h, w))
        return self.cabeza(f).reshape(b, v, -1)

    def forward(self, x: torch.Tensor, vistas_validas: torch.Tensor | None = None) -> torch.Tensor:
        """Logits del fruto = promedio de los logits de las vistas (opcionalmente solo las válidas: máscara (B, V) booleana)."""
        lv = self.logits_por_vista(x)
        if vistas_validas is None:
            return lv.mean(1)
        m = vistas_validas.float().unsqueeze(-1)
        return (lv * m).sum(1) / m.sum(1).clamp(min=1.0)


class FocalLoss(nn.Module):
    """Focal loss multiclase (Lin et al., 2017). gamma = 0 equivale a la entropía cruzada (con pesos de clase si se dan).
    `alpha`: escalar (peso global) o tensor por clase."""
    def __init__(self, gamma: float = 2.0, alpha=None, pesos_clase: torch.Tensor | None = None):
        super().__init__()
        self.gamma, self.alpha = gamma, alpha
        self.register_buffer("w", pesos_clase if pesos_clase is not None else None)

    def forward(self, logits, y):
        ce = F.cross_entropy(logits, y, weight=self.w, reduction="none")
        pt = torch.exp(-F.cross_entropy(logits, y, reduction="none"))
        loss = (1 - pt) ** self.gamma * ce
        if self.alpha is not None:
            a = self.alpha if not torch.is_tensor(self.alpha) else self.alpha[y]
            loss = a * loss
        return loss.mean()


if __name__ == "__main__":
    m = DirectoCincoVistas(preentrenado=False)
    print(m(torch.randn(2, N_VISTAS, 3, 224, 224)).shape)      # (2, 5)
