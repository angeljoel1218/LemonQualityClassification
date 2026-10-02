"""
modelo_multitarea.py
----------------------
Pipeline multitarea propuesto (no es un clasificador directo): un backbone
convolucional compartido alimenta tres cabezales por vista, y la categoria final
la decide la regla comercial (regla_comercial.py), no una capa aprendida.

    Backbone compartido (ResNet50, preentrenado)
          |
          +--> Cabezal de segmentacion  (Mask R-CNN / YOLOv8-seg) -> mascara fruto + defectos
          +--> Cabezal de calibre       (regresion, 1 salida)     -> calibre en mm
          +--> Cabezal de color/defecto (regresion, 3-4 salidas)  -> L*, a*, b*, severidad

Cada cabezal corre POR VISTA (una de las 5 imagenes del fruto). La fusion de las 5
vistas en "un solo fruto" ocurre DESPUES de los cabezales, por agregacion estadistica
(atributos.py: mediana de calibre, media de color, maximo de severidad), no por
fusion de features como en el baseline (modelo_baseline.py). Esta es la diferencia
clave que se evalua en la ablacion E2 vs E5 (hipotesis H6: aporta mas descomponer
por atributos que clasificar de punta a punta).

Perdida de entrenamiento multitarea:
    L_total = w_seg * L_segmentacion + w_cal * L_calibre(MSE) + w_col * L_color(MSE)
    Los pesos w_* se ajustan en el piloto (ponderacion por incertidumbre,
    Kendall et al., 2018) para evitar que una tarea domine a las demas.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class BackboneCompartido(nn.Module):
    """ResNet50 preentrenado, usado como extractor de features para los 3 cabezales."""

    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.features = nn.Sequential(*list(resnet.children())[:-2])  # sin avgpool/fc
        self.out_channels = 2048

    def forward(self, imagen: torch.Tensor) -> torch.Tensor:
        return self.features(imagen)  # mapa de features (b, 2048, H', W')


class CabezalCalibre(nn.Module):
    """Regresion de calibre (mm) a partir del mapa de features de una vista."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, 1)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        x = self.pool(feature_map).flatten(1)
        return self.fc(x).squeeze(-1)  # (b,) calibre en mm


class CabezalColorDefecto(nn.Module):
    """Regresion de L*, a*, b* y severidad de defecto a partir del mapa de features."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, 4)  # L, a, b, severidad

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        x = self.pool(feature_map).flatten(1)
        return self.fc(x)  # (b, 4)


class PipelineMultitarea(nn.Module):
    """
    Nota: el cabezal de segmentacion (Mask R-CNN / YOLOv8-seg) se entrena e integra
    por separado -ver segmentacion.py-, porque usa su propia cabeza de region
    proposals; aqui solo se muestran los cabezales de regresion (calibre, color)
    que comparten el backbone.
    """

    def __init__(self):
        super().__init__()
        self.backbone = BackboneCompartido()
        self.cabezal_calibre = CabezalCalibre(self.backbone.out_channels)
        self.cabezal_color_defecto = CabezalColorDefecto(self.backbone.out_channels)

    def forward(self, imagen_vista: torch.Tensor) -> dict:
        """imagen_vista: (batch, 3, H, W) -- UNA vista a la vez (no las 5 juntas)."""
        feature_map = self.backbone(imagen_vista)
        return {
            "calibre_mm": self.cabezal_calibre(feature_map),
            "color_defecto": self.cabezal_color_defecto(feature_map),
        }


if __name__ == "__main__":
    modelo = PipelineMultitarea()
    entrada_dummy = torch.randn(2, 3, 224, 224)  # 2 vistas de ejemplo (no 2 frutos)
    salida = modelo(entrada_dummy)
    print("Calibre:", salida["calibre_mm"].shape)          # (2,)
    print("Color/defecto:", salida["color_defecto"].shape)  # (2, 4)
    # La agregacion entre las 5 vistas de un mismo fruto se hace despues,
    # con las funciones de atributos.py (mediana/media/maximo).
