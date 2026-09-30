"""Rutas, clases y utilidades compartidas del proyecto."""
from pathlib import Path

import torch
from torchvision import transforms

RAIZ = Path(__file__).resolve().parent.parent
DATA = RAIZ / "data"
SALIDAS = RAIZ / "outputs"
MODELOS = RAIZ / "models"
RUTA_MODELO = MODELOS / "cnn_frijol.pt"
SEMILLA = 42

# Orden alfabético = orden que asigna torchvision.datasets.ImageFolder
CLASES = ["angular_leaf_spot", "bean_rust", "healthy"]
NOMBRES_ES = {
    "angular_leaf_spot": "Mancha angular",
    "bean_rust": "Roya del frijol",
    "healthy": "Sana",
}

TAM = 224
MEDIA, DESV = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # normalización ImageNet

tf_entrenamiento = transforms.Compose([
    transforms.RandomResizedCrop(TAM, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEDIA, DESV),
])
tf_evaluacion = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(TAM),
    transforms.ToTensor(),
    transforms.Normalize(MEDIA, DESV),
])


def dispositivo() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")  # GPU de Mac (Apple Silicon)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
