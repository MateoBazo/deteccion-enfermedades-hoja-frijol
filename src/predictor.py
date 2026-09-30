"""Carga de la CNN, predicción y Grad-CAM (mapa de calor de las zonas que usa la red)."""
import matplotlib.cm as cm
import numpy as np
import torch
from PIL import Image

from config import CLASES, RUTA_MODELO, dispositivo, tf_evaluacion
from entrenar_cnn import crear_modelo


class Predictor:
    def __init__(self):
        self.disp = dispositivo()
        self.modelo = crear_modelo(preentrenado=False)
        self.modelo.load_state_dict(torch.load(RUTA_MODELO, map_location="cpu"))
        self.modelo.to(self.disp).eval()
        # Ganchos en la última capa convolucional para Grad-CAM
        self._act, self._grad = None, None
        capa = self.modelo.layer4
        capa.register_forward_hook(lambda m, i, o: setattr(self, "_act", o))
        capa.register_full_backward_hook(lambda m, gi, go: setattr(self, "_grad", go[0]))

    def predecir(self, imagen: Image.Image):
        """Devuelve (clase, {clase: probabilidad}, mapa_gradcam 224x224 en [0,1])."""
        x = tf_evaluacion(imagen.convert("RGB")).unsqueeze(0).to(self.disp)
        self.modelo.zero_grad()
        logits = self.modelo(x)
        probs = torch.softmax(logits, 1)[0]
        idx = int(probs.argmax())
        logits[0, idx].backward()
        pesos = self._grad.mean(dim=(2, 3), keepdim=True)            # importancia de cada canal
        cam = torch.relu((pesos * self._act).sum(1)).squeeze(0)      # combinación ponderada
        cam = torch.nn.functional.interpolate(cam[None, None], size=(224, 224), mode="bilinear")[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return CLASES[idx], {c: float(p) for c, p in zip(CLASES, probs.detach().cpu())}, cam.detach().cpu().numpy()


def superponer(imagen: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Superpone el mapa Grad-CAM sobre la imagen recortada igual que la ve la red."""
    lado = min(imagen.size)
    izq, arr = (imagen.width - lado) // 2, (imagen.height - lado) // 2
    base = imagen.convert("RGB").crop((izq, arr, izq + lado, arr + lado)).resize((224, 224))
    calor = Image.fromarray((cm.jet(cam)[..., :3] * 255).astype(np.uint8))
    return Image.blend(base, calor, alpha)
