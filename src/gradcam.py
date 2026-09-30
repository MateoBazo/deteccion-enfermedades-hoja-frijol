"""Paso 3: Interpretabilidad — Grad-CAM sobre imágenes de test (aciertos y errores)."""
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from config import NOMBRES_ES, RAIZ, SALIDAS, SEMILLA
from predictor import Predictor, superponer


def main():
    pred = Predictor()
    df = pd.read_csv(SALIDAS / "predicciones_cnn_test.csv")
    errores = df[df.real != df.pred_cnn]
    aciertos = df[df.real == df.pred_cnn].groupby("real").sample(2, random_state=SEMILLA)
    filas = pd.concat([aciertos, errores]).reset_index(drop=True)

    fig, axes = plt.subplots(len(filas), 2, figsize=(6, 3 * len(filas)))
    for i, f in filas.iterrows():
        img = Image.open(RAIZ / f.ruta)
        clase, probs, cam = pred.predecir(img)
        ok = "✓" if f.real == clase else "✗"
        axes[i, 0].imshow(img); axes[i, 0].axis("off")
        axes[i, 0].set_title(f"Real: {NOMBRES_ES[f.real]}", fontsize=10)
        axes[i, 1].imshow(superponer(img, cam)); axes[i, 1].axis("off")
        axes[i, 1].set_title(f"{ok} CNN: {NOMBRES_ES[clase]} ({probs[clase]:.0%})", fontsize=10,
                             color="green" if ok == "✓" else "red")
    fig.suptitle("Grad-CAM: zonas en las que se fija la CNN", fontsize=13, y=0.999)
    fig.tight_layout(rect=(0, 0, 1, 0.99)); fig.savefig(SALIDAS / "06_gradcam.png", dpi=100); plt.close(fig)
    print(f"{len(aciertos)} aciertos y {len(errores)} errores visualizados en outputs/06_gradcam.png")


if __name__ == "__main__":
    main()
