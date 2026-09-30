"""Paso 1: Análisis exploratorio del dataset de imágenes."""
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from config import CLASES, DATA, NOMBRES_ES, SALIDAS, SEMILLA

COLORES = {"angular_leaf_spot": "#C9A227", "bean_rust": "#B5523B", "healthy": "#3A8D5B"}


def inventario() -> pd.DataFrame:
    filas = []
    for split in ["train", "validation", "test"]:
        for clase in CLASES:
            for ruta in sorted((DATA / split / clase).glob("*.jpg")):
                filas.append({"split": split, "clase": clase, "ruta": ruta})
    return pd.DataFrame(filas)


def main():
    SALIDAS.mkdir(exist_ok=True)
    random.seed(SEMILLA)
    df = inventario()

    conteo = df.pivot_table(index="clase", columns="split", aggfunc="size")[["train", "validation", "test"]]
    conteo.index = [NOMBRES_ES[c] for c in conteo.index]
    print(f"Total de imágenes: {len(df)}\n")
    print(conteo.to_string(), "\n")

    # Tamaños de imagen (muestra)
    muestra = df.sample(150, random_state=SEMILLA)
    tamanos = muestra["ruta"].apply(lambda p: Image.open(p).size).value_counts()
    print("Tamaños de imagen más comunes (muestra de 150):")
    print(tamanos.head().to_string(), "\n")

    # 1. Distribución de clases por split
    fig, ax = plt.subplots(figsize=(7, 4))
    conteo.plot(kind="bar", ax=ax, color=["#4C72B0", "#8C8C8C", "#DD8452"], rot=0)
    ax.set_title("Imágenes por clase y partición"); ax.set_ylabel("Cantidad"); ax.set_xlabel("")
    fig.tight_layout(); fig.savefig(SALIDAS / "01_distribucion_clases.png", dpi=120); plt.close(fig)

    # 2. Ejemplos de cada clase
    fig, axes = plt.subplots(3, 5, figsize=(13, 8))
    for fila, clase in enumerate(CLASES):
        rutas = random.sample(list(df.query("split == 'train' and clase == @clase").ruta), 5)
        for col, ruta in enumerate(rutas):
            ax = axes[fila, col]
            ax.imshow(Image.open(ruta)); ax.axis("off")
            if col == 0:
                ax.set_title(NOMBRES_ES[clase], loc="left", fontsize=13, fontweight="bold")
    fig.suptitle("Ejemplos del dataset (entrenamiento)", fontsize=15)
    fig.tight_layout(); fig.savefig(SALIDAS / "02_ejemplos.png", dpi=110); plt.close(fig)

    # 3. Color: histograma de tono (Hue) y proporción de píxeles "verdes" por clase
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    resumen = []
    for clase in CLASES:
        rutas = df.query("split == 'train' and clase == @clase").ruta.sample(80, random_state=SEMILLA)
        hues, verdes = [], []
        for ruta in rutas:
            hsv = np.asarray(Image.open(ruta).convert("HSV").resize((128, 128))).reshape(-1, 3)
            hsv = hsv[hsv[:, 1] > 60]  # descarta fondo poco saturado
            hues.append(hsv[:, 0])
            verdes.append(((hsv[:, 0] > 45) & (hsv[:, 0] < 110)).mean())
        h = np.concatenate(hues)
        axes[0].hist(h, bins=64, range=(0, 255), density=True, histtype="step", lw=2,
                     color=COLORES[clase], label=NOMBRES_ES[clase])
        resumen.append({"clase": NOMBRES_ES[clase], "pct_verde": np.mean(verdes) * 100})
    axes[0].set_title("Distribución de tono (canal H de HSV)")
    axes[0].set_xlabel("Tono (0-255)  —  amarillo/marrón ← → verde"); axes[0].legend()
    r = pd.DataFrame(resumen)
    axes[1].bar(r.clase, r.pct_verde, color=[COLORES[c] for c in CLASES])
    axes[1].set_title("% de píxeles verdes (promedio por imagen)"); axes[1].set_ylabel("%")
    fig.tight_layout(); fig.savefig(SALIDAS / "03_analisis_color.png", dpi=120); plt.close(fig)
    print("Porcentaje promedio de píxeles verdes por clase:")
    print(r.round(1).to_string(index=False))

    print(f"\nGráficos guardados en {SALIDAS}/")


if __name__ == "__main__":
    main()
