"""Paso 2: Entrenamiento de una CNN (ResNet-18) con transfer learning."""
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models

from config import (CLASES, DATA, RAIZ, MODELOS, NOMBRES_ES, RUTA_MODELO, SALIDAS, SEMILLA, dispositivo,
                    tf_entrenamiento, tf_evaluacion)

EPOCAS = 10
LOTE = 32


def crear_modelo(preentrenado: bool = True) -> nn.Module:
    pesos = models.ResNet18_Weights.IMAGENET1K_V1 if preentrenado else None
    modelo = models.resnet18(weights=pesos)
    modelo.fc = nn.Linear(modelo.fc.in_features, len(CLASES))  # nueva "cabeza" de 3 clases
    return modelo


def evaluar(modelo, cargador, disp):
    modelo.eval()
    etiquetas, preds, perdida = [], [], 0.0
    with torch.no_grad():
        for x, y in cargador:
            x, y = x.to(disp), y.to(disp)
            salida = modelo(x)
            perdida += nn.functional.cross_entropy(salida, y, reduction="sum").item()
            preds += salida.argmax(1).cpu().tolist()
            etiquetas += y.cpu().tolist()
    acc = sum(p == e for p, e in zip(preds, etiquetas)) / len(etiquetas)
    return perdida / len(etiquetas), acc, etiquetas, preds


def main():
    torch.manual_seed(SEMILLA)
    SALIDAS.mkdir(exist_ok=True); MODELOS.mkdir(exist_ok=True)
    disp = dispositivo()
    print(f"Dispositivo: {disp}")

    ds = {s: datasets.ImageFolder(DATA / s, tf_entrenamiento if s == "train" else tf_evaluacion)
          for s in ["train", "validation", "test"]}
    assert ds["train"].classes == CLASES
    cargas = {s: DataLoader(d, batch_size=LOTE, shuffle=(s == "train"), num_workers=0)
              for s, d in ds.items()}

    modelo = crear_modelo().to(disp)
    optim = torch.optim.AdamW(modelo.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCAS)

    historial, mejor_acc = [], 0.0
    for epoca in range(1, EPOCAS + 1):
        t0 = time.time()
        modelo.train()
        perd_tr, aciertos, n = 0.0, 0, 0
        for x, y in cargas["train"]:
            x, y = x.to(disp), y.to(disp)
            optim.zero_grad()
            salida = modelo(x)
            perdida = nn.functional.cross_entropy(salida, y)
            perdida.backward(); optim.step()
            perd_tr += perdida.item() * len(y)
            aciertos += (salida.argmax(1) == y).sum().item(); n += len(y)
        sched.step()
        perd_val, acc_val, _, _ = evaluar(modelo, cargas["validation"], disp)
        historial.append({"epoca": epoca, "perdida_train": perd_tr / n, "acc_train": aciertos / n,
                          "perdida_val": perd_val, "acc_val": acc_val})
        marca = ""
        if acc_val > mejor_acc:  # guarda el mejor según validación
            mejor_acc = acc_val
            torch.save(modelo.state_dict(), RUTA_MODELO); marca = "  ← mejor"
        print(f"Época {epoca:2d} | train acc {aciertos / n:.3f} | val acc {acc_val:.3f} "
              f"| val loss {perd_val:.3f} | {time.time() - t0:.0f}s{marca}")

    # Curvas de aprendizaje
    h = pd.DataFrame(historial)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(h.epoca, h.perdida_train, "o-", label="train"); axes[0].plot(h.epoca, h.perdida_val, "o-", label="validación")
    axes[0].set_title("Pérdida (cross-entropy)"); axes[0].set_xlabel("Época"); axes[0].legend()
    axes[1].plot(h.epoca, h.acc_train, "o-", label="train"); axes[1].plot(h.epoca, h.acc_val, "o-", label="validación")
    axes[1].set_title("Exactitud"); axes[1].set_xlabel("Época"); axes[1].legend()
    fig.tight_layout(); fig.savefig(SALIDAS / "04_curvas_aprendizaje.png", dpi=120); plt.close(fig)

    # Evaluación final en TEST con el mejor modelo
    modelo.load_state_dict(torch.load(RUTA_MODELO, map_location=disp))
    _, acc_test, y_true, y_pred = evaluar(modelo, cargas["test"], disp)
    nombres = [NOMBRES_ES[c] for c in CLASES]
    reporte = classification_report(y_true, y_pred, target_names=nombres, digits=4, output_dict=True)
    print(f"\n=== TEST (imágenes nunca vistas) — exactitud: {acc_test:.4f} ===")
    print(classification_report(y_true, y_pred, target_names=nombres, digits=4))

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=nombres, cmap="Greens",
                                            ax=ax, colorbar=False)
    ax.set_title("Matriz de confusión — CNN (test)")
    fig.tight_layout(); fig.savefig(SALIDAS / "05_matriz_confusion_cnn.png", dpi=120); plt.close(fig)

    # Predicciones por imagen (se reutilizan para comparar con el LLM)
    rutas = [p for p, _ in ds["test"].samples]
    pd.DataFrame({"ruta": [str(Path(r).relative_to(RAIZ)) for r in rutas], "real": [CLASES[i] for i in y_true],
                  "pred_cnn": [CLASES[i] for i in y_pred]}).to_csv(SALIDAS / "predicciones_cnn_test.csv", index=False)
    (SALIDAS / "metricas_cnn.json").write_text(json.dumps(
        {"exactitud_test": acc_test, "mejor_acc_validacion": mejor_acc, "reporte": reporte,
         "historial": historial}, indent=2, ensure_ascii=False))
    print(f"Modelo guardado en {RUTA_MODELO}")


if __name__ == "__main__":
    main()
