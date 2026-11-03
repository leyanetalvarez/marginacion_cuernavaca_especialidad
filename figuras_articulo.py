"""Figuras del articulo a partir de los resultados del estudio.

Requiere haber ejecutado antes ejecutar.py (y mapa.py para el mapa).
Uso:
    python figuras_articulo.py
Genera en resultados/:
    image_3.png  matrices de confusion (Baseline RF y RL + ROS)
    image_4.png  valores SHAP medios absolutos (RF con ponderacion)
    image_5.png  copia del mapa de probabilidad (si existe)
"""
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

plt.rcParams.update({"font.family": "serif", "font.size": 9})
R = config.CARPETA_RESULTADOS


def matrices_confusion():
    """Matrices de confusion agregadas de las 56 predicciones LOOCV."""
    t = pd.read_csv(R / "16_2_resultados_loocv.csv")
    casos = [("RF", "ninguna", "Baseline (Random Forest)"),
             ("RL", "ROS", "Regresión Logística + ROS")]
    fig, axs = plt.subplots(1, 2, figsize=(5.6, 2.7))
    for ax, (modelo, estrategia, titulo) in zip(axs, casos):
        r = t[(t.Modelo == modelo) & (t.Estrategia == estrategia)].iloc[0]
        m = np.array([[r.VN, r.FP], [r.FN, r.VP]])
        ax.imshow(m, cmap="Blues", vmin=0, vmax=60)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(m[i, j]), ha="center", va="center", fontsize=12,
                        color="white" if m[i, j] > 30 else "black")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Pred. 0", "Pred. 1"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Real 0", "Real 1"])
        ax.set_title(f"{titulo}\nExactitud = {r.Exactitud:.3f}; Recall = {r.Recall:.2f}",
                     fontsize=8.5)
    fig.tight_layout()
    fig.savefig(R / "image_3.png", dpi=300)
    plt.close(fig)


def grafica_shap():
    """Barras con el valor SHAP medio absoluto; resalta las tres principales."""
    sh = pd.read_csv(R / "16_3_shap.csv").sort_values("SHAP_medio_abs")
    principales = set(sh.Variable.tail(3))
    colores = ["#1f4e79" if v in principales else "#9fb8d6" for v in sh.Variable]
    fig, ax = plt.subplots(figsize=(4.4, 2.5))
    ax.barh(sh.Variable, sh.SHAP_medio_abs, color=colores)
    for y, v in enumerate(sh.SHAP_medio_abs):
        ax.text(v + 0.002, y, f"{round(v + 1e-9, 3):.3f}", va="center", fontsize=7)
    ax.set_xlabel("Valor SHAP medio absoluto", fontsize=8.5)
    ax.tick_params(labelsize=8)
    ax.grid(axis="x", color="#e5e5e5")
    ax.set_axisbelow(True)
    for lado in ["top", "right"]:
        ax.spines[lado].set_visible(False)
    fig.tight_layout()
    fig.savefig(R / "image_4.png", dpi=300)
    plt.close(fig)


def copiar_mapa():
    """Copia el mapa generado por mapa.py con el nombre que usa el articulo."""
    origen = R / "mapa_probabilidad.png"
    if origen.exists():
        shutil.copy(origen, R / "image_5.png")
        return True
    return False


if __name__ == "__main__":
    matrices_confusion()
    grafica_shap()
    mapa = copiar_mapa()
    print(f"Figuras guardadas en: {R}")
    if not mapa:
        print("Aviso: no se encontró mapa_probabilidad.png; ejecuta antes mapa.py.")