"""Mapa de probabilidad de marginacion por localidad (Fase 5 de la tesis).

Une los puntajes fuera de muestra del modelo seleccionado con las
coordenadas del ITER 2020 (INEGI) y dibuja un mapa de puntos.

Uso:
    python mapa.py iter_17_cpv2020_csv.zip
"""
import re
import sys
import zipfile

import matplotlib
matplotlib.use("Agg")  # dibuja sin abrir ventanas
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from adjustText import adjust_text

import config

RUTA_PUNTAJES = config.CARPETA_RESULTADOS / "16_6_puntajes_para_mapa.csv"
RUTA_MAPA = config.CARPETA_RESULTADOS / "mapa_probabilidad.png"
LOCALIDADES_A_VERIFICAR = ["Tlachichilpa"]  # grado bajo con probabilidad alta


def dms_a_decimal(texto):
    """Convierte '99°18'14.699" W' en -99.3041 (grados decimales).

    Oeste (W) y Sur (S) son negativos.
    """
    grados, minutos, segundos, hemisferio = re.match(
        r"(\d+)°(\d+)'([\d.]+)\"\s*([NSEW])", texto.strip()).groups()
    valor = float(grados) + float(minutos) / 60 + float(segundos) / 3600
    return -valor if hemisferio in "SW" else valor


def leer_iter(ruta_zip):
    """Lee el CSV del ITER directamente del .zip y devuelve CVE_LOC, lat y lon."""
    with zipfile.ZipFile(ruta_zip) as z:
        nombre = next(n for n in z.namelist() if "conjunto_de_datos_iter" in n)
        with z.open(nombre) as f:
            iter_df = pd.read_csv(f, dtype=str, usecols=[
                "ENTIDAD", "MUN", "LOC", "LONGITUD", "LATITUD"])
    iter_df = iter_df.dropna(subset=["LATITUD", "LONGITUD"])  # quita filas de totales
    iter_df["CVE_LOC"] = (iter_df["ENTIDAD"].str.zfill(2) + iter_df["MUN"].str.zfill(3)
                          + iter_df["LOC"].str.zfill(4))
    iter_df["lat"] = iter_df["LATITUD"].map(dms_a_decimal)
    iter_df["lon"] = iter_df["LONGITUD"].map(dms_a_decimal)
    return iter_df[["CVE_LOC", "lat", "lon"]]


def unir_coordenadas(ruta_zip):
    """Agrega lat/lon a los puntajes y verifica que ninguna localidad quede sin punto."""
    puntajes = pd.read_csv(RUTA_PUNTAJES, dtype={"CVE_LOC": str})
    tabla = puntajes.merge(leer_iter(ruta_zip), on="CVE_LOC", how="left")
    sin_punto = tabla[tabla["lat"].isna()]
    if len(sin_punto):
        raise ValueError(f"Localidades sin coordenadas: {list(sin_punto['NOM_LOC'])}")
    return tabla


def dibujar(tabla, ruta_salida=RUTA_MAPA):
    """Mapa de puntos: color = probabilidad; forma = clase oficial."""
    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    mapa_color = plt.get_cmap("YlOrRd")
    normal = plt.Normalize(0, 1)

    # Clase 0 (circulos) y clase 1 (triangulos), coloreados por probabilidad
    for clase, marcador, tam, etiqueta in [(0, "o", 55, "Grado bajo o muy bajo"),
                                           (1, "^", 120, "Grado medio o alto")]:
        sub = tabla[tabla["y"] == clase]
        ax.scatter(sub["lon"], sub["lat"], c=sub["puntaje_riesgo"], cmap=mapa_color,
                   norm=normal, marker=marcador, s=tam, edgecolors="#333333",
                   linewidths=0.8, label=etiqueta, zorder=3)

    # Etiquetas: localidades de interes y localidades a verificar
    textos = []
    marcar = tabla[(tabla["y"] == 1) | tabla["NOM_LOC"].isin(LOCALIDADES_A_VERIFICAR)]
    for _, fila in marcar.iterrows():
        verificar = fila["NOM_LOC"] in LOCALIDADES_A_VERIFICAR
        textos.append(ax.text(fila["lon"], fila["lat"],
                              fila["NOM_LOC"] + (" *" if verificar else ""),
                              fontsize=8.5, style="italic" if verificar else "normal",
                              zorder=4))
    cab = tabla[tabla["NOM_LOC"] == "Cuernavaca"].iloc[0]
    textos.append(ax.text(cab["lon"], cab["lat"], "Cuernavaca (cabecera)",
                          fontsize=8.5, color="#555555", zorder=4))
    # Aleja las etiquetas de todos los puntos para que no se encimen
    adjust_text(textos, x=tabla["lon"].to_numpy(), y=tabla["lat"].to_numpy(), ax=ax,
                expand=(1.4, 1.8), force_text=(0.4, 0.6), force_static=(0.6, 0.9),
                arrowprops=dict(arrowstyle="-", color="#777777", lw=0.6))

    # Proporcion correcta entre grados de longitud y latitud a esta latitud
    ax.set_aspect(1 / np.cos(np.radians(tabla["lat"].mean())))
    ax.set_xlabel("Longitud (°)")
    ax.set_ylabel("Latitud (°)")
    ax.grid(color="#dddddd", lw=0.5, zorder=0)

    barra = fig.colorbar(plt.cm.ScalarMappable(norm=normal, cmap=mapa_color), ax=ax,
                         fraction=0.04, pad=0.02)
    barra.set_label("Probabilidad estimada de grado medio o alto (LOOCV)")
    barra.ax.axhline(0.5, color="black", lw=1)  # umbral de clasificacion

    # Leyenda con simbolos neutros (el color ya lo explica la barra)
    simbolos = [Line2D([], [], marker="o", ls="", ms=7, mfc="#cccccc", mec="#333333",
                       label="Grado bajo o muy bajo"),
                Line2D([], [], marker="^", ls="", ms=9, mfc="#cccccc", mec="#333333",
                       label="Grado medio o alto")]
    ax.legend(handles=simbolos, title="Grado oficial (CONAPO)", loc="lower left",
              fontsize=8.5, title_fontsize=8.5, framealpha=0.95)

    _flecha_norte(ax)
    _barra_escala(ax, tabla["lat"].mean())
    ax.text(0.99, 0.01, "* Grado bajo con probabilidad alta: verificar en campo",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, style="italic")

    fig.tight_layout()
    fig.savefig(ruta_salida, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return ruta_salida


def _flecha_norte(ax):
    ax.annotate("N", xy=(0.95, 0.95), xytext=(0.95, 0.87), xycoords="axes fraction",
                ha="center", va="center", fontsize=11, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2))


def _barra_escala(ax, latitud, km=2):
    """Barra de escala de 'km' kilometros en la esquina inferior derecha."""
    grados = km / (111.32 * np.cos(np.radians(latitud)))  # km -> grados de longitud
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    x_ini = x1 - (x1 - x0) * 0.05 - grados
    y = y0 + (y1 - y0) * 0.07
    ax.plot([x_ini, x_ini + grados], [y, y], color="black", lw=3, zorder=5)
    ax.text(x_ini + grados / 2, y + (y1 - y0) * 0.012, f"{km} km",
            ha="center", va="bottom", fontsize=8.5)


if __name__ == "__main__":
    ruta = sys.argv[1] if len(sys.argv) > 1 else "iter_17_cpv2020_csv.zip"
    tabla = unir_coordenadas(ruta)
    print(f"Localidades con coordenadas: {len(tabla)}")
    print(f"Mapa guardado en: {dibujar(tabla)}")
