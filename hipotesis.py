"""Contraste de las hipotesis H1 y H2 con los criterios de la tesis.

La estabilidad se reporta de forma descriptiva (resumir_estabilidad).
"""
import config

ENSEMBLES = ["RF", "XGB"]
UMBRAL_H1 = 0.15  # 15 puntos porcentuales = una localidad adicional de seis


def contrastar_h1(tabla_loocv):
    """H1: alguna estrategia sube el Recall de RF y de XGBoost >= 15 pp."""
    detalle = {}
    for modelo in ENSEMBLES:
        filas = tabla_loocv[tabla_loocv["Modelo"] == modelo].set_index("Estrategia")
        base = filas.loc["ninguna", "Recall"]
        mejor = filas.drop(index="ninguna")["Recall"].max()
        detalle[modelo] = {"Recall_sin_tratamiento": base,
                           "Mejor_Recall_con_tratamiento": mejor,
                           "Aumento_pp": round(100 * (mejor - base), 1)}
    cumple = all(d["Aumento_pp"] >= 100 * UMBRAL_H1 for d in detalle.values())
    return cumple, detalle


def contrastar_h2(coincidencia):
    """H2: al menos 2 de las 3 variables principales coinciden."""
    return coincidencia["n_comunes"] >= 2, coincidencia


def resumir_estabilidad(tabla_estabilidad):
    """Analisis descriptivo (no es hipotesis): para cada modelo, la estrategia
    con menor y con mayor desviacion estandar del Recall."""
    resumen = {}
    for modelo, filas in tabla_estabilidad.groupby("Modelo"):
        filas = filas.set_index("Estrategia")["Recall_DE"]
        resumen[modelo] = {"mas_estable": filas.idxmin(), "DE_min": round(float(filas.min()), 3),
                           "menos_estable": filas.idxmax(), "DE_max": round(float(filas.max()), 3)}
    return resumen
