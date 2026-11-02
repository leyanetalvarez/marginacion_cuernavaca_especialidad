"""Analisis exploratorio y diagnostico del desbalance (Fase 1 de la tesis).

Produce las tablas de la seccion 16.1: distribucion por grado, localidades
de la clase de interes, indicadores por clase y el clasificador trivial.
"""
import pandas as pd

import config


def distribucion_grados(df):
    """Numero y porcentaje de localidades por grado de marginacion."""
    conteo = (df[config.COLUMNA_GRADO]
              .value_counts()
              .reindex(config.GRADOS_VALIDOS, fill_value=0))
    tabla = pd.DataFrame({
        "Localidades": conteo,
        "Porcentaje": (100 * conteo / len(df)).round(1),
        "Clase": [1 if g in config.GRADOS_POSITIVOS else 0 for g in conteo.index],
    })
    tabla.index.name = "Grado"
    return tabla


def resumen_desbalance(y):
    """Tamano de cada clase y razon de desbalance (negativos / positivos)."""
    positivos = int(y.sum())
    negativos = int(len(y) - positivos)
    return {
        "n": len(y),
        "positivos": positivos,
        "negativos": negativos,
        "razon_desbalance": round(negativos / positivos, 2),
        "porcentaje_positivos": round(100 * positivos / len(y), 1),
    }


def localidades_interes(df):
    """Lista de las localidades con y = 1, ordenadas por poblacion."""
    columnas = ["CVE_LOC", "NOM_LOC", "POB_TOT", config.COLUMNA_GRADO]
    return df.loc[df["y"] == 1, columnas].sort_values("POB_TOT").reset_index(drop=True)


def indicadores_por_clase(df):
    """Media y mediana de cada indicador en cada clase."""
    agrupado = df.groupby("y")[config.VARIABLES].agg(["mean", "median"]).T
    tabla = agrupado.unstack(level=1)
    tabla.columns = [f"{estad}_clase{clase}" for clase, estad in tabla.columns]
    orden = ["mean_clase0", "median_clase0", "mean_clase1", "median_clase1"]
    tabla = tabla[orden].round(2)
    tabla.index.name = "Variable"
    return tabla


def clasificador_trivial(y):
    """Metricas de un modelo que predice siempre la clase mayoritaria.

    Ilustra la 'trampa de la exactitud': exactitud alta, Recall cero.
    """
    exactitud = (y == 0).mean()
    return {
        "Exactitud": round(float(exactitud), 3),
        "Recall": 0.0,
        "Exactitud_balanceada": 0.5,
    }
