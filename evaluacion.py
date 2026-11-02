"""Evaluacion de las configuraciones (Fase 3 de la tesis).

- LOOCV: estimacion principal del desempeno.
- Validacion cruzada estratificada repetida: estabilidad.
- Regla de seleccion de la mejor configuracion.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, fbeta_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import LeaveOneOut, RepeatedStratifiedKFold

import config
from modelos import ajustar, puntaje


def calcular_metricas(y, pred, score):
    """Matriz de confusion y metricas de la Tabla de metricas de la tesis.

    La exactitud se incluye solo para mostrar la 'trampa de la exactitud'.
    """
    vn, fp, fn, vp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "VP": int(vp), "FN": int(fn), "FP": int(fp), "VN": int(vn),
        "Exactitud": (vp + vn) / len(y),
        "Recall": recall_score(y, pred, zero_division=0),
        "Precision": precision_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "F2": fbeta_score(y, pred, beta=2, zero_division=0),
        "EB": balanced_accuracy_score(y, pred),
        "PR_AUC": average_precision_score(y, score),
        "ROC_AUC": roc_auc_score(y, score),
    }


def evaluar_loocv(X, y, modelo, estrategia, semilla=config.SEMILLA, reducido=False):
    """Validacion cruzada dejando uno fuera.

    En cada una de las 56 iteraciones se entrena con 55 localidades y se
    predice la restante. Devuelve las 56 predicciones fuera de muestra,
    sus puntajes y cuantas veces ADASYN necesito respaldo.
    """
    pred = np.zeros(len(y), dtype=int)
    score = np.zeros(len(y))
    respaldos = []
    for tr, te in LeaveOneOut().split(X):
        pipe = ajustar(modelo, estrategia, X[tr], y[tr], semilla, respaldos, reducido)
        pred[te] = pipe.predict(X[te])
        score[te] = puntaje(pipe, modelo, X[te])
    return pred, score, len(respaldos)


def evaluar_todas(X, y, reducido=False, modelos=config.MODELOS):
    """Evalua las 25 configuraciones con LOOCV.

    Devuelve la tabla de resultados y un diccionario con los puntajes de
    cada configuracion (se usan despues para el mapa).
    """
    filas, puntajes = [], {}
    for modelo in modelos:
        for estrategia in config.ESTRATEGIAS:
            pred, score, n_resp = evaluar_loocv(X, y, modelo, estrategia, reducido=reducido)
            fila = {"Modelo": modelo, "Estrategia": estrategia, "Respaldos": n_resp}
            fila.update(calcular_metricas(y, pred, score))
            filas.append(fila)
            puntajes[(modelo, estrategia)] = (pred, score)
    return pd.DataFrame(filas), puntajes


CRITERIOS = ["Recall", "F2", "Precision"]


def seleccionar(tabla, eb_minima=config.EB_MINIMA):
    """Regla de seleccion: EB >= 0.70, luego mayor Recall; empates por F2 y Precision.

    Si dos configuraciones empatan en los tres criterios, se reporta la de
    mayor PR-AUC (asi se documenta en la seccion 16.2 del informe).
    """
    candidatos = tabla[tabla["EB"] >= eb_minima]
    if candidatos.empty:
        raise ValueError("Ninguna configuracion alcanza la exactitud balanceada minima.")
    orden = candidatos.sort_values(CRITERIOS + ["PR_AUC"], ascending=False)
    return orden.iloc[0]


def empatadas(tabla, elegida, eb_minima=config.EB_MINIMA):
    """Configuraciones que empatan con la elegida en los tres criterios de la regla."""
    candidatos = tabla[tabla["EB"] >= eb_minima]
    iguales = (candidatos[CRITERIOS].round(6) == elegida[CRITERIOS].astype(float).round(6)).all(axis=1)
    return candidatos.loc[iguales, ["Modelo", "Estrategia", "PR_AUC"]]


def evaluar_estabilidad(X, y, modelo, estrategia,
                        repeticiones=config.REPETICIONES_ESTABILIDAD,
                        pliegues=config.PLIEGUES_ESTABILIDAD):
    """Validacion cruzada estratificada repetida (3 pliegues x 30).

    En cada repeticion se juntan las predicciones de los 3 pliegues y se
    calculan las metricas. Devuelve media y DE de Recall, F2 y EB, y la
    frecuencia con que se detecto cada localidad de interes.
    """
    rskf = RepeatedStratifiedKFold(n_splits=pliegues, n_repeats=repeticiones,
                                   random_state=config.SEMILLA_ESTABILIDAD)
    pred = np.zeros(len(y), dtype=int)
    score = np.zeros(len(y))
    detecciones = np.zeros(len(y))
    filas, respaldos = [], []
    for i, (tr, te) in enumerate(rskf.split(X, y)):
        pipe = ajustar(modelo, estrategia, X[tr], y[tr], i, respaldos)
        pred[te] = pipe.predict(X[te])
        score[te] = puntaje(pipe, modelo, X[te])
        if (i + 1) % pliegues == 0:  # se completo una repeticion
            filas.append(calcular_metricas(y, pred, score))
            detecciones += ((pred == 1) & (y == 1))
    metricas = pd.DataFrame(filas)[["Recall", "F2", "EB"]]
    resumen = {
        "Modelo": modelo, "Estrategia": estrategia,
        "Recall_media": metricas["Recall"].mean(), "Recall_DE": metricas["Recall"].std(),
        "F2_media": metricas["F2"].mean(), "F2_DE": metricas["F2"].std(),
        "EB_media": metricas["EB"].mean(), "EB_DE": metricas["EB"].std(),
        "Respaldos": len(respaldos),
    }
    frecuencia = detecciones[y == 1] / repeticiones
    return resumen, frecuencia


def estabilidad_todas(X, y, nombres_positivos, modelos=config.MODELOS):
    """Aplica evaluar_estabilidad a las 25 configuraciones."""
    resumenes, frecuencias = [], []
    for modelo in modelos:
        for estrategia in config.ESTRATEGIAS:
            resumen, frec = evaluar_estabilidad(X, y, modelo, estrategia)
            resumenes.append(resumen)
            fila = {"Modelo": modelo, "Estrategia": estrategia}
            fila.update(dict(zip(nombres_positivos, frec)))
            frecuencias.append(fila)
    return pd.DataFrame(resumenes), pd.DataFrame(frecuencias)
