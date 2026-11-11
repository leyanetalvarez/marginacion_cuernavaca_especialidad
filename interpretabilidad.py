"""Interpretabilidad de los modelos (Fase 4 de la tesis).

- Coeficientes y razones de momios de la Regresion Logistica con ROS.
- Valores SHAP del mejor modelo de ensemble.
- Comparacion de las tres variables principales (hipotesis H2).

Los modelos de esta seccion se ajustan con las 56 localidades porque solo
sirven para interpretar, no para medir el desempeno.
"""
import numpy as np
import pandas as pd
import shap

import config
from modelos import construir_pipeline


def coeficientes_rl(X, y, semilla=config.SEMILLA):
    """Coeficientes estandarizados y razon de momios por cada DE del indicador."""
    n_neg, n_pos = int((y == 0).sum()), int((y == 1).sum())
    pipe = construir_pipeline("RL", "ROS", semilla, n_neg, n_pos)
    pipe.fit(X, y)
    rl = pipe.named_steps["clf"]
    coef = rl.coef_[0]
    tabla = pd.DataFrame({
        "Variable": config.VARIABLES,
        "Coeficiente": coef,
        "Razon_momios": np.exp(coef),
        "Abs": np.abs(coef),
    })
    return tabla.sort_values("Abs", ascending=False).drop(columns="Abs").reset_index(drop=True)


def shap_ensemble(X, y, modelo, estrategia):
    """Valor SHAP medio absoluto de cada variable para un modelo de arboles.

    El modelo se ajusta con el mismo pipeline de la evaluacion; los valores
    SHAP se calculan sobre las 56 localidades originales (no sinteticas).
    """
    n_neg, n_pos = int((y == 0).sum()), int((y == 1).sum())
    pipe = construir_pipeline(modelo, estrategia, config.SEMILLA, n_neg, n_pos)
    pipe.fit(X, y)
    clf = pipe.named_steps["clf"]
    valores = shap.TreeExplainer(clf).shap_values(X)
    valores = np.asarray(valores)
    if valores.ndim == 3:            # Random Forest: (muestras, variables, clases)
        valores = valores[:, :, 1]
    tabla = pd.DataFrame({
        "Variable": config.VARIABLES,
        "SHAP_medio_abs": np.abs(valores).mean(axis=0),
    })
    return tabla.sort_values("SHAP_medio_abs", ascending=False).reset_index(drop=True)


def coincidencia_top(coef, shap_tabla, k=config.N_VARIABLES_REDUCIDAS):
    """Variables en comun entre los top-k de coeficientes y de SHAP (H2)."""
    top_coef = list(coef["Variable"].head(k))
    top_shap = list(shap_tabla["Variable"].head(k))
    comunes = sorted(set(top_coef) & set(top_shap))
    return {"top_coeficientes": top_coef, "top_shap": top_shap,
            "comunes": comunes, "n_comunes": len(comunes)}
