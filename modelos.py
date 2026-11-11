"""Construccion de los modelos y de las estrategias de balanceo.

Cada configuracion se arma como un pipeline de imbalanced-learn:

    [estandarizacion] -> [seleccion de variables] -> [sobremuestreo] -> clasificador

El pipeline se ajusta solo con datos de entrenamiento, por lo que la
estandarizacion y el sobremuestreo nunca "ven" la localidad que se evalua.
"""
import numpy as np
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

import config


class SeleccionPorCoeficientes(BaseEstimator, TransformerMixin):
    """Conserva las k variables con mayor |coeficiente| de una Regresion
    Logistica ajustada con ROS sobre los datos de entrenamiento.

    Se usa en el analisis del subconjunto reducido. Como forma parte del
    pipeline, la eleccion de variables se repite en cada pliegue y no usa
    la localidad de prueba.
    """

    def __init__(self, k=config.N_VARIABLES_REDUCIDAS, random_state=config.SEMILLA):
        self.k = k
        self.random_state = random_state

    def fit(self, X, y):
        X_res, y_res = RandomOverSampler(
            random_state=self.random_state
        ).fit_resample(X, y)
        rl = LogisticRegression(max_iter=1000).fit(X_res, y_res)
        self.indices_ = np.argsort(-np.abs(rl.coef_[0]))[: self.k]
        return self

    def transform(self, X):
        return X[:, self.indices_]


def crear_clasificador(modelo, estrategia, semilla, n_neg, n_pos):
    """Devuelve el clasificador con su configuracion fija (Tabla de hiperparametros).

    Si la estrategia es 'pesos', se activa la ponderacion de clases:
    class_weight='balanced' en scikit-learn y scale_pos_weight en XGBoost.
    """
    pesos = "balanced" if estrategia == "pesos" else None
    if modelo == "RL":
        return LogisticRegression(class_weight=pesos, max_iter=1000)
    if modelo == "AD":
        return DecisionTreeClassifier(class_weight=pesos, random_state=semilla)
    if modelo == "RF":
        return RandomForestClassifier(n_estimators=500, class_weight=pesos,
                                      random_state=semilla, n_jobs=-1)
    if modelo == "XGB":
        spw = n_neg / n_pos if estrategia == "pesos" else 1.0
        return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.1,
                             scale_pos_weight=spw, eval_metric="logloss",
                             random_state=semilla, n_jobs=1)
    if modelo == "SVM":
        return SVC(kernel="rbf", C=1.0, gamma="scale", class_weight=pesos)
    raise ValueError(f"Modelo desconocido: {modelo}")


def crear_muestreador(estrategia, semilla):
    """Devuelve el metodo de sobremuestreo o None si la estrategia no lo usa."""
    if estrategia == "ROS":
        return RandomOverSampler(random_state=semilla)
    if estrategia == "SMOTE":
        return SMOTE(k_neighbors=config.K_VECINOS, random_state=semilla)
    if estrategia == "ADASYN":
        return ADASYN(n_neighbors=config.K_VECINOS, random_state=semilla)
    return None


def construir_pipeline(modelo, estrategia, semilla, n_neg, n_pos, reducido=False):
    """Arma el pipeline completo de una configuracion modelo + estrategia."""
    pasos = []
    if modelo in ("RL", "SVM"):  # sensibles a la escala
        pasos.append(("escala", StandardScaler()))
    if reducido:
        pasos.append(("seleccion", SeleccionPorCoeficientes(random_state=semilla)))
    muestreador = crear_muestreador(estrategia, semilla)
    if muestreador is not None:
        pasos.append(("muestreo", muestreador))
    pasos.append(("clf", crear_clasificador(modelo, estrategia, semilla, n_neg, n_pos)))
    return Pipeline(pasos)


def ajustar(modelo, estrategia, X_tr, y_tr, semilla, respaldos, reducido=False):
    """Ajusta el pipeline con los datos de entrenamiento.

    Si ADASYN no puede generar muestras (ningun positivo tiene vecinos de la
    clase mayoritaria), ese pliegue se entrena sin sobremuestreo y se anota
    en la lista 'respaldos', como indica la metodologia.
    """
    n_neg, n_pos = int((y_tr == 0).sum()), int((y_tr == 1).sum())
    pipe = construir_pipeline(modelo, estrategia, semilla, n_neg, n_pos, reducido)
    try:
        pipe.fit(X_tr, y_tr)
    except (RuntimeError, ValueError) as error:
        respaldos.append(str(error))
        pipe = construir_pipeline(modelo, "ninguna", semilla, n_neg, n_pos, reducido)
        pipe.fit(X_tr, y_tr)
    return pipe


def puntaje(pipe, modelo, X):
    """Puntaje de riesgo: probabilidad de y = 1 (funcion de decision en SVM)."""
    if modelo == "SVM":
        return pipe.decision_function(X)
    return pipe.predict_proba(X)[:, 1]
