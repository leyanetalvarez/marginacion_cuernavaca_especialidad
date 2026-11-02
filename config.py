"""Configuracion general del estudio.

Concentra en un solo lugar las rutas, los nombres de las variables y los
parametros fijos, para que el resto de los modulos no tenga valores
"escondidos" y cualquier cambio se haga aqui.
"""
from pathlib import Path

# --- Rutas -----------------------------------------------------------------
CARPETA_BASE = Path(__file__).resolve().parent
RUTA_DATOS = CARPETA_BASE / "datos.xlsx"          # Excel de trabajo
CARPETA_RESULTADOS = CARPETA_BASE / "resultados"  # aqui se guardan las salidas

# --- Estructura esperada del Excel -------------------------------------------
# Columnas del Indice de Marginacion por Localidad 2020 (CONAPO).
COLUMNAS_ESPERADAS = [
    "CVE_LOC", "ENT", "NOM_ENT", "MUN", "NOM_MUN", "LOC", "NOM_LOC",
    "POB_TOT", "ANALF", "SBASC", "OVSDE", "OVSEE", "OVSAE", "OVPT",
    "OVHAC", "OVSREF", "IM_2020", "GM_2020", "IMN_2020",
]
# Columnas de clave: se leen como texto para conservar los ceros iniciales.
COLUMNAS_TEXTO = ["CVE_LOC", "ENT", "MUN", "LOC"]

CLAVE_ENTIDAD = "17"     # Morelos
CLAVE_MUNICIPIO = "007"  # Cuernavaca
N_LOCALIDADES_ESPERADO = 56

# --- Variables del modelo --------------------------------------------------
# Ocho indicadores del indice por localidad (predictores).
VARIABLES = ["ANALF", "SBASC", "OVSDE", "OVSEE", "OVSAE", "OVPT", "OVHAC", "OVSREF"]
# Columnas que NO deben usarse como predictores (fuga de informacion).
COLUMNAS_EXCLUIDAS = ["IM_2020", "IMN_2020", "GM_2020", "POB_TOT"]
COLUMNA_GRADO = "GM_2020"
GRADOS_POSITIVOS = ["Medio", "Alto"]  # clase de interes (y = 1)
GRADOS_VALIDOS = ["Muy bajo", "Bajo", "Medio", "Alto", "Muy alto"]

# --- Diseno experimental ---------------------------------------------------
MODELOS = ["RL", "AD", "RF", "XGB", "SVM"]
NOMBRES_MODELOS = {
    "RL": "Regresión Logística", "AD": "Árbol de Decisión",
    "RF": "Random Forest", "XGB": "XGBoost", "SVM": "SVM",
}
ESTRATEGIAS = ["ninguna", "pesos", "ROS", "SMOTE", "ADASYN"]
K_VECINOS = 3        # SMOTE/ADASYN: debe ser menor que los positivos en entrenamiento (4)
SEMILLA = 0          # semilla para la evaluacion LOOCV

# --- Estabilidad -------------------------------------------------------------
PLIEGUES_ESTABILIDAD = 3
REPETICIONES_ESTABILIDAD = 30
SEMILLA_ESTABILIDAD = 2026

# --- Regla de seleccion ----------------------------------------------------
EB_MINIMA = 0.70     # exactitud balanceada minima

# --- Interpretabilidad -----------------------------------------------------
N_VARIABLES_REDUCIDAS = 3
