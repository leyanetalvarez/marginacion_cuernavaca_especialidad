"""Tabla de localidades prioritarias para la guia practica (seccion 16.6.2).

Incluye las localidades de la clase de interes y las de grado bajo que el
modelo clasifico como positivas. Para cada una identifica sus carencias
principales y el rubro de inversion sugerido.
"""
import pandas as pd

import config

N_CARENCIAS = 3

NOMBRES = {
    "ANALF": "Analfabetismo",
    "SBASC": "Sin educación básica",
    "OVSDE": "Sin drenaje",
    "OVSEE": "Sin energía eléctrica",
    "OVSAE": "Sin agua entubada",
    "OVPT": "Piso de tierra",
    "OVHAC": "Hacinamiento",
    "OVSREF": "Sin refrigerador",
}

# Rubro asociado a cada carencia. OVSREF (bienes del hogar) no tiene rubro FAIS.
RUBROS = {
    "OVSAE": "agua potable",
    "OVSDE": "drenaje",
    "OVSEE": "electrificación",
    "OVPT": "mejoramiento de vivienda",
    "OVHAC": "mejoramiento de vivienda",
    "SBASC": "coordinación con programas educativos",
    "ANALF": "coordinación con programas educativos",
}


def carencias_principales(fila, medianas, n=N_CARENCIAS):
    """Las n carencias que mas superan la mediana municipal (en puntos porcentuales)."""
    diferencia = (fila[config.VARIABLES] - medianas).astype(float)
    diferencia = diferencia[diferencia > 0].sort_values(ascending=False)
    return list(diferencia.index[:n])


def rubros_sugeridos(carencias):
    """Rubros sin repetir, en el orden de las carencias."""
    rubros = []
    for variable in carencias:
        rubro = RUBROS.get(variable)
        if rubro and rubro not in rubros:
            rubros.append(rubro)
    return rubros


def tabla_guia(df, puntajes):
    """Une datos y puntajes, y arma la tabla de localidades prioritarias.

    df: tabla completa de datos.preparar(); puntajes: DataFrame con CVE_LOC,
    prediccion y puntaje_riesgo de la configuracion seleccionada.
    """
    medianas = df[config.VARIABLES].median()
    datos = df.merge(puntajes[["CVE_LOC", "prediccion", "puntaje_riesgo"]], on="CVE_LOC")
    prioritarias = datos[(datos["y"] == 1) | (datos["prediccion"] == 1)]
    prioritarias = prioritarias.sort_values("puntaje_riesgo", ascending=False)  # sin redondear

    filas = []
    for _, fila in prioritarias.iterrows():
        carencias = carencias_principales(fila, medianas)
        filas.append({
            "CVE_LOC": fila["CVE_LOC"],
            "Localidad": fila["NOM_LOC"],
            "Habitantes": int(fila["POB_TOT"]),
            "Grado": fila[config.COLUMNA_GRADO],
            "Probabilidad": round(float(fila["puntaje_riesgo"]), 2),
            "Verificar_en_campo": fila["y"] == 0,
            "Carencias": "; ".join(f"{NOMBRES[v]} ({fila[v]:.1f} %)" for v in carencias),
            "Rubro_sugerido": ", ".join(rubros_sugeridos(carencias)),
        })
    tabla = pd.DataFrame(filas)
    referencia = medianas.round(1).rename(index=NOMBRES)
    return tabla.reset_index(drop=True), referencia
