"""Carga, validacion y preparacion de los datos.

Lee el Excel de trabajo, comprueba que tenga la estructura esperada y
construye la variable objetivo binaria.
"""
import pandas as pd

import config


def cargar_excel(ruta=config.RUTA_DATOS):
    """Lee el Excel conservando las claves como texto (p. ej., MUN = '007')."""
    tipos = {columna: str for columna in config.COLUMNAS_TEXTO}
    return pd.read_excel(ruta, dtype=tipos)


def validar(df):
    """Detiene el proceso con un mensaje claro si los datos no son los esperados.

    Revisa columnas, municipio, numero de localidades, faltantes, claves
    duplicadas, grados validos y que los porcentajes esten entre 0 y 100.
    """
    faltan = set(config.COLUMNAS_ESPERADAS) - set(df.columns)
    if faltan:
        raise ValueError(f"Faltan columnas en el Excel: {sorted(faltan)}")

    otros = df[(df["ENT"] != config.CLAVE_ENTIDAD) | (df["MUN"] != config.CLAVE_MUNICIPIO)]
    if len(otros):
        raise ValueError(f"Hay {len(otros)} registros que no son de Cuernavaca.")

    agregados = df["NOM_LOC"].str.contains("hasta 2 viviendas", case=False, na=False)
    if agregados.any():
        raise ValueError("El Excel incluye el registro agregado "
                         "'Localidades de hasta 2 viviendas'; debe excluirse.")

    if len(df) != config.N_LOCALIDADES_ESPERADO:
        raise ValueError(f"Se esperaban {config.N_LOCALIDADES_ESPERADO} localidades "
                         f"y hay {len(df)}.")

    columnas_modelo = config.VARIABLES + [config.COLUMNA_GRADO]
    if df[columnas_modelo].isna().any().any():
        raise ValueError("Hay valores faltantes en los indicadores o en el grado.")

    if df["CVE_LOC"].duplicated().any():
        raise ValueError("Hay claves de localidad (CVE_LOC) duplicadas.")

    grados_raros = set(df[config.COLUMNA_GRADO]) - set(config.GRADOS_VALIDOS)
    if grados_raros:
        raise ValueError(f"Grados de marginacion no reconocidos: {grados_raros}")

    fuera_de_rango = (df[config.VARIABLES] < 0) | (df[config.VARIABLES] > 100)
    if fuera_de_rango.any().any():
        raise ValueError("Hay porcentajes fuera del rango 0-100.")
    return True


def crear_objetivo(df):
    """Agrega la columna y: 1 si el grado es Medio o Alto, 0 en otro caso."""
    df = df.copy()
    df["y"] = df[config.COLUMNA_GRADO].isin(config.GRADOS_POSITIVOS).astype(int)
    return df


def preparar(ruta=config.RUTA_DATOS):
    """Flujo completo: cargar -> validar -> variable objetivo.

    Devuelve la tabla completa (para reportes y mapas), la matriz X de
    predictores y el vector y.
    """
    df = cargar_excel(ruta)
    validar(df)
    df = crear_objetivo(df)
    X = df[config.VARIABLES].to_numpy(dtype=float)
    y = df["y"].to_numpy()
    return df, X, y
