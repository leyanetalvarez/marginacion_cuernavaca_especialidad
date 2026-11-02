"""Script principal: ejecuta el estudio completo y guarda los resultados.

Uso:
    python ejecutar.py                     # todo (unos 20 minutos)
    python ejecutar.py --sin-estabilidad   # omite la validacion repetida
    python ejecutar.py --datos otra/ruta/datos.xlsx
    python ejecutar.py --iter iter_17_cpv2020_csv.zip   # ademas genera el mapa
"""
import argparse
import json
import warnings

import pandas as pd

import config
import datos
import evaluacion
import exploracion
import guia
import hipotesis
import interpretabilidad

warnings.filterwarnings("ignore")  # avisos internos de las bibliotecas


def guardar(tabla, nombre):
    ruta = config.CARPETA_RESULTADOS / nombre
    tabla.to_csv(ruta, index=isinstance(tabla.index, pd.Index) and tabla.index.name is not None,
                 encoding="utf-8-sig")  # utf-8-sig: Excel muestra bien los acentos
    print(f"  -> {ruta.name}")


def mejor_ensemble(tabla):
    """Mejor configuracion de RF o XGBoost (para calcular SHAP)."""
    ens = tabla[tabla["Modelo"].isin(hipotesis.ENSEMBLES)]
    try:
        return evaluacion.seleccionar(ens)
    except ValueError:  # ninguna cumple la EB minima: se toma la de mayor Recall
        return ens.sort_values(["Recall", "F2", "Precision"], ascending=False).iloc[0]


def linea_base(tabla, trivial):
    """Seccion 16.1: clasificador trivial y los cinco modelos sin tratamiento."""
    sin_trat = tabla[tabla["Estrategia"] == "ninguna"].copy()
    sin_trat["Modelo"] = sin_trat["Modelo"].map(config.NOMBRES_MODELOS)
    fila_trivial = pd.DataFrame([{"Modelo": "Clasificador trivial", "VP": 0, "FN": 6, "FP": 0,
                                  "Exactitud": trivial["Exactitud"], "Recall": 0.0,
                                  "Precision": None, "EB": 0.5}])
    columnas = ["Modelo", "VP", "FN", "FP", "Exactitud", "Recall", "Precision", "EB"]
    return pd.concat([fila_trivial, sin_trat[columnas]], ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datos", default=config.RUTA_DATOS)
    parser.add_argument("--sin-estabilidad", action="store_true")
    parser.add_argument("--iter", default=None, help="ZIP del ITER 2020 para generar el mapa")
    args = parser.parse_args()
    config.CARPETA_RESULTADOS.mkdir(exist_ok=True)

    # Fase 1: datos y exploracion --------------------------------------------
    print("1. Preparando datos...")
    df, X, y = datos.preparar(args.datos)
    desbalance = exploracion.resumen_desbalance(y)
    print(f"   {desbalance}")
    guardar(exploracion.distribucion_grados(df), "16_1_distribucion_grados.csv")
    guardar(exploracion.localidades_interes(df), "16_1_localidades_interes.csv")
    guardar(exploracion.indicadores_por_clase(df), "16_1_indicadores_por_clase.csv")
    trivial = exploracion.clasificador_trivial(y)

    # Fase 3: evaluacion con LOOCV -------------------------------------------
    print("2. Evaluando 25 configuraciones con LOOCV...")
    tabla, puntajes = evaluacion.evaluar_todas(X, y)
    guardar(tabla.round(4), "16_2_resultados_loocv.csv")
    guardar(linea_base(tabla, trivial).round(4), "16_1_linea_base.csv")
    elegida = evaluacion.seleccionar(tabla)
    empates = evaluacion.empatadas(tabla, elegida)
    print(f"   Seleccionada: {elegida['Modelo']} + {elegida['Estrategia']} "
          f"(Recall={elegida['Recall']:.3f}, EB={elegida['EB']:.3f})")
    if len(empates) > 1:
        print(f"   Empate en la regla con: {list(empates['Estrategia'])} (se usa PR-AUC)")

    # Puntajes fuera de muestra de la configuracion elegida (mapa y guia)
    pred, score = puntajes[(elegida["Modelo"], elegida["Estrategia"])]
    mapa = df[["CVE_LOC", "NOM_LOC", "POB_TOT", config.COLUMNA_GRADO, "y"]].copy()
    mapa["prediccion"], mapa["puntaje_riesgo"] = pred, score
    guardar(mapa.sort_values("puntaje_riesgo", ascending=False), "16_6_puntajes_para_mapa.csv")
    tabla_guia, referencia = guia.tabla_guia(df, mapa)
    guardar(tabla_guia, "16_6_guia_localidades.csv")

    # Fase 3: estabilidad ------------------------------------------------------
    tabla_est = None
    if not args.sin_estabilidad:
        print("3. Midiendo estabilidad (3 pliegues x 30 repeticiones)...")
        nombres = list(df.loc[df["y"] == 1, "NOM_LOC"])
        tabla_est, frec = evaluacion.estabilidad_todas(X, y, nombres)
        guardar(tabla_est.round(4), "16_2_estabilidad.csv")
        guardar(frec.round(4), "16_2_frecuencia_deteccion.csv")

    # Fase 4: interpretabilidad -----------------------------------------------
    print("4. Interpretabilidad...")
    coef = interpretabilidad.coeficientes_rl(X, y)
    guardar(coef.round(4), "16_3_coeficientes_rl.csv")
    ens = mejor_ensemble(tabla)
    shap_tabla = interpretabilidad.shap_ensemble(X, y, ens["Modelo"], ens["Estrategia"])
    guardar(shap_tabla.round(4), "16_3_shap.csv")
    coincid = interpretabilidad.coincidencia_top(coef, shap_tabla)

    print("   Subconjunto reducido (3 variables elegidas dentro de cada pliegue)...")
    pred_r, score_r, _ = evaluacion.evaluar_loocv(X, y, elegida["Modelo"],
                                                  elegida["Estrategia"], reducido=True)
    reducido = evaluacion.calcular_metricas(y, pred_r, score_r)

    # Fase 4: hipotesis ---------------------------------------------------------
    h1, h1_det = hipotesis.contrastar_h1(tabla)
    h2, h2_det = hipotesis.contrastar_h2(coincid)
    resumen = {
        "desbalance": desbalance,
        "clasificador_trivial": trivial,
        "configuracion_seleccionada": elegida.to_dict(),
        "configuraciones_empatadas": empates.round(4).to_dict(orient="records"),
        "configuraciones_sobre_EB_minima": int((tabla["EB"] >= config.EB_MINIMA).sum()),
        "ensemble_para_shap": f"{ens['Modelo']} + {ens['Estrategia']}",
        "subconjunto_reducido": {k: round(float(v), 4) for k, v in reducido.items()},
        "H1": {"cumple": h1, "detalle": h1_det},
        "H2": {"cumple": h2, "detalle": h2_det},
        "medianas_municipales": referencia.to_dict(),
    }
    if tabla_est is not None:
        resumen["estabilidad_descriptiva"] = hipotesis.resumir_estabilidad(tabla_est)

    ruta = config.CARPETA_RESULTADOS / "resumen.json"
    ruta.write_text(json.dumps(resumen, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8")
    print(f"  -> {ruta.name}")

    # Fase 5: mapa (opcional) -------------------------------------------------
    if args.iter:
        import mapa as modulo_mapa  # se importa aqui porque requiere matplotlib
        print("5. Generando mapa...")
        print(f"  -> {modulo_mapa.dibujar(modulo_mapa.unir_coordenadas(args.iter)).name}")

    print(f"Listo. Resultados en: {config.CARPETA_RESULTADOS}")


if __name__ == "__main__":
    main()