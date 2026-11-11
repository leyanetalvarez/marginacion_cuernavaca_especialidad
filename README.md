# Técnicas para datos desbalanceados aplicadas al índice de marginación de Cuernavaca

Código del proyecto de la Especialidad en *Data Science for Experimental Research*:
**"Evaluación comparativa de técnicas de aprendizaje y estrategias para datos desbalanceados aplicadas a los índices de marginación del municipio de Cuernavaca"**.

El estudio compara **5 clasificadores** y **5 condiciones de tratamiento del desbalance** (25 configuraciones) para identificar las localidades de Cuernavaca, Morelos, con grado de marginación **medio o alto**. Al final genera un mapa de probabilidad y una tabla de localidades prioritarias para la planeación municipal.

---

## Resumen del estudio

| Aspecto | Detalle |
|---|---|
| Datos | Índice de Marginación por Localidad 2020 (CONAPO), 56 localidades de Cuernavaca |
| Clase de interés | Grado **Medio o Alto** (6 localidades) frente a Bajo o Muy bajo (50) |
| Razón de desbalance | 50 / 6 ≈ **8.3 : 1** (desbalance medio) |
| Predictores | 8 indicadores: `ANALF`, `SBASC`, `OVSDE`, `OVSEE`, `OVSAE`, `OVPT`, `OVHAC`, `OVSREF` |
| Modelos | Regresión Logística, Árbol de Decisión, Random Forest, XGBoost, SVM |
| Estrategias | Sin tratamiento, ponderación de clases, ROS, SMOTE, ADASYN |
| Validación | Validación cruzada dejando uno fuera (LOOCV), sin fuga de información |
| Estabilidad | Validación cruzada estratificada de 3 pliegues, repetida 30 veces |
| Interpretabilidad | Coeficientes de Regresión Logística con ROS y valores SHAP del ensemble |

**Resultado principal:** la Regresión Logística con ROS detectó las 6 localidades de interés con un solo falso positivo (Recall = 1.00, exactitud balanceada = 0.99).

---

## Estructura del repositorio

```
.
├── config.py              # Rutas, variables, semillas y parámetros del estudio
├── datos.py               # Carga y validación del Excel; variable objetivo
├── exploracion.py         # Distribución por grado, desbalance, clasificador trivial
├── modelos.py             # Clasificadores, estrategias de balanceo y pipelines
├── evaluacion.py          # Métricas, LOOCV, regla de selección y estabilidad
├── interpretabilidad.py   # Coeficientes de Regresión Logística y SHAP
├── hipotesis.py           # Contraste de hipótesis H1 y H2
├── guia.py                # Localidades prioritarias y rubro sugerido
├── mapa.py                # Mapa de probabilidad (requiere el ITER 2020)
├── ejecutar.py            # Script principal: corre todo el estudio
├── datos.xlsx             # Datos de trabajo (56 localidades)
├── requirements.txt       # Bibliotecas necesarias
└── resultados/            # Se crea al ejecutar: tablas, figuras y resumen
```

---

## Requisitos

- **Python 3.10 o superior**
- Bibliotecas: `pandas`, `openpyxl`, `numpy`, `scikit-learn`, `imbalanced-learn`, `xgboost`, `shap`, `matplotlib`, `adjustText`

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/leyanetalvarez/marginacion_cuernavaca_especialidad.git
cd REPOSITORIO

# 2. (Opcional, recomendado) crear un entorno virtual
python3 -m venv .venv

# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3. Instalar las dependencias
pip install -r requirements.txt
```

---

## Uso

### Ejecutar el estudio completo

```bash
# Windows:
python ejecutar.py
# Linux / macOS:
python3 ejecutar.py
```

Tarda alrededor de **20 minutos**, sobre todo por la evaluación de estabilidad (90 particiones × 25 configuraciones).

### Opciones

| Comando | Qué hace |
|---|---|
| `python ejecutar.py` / `python3 ejecutar.py` | Estudio completo |
| `python ejecutar.py --sin-estabilidad` / `python3 ejecutar.py --sin-estabilidad` | Omite la validación repetida (≈ 8 minutos) |
| `python ejecutar.py --datos ruta/otro.xlsx` / `python3 ejecutar.py --datos ruta/otro.xlsx` | Usa otro archivo de datos con la misma estructura |
| `python ejecutar.py --iter iter_17_cpv2020_csv.zip` / `python3 ejecutar.py --iter iter_17_cpv2020_csv.zip` | Además genera el mapa |

### Generar el mapa

El mapa necesita las coordenadas de las localidades, que vienen en los **Principales Resultados por Localidad (ITER) 2020** del INEGI:

1. Descarga el archivo de Morelos en formato CSV desde el sitio del INEGI (*Censo de Población y Vivienda 2020 → Datos abiertos → ITER → Morelos*). El archivo se llama `iter_17_cpv2020_csv.zip`.
2. Colócalo en la carpeta del proyecto, **sin descomprimir**.
3. Ejecuta `python ejecutar.py --iter iter_17_cpv2020_csv.zip` (Windows) o `python3 ejecutar.py --iter iter_17_cpv2020_csv.zip` (Linux/macOS), o solo el mapa con `python mapa.py iter_17_cpv2020_csv.zip` / `python3 mapa.py iter_17_cpv2020_csv.zip` (requiere haber corrido antes `ejecutar.py`).

---

## Resultados

Todos los archivos se guardan en `resultados/`. Su nombre indica la sección del informe donde se usan:

| Archivo | Contenido |
|---|---|
| `16_1_distribucion_grados.csv` | Localidades por grado de marginación |
| `16_1_localidades_interes.csv` | Las 6 localidades de la clase de interés |
| `16_1_indicadores_por_clase.csv` | Media y mediana de cada indicador por clase |
| `16_1_linea_base.csv` | Clasificador trivial y modelos sin tratamiento |
| `16_2_resultados_loocv.csv` | Métricas de las 25 configuraciones |
| `16_2_estabilidad.csv` | Media y desviación estándar en 30 repeticiones |
| `16_2_frecuencia_deteccion.csv` | Frecuencia de detección de cada localidad de interés |
| `16_3_coeficientes_rl.csv` | Coeficientes y razones de momios |
| `16_3_shap.csv` | Valores SHAP medios absolutos |
| `16_6_puntajes_para_mapa.csv` | Probabilidad estimada para cada localidad |
| `16_6_guia_localidades.csv` | Localidades prioritarias, carencias y rubro sugerido |
| `mapa_probabilidad.png` | Mapa de probabilidad |
| `resumen.json` | Configuración seleccionada, empates, contraste de hipótesis y estabilidad |

Los archivos CSV se guardan en UTF-8 con BOM, para que Excel muestre bien los acentos.

---

## Datos

`datos.xlsx` contiene las 56 localidades de Cuernavaca (clave de entidad `17`, municipio `007`) extraídas de la base del **Índice de Marginación por Localidad 2020** de CONAPO, con sus 19 columnas originales. Se excluyó el registro agregado *"Localidades de hasta 2 viviendas"*.

Al cargarlo, `datos.py` comprueba automáticamente columnas, municipio, número de localidades, faltantes, claves duplicadas, grados válidos y que los porcentajes estén entre 0 y 100. Si algo no cuadra, el programa se detiene con un mensaje que explica el problema.

**Fuentes:**
- Consejo Nacional de Población (CONAPO). *Índice de marginación por localidad 2020* [Base de datos]. https://www.gob.mx/conapo/documentos/indices-de-marginacion-2020-284372
- Instituto Nacional de Estadística y Geografía (INEGI). *Censo de Población y Vivienda 2020: Principales resultados por localidad (ITER)*.

---

## Decisiones metodológicas implementadas

- **Sin fuga de información.** La estandarización y el sobremuestreo forman parte de un `Pipeline` de imbalanced-learn y se ajustan solo con los datos de entrenamiento de cada iteración.
- **LOOCV.** Con 6 casos positivos, una partición 80/20 dejaría uno o dos casos para evaluar. En LOOCV cada localidad se evalúa una vez como dato no visto.
- **`k = 3` vecinos** en SMOTE y ADASYN, porque en la validación de estabilidad solo quedan 4 positivos en entrenamiento.
- **Hiperparámetros fijos** (ver `modelos.py`). Con 6 positivos, un ajuste por validación cruzada anidada no sería confiable.
- **Regla de selección:** exactitud balanceada ≥ 0.70, luego mayor Recall; desempate por F2, Precision y PR-AUC.
- **Reproducibilidad:** las semillas se fijan en `config.py`.

Para cambiar cualquier parámetro (umbral, número de repeticiones, vecinos, etc.) edita solo `config.py`.

---

## Autora

**Ing. Leyanet Beatriz Alvarez Herrera**
Tutora: Dra. Julia Guadalupe Juárez Hernández

## Licencia

Este proyecto se distribuye con fines académicos. Si utilizas este código, cita el trabajo:

> Alvarez Herrera, L. B. (2026). *Evaluación comparativa de técnicas de aprendizaje y estrategias para datos desbalanceados aplicadas a los índices de marginación del municipio de Cuernavaca*. Especialidad en Data Science for Experimental Research.
