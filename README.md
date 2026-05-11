# Predicción de estrés hídrico por parcela a 5 y 7 días

Este repositorio es un MVP técnico para predecir el `stress_index` de parcelas agrícolas aproximadamente a 5 y 7 días vista. Usa el CSV ya construido con datos Sentinel por `parcela + fecha`, genera variables temporales y espectrales, entrena modelos comparables y muestra resultados en una web sencilla.

## Dataset

Los datos de entrada están en `data/raw/`:

- `sentinel_stress_by_parcel_20160101_to_20260510.csv`: dataset principal por parcela y fecha.
- `Jerovia - Parcelas - Coordenadas (1).xlsx`: coordenadas de parcelas para generar `parcels.geojson`.

La variable objetivo original es `stress_index`, una etiqueta derivada entre 0 y 1. Debe interpretarse como una aproximación o "silver truth", no como una medición directa de campo.

El script de preparación calcula:

- `ndvi = (B8A - B4) / (B8A + B4)`
- `ndmi = (B8A - B12) / (B8A + B12)`
- `msi = B12 / B8A`
- variables de calendario, lags, diferencias, medias móviles y días desde la observación anterior.

Para el objetivo supervisado se crea `target_stress_7d`, buscando por parcela la observación futura más cercana a `fecha + 7 días`, con tolerancia controlada. Las filas sin futuro útil se eliminan del entrenamiento.

También puede generarse `target_stress_5d` con el mismo criterio para comparar predicciones a 5 días.

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

En este equipo también se puede usar el Python de Anaconda con:

```bash
py -3.13 scripts/prepare_dataset.py
```

## Preparación de datos

```bash
python scripts/prepare_dataset.py
python scripts/prepare_dataset.py --horizon-days 5
python scripts/prepare_dataset.py --horizon-days 7
```

Genera:

- `data/processed/dataset_modeling.csv`
- `data/processed/dataset_modeling_h5.csv`
- `data/processed/dataset_modeling_h7.csv`
- `data/processed/parcels.geojson`

`dataset_modeling.csv` se mantiene como alias de compatibilidad del horizonte de 7 días.

Si el Excel no permite generar el GeoJSON, se crea `data/processed/parcels_geojson_warning.txt` y el resto del proyecto sigue funcionando.

## Entrenamiento

Cada modelo puede ejecutarse por separado:

```bash
python scripts/train_random_forest.py
python scripts/train_random_forest.py --horizon-days 5
python scripts/train_xgboost.py
python scripts/train_lstm.py
python scripts/train_gru.py
python scripts/train_cnn_lstm.py
python scripts/train_convlstm.py
python scripts/train_chronos_bolt.py
python scripts/train_tft.py
python scripts/train_tcn.py
python scripts/train_prophet.py
python scripts/train_sarimax.py
```

Para preparar datos y lanzar todos los modelos uno a uno:

```bash
python scripts/train_all_models.py
python scripts/train_all_models.py --horizons 5 7
```

Si el dataset ya está preparado:

```bash
python scripts/train_all_models.py --skip-prepare
python scripts/train_all_models.py --skip-prepare --horizons 5 7
```

Los scripts con dependencias opcionales registran un JSON explicativo si no pueden ejecutarse. Esto permite que el pipeline completo no se rompa por falta de dependencias avanzadas.

En este MVP, Chronos-Bolt, ConvLSTM y TFT se ejecutan con versiones adaptadas al dataset disponible:

- Chronos-Bolt usa un baseline temporal local de persistencia por parcela cuando no está instalada la librería fundacional.
- ConvLSTM transforma cada vector de características en una pequeña rejilla pseudo-espacial, porque el CSV no contiene imágenes reales.
- TFT usa una versión ligera en PyTorch con selección/gating de variables y atención temporal, evitando la dependencia pesada `pytorch-forecasting`.

## Resultados

Los resultados se guardan de forma homogénea:

- `outputs/models/`: modelos entrenados.
- `outputs/predictions/`: CSV con sufijo de horizonte, por ejemplo `random_forest_h5_predictions.csv` y `random_forest_h7_predictions.csv`.
- `outputs/metrics/`: JSON por modelo/horizonte y `model_comparison.csv` con columna `horizon_days`.
- `outputs/plots/`: gráficas real vs predicho con sufijo `_h5` o `_h7`.

## Web

```bash
streamlit run web/app.py
```

La web se organiza en tres pestañas:

- `Información por parcela`: mapa, histórico de `stress_index` y gráfica de real histórico, predicción y real evaluado.
- `Predicción climática`: lee `data/raw/climate_forecast.csv` si existe o permite cargar un CSV desde la web.
- `Rendimiento del modelo`: permite elegir horizonte de 5 o 7 días, modelo, métricas, gráfica real vs predicho, mejores modelos y tabla de predicciones.

Las parcelas del GeoJSON sin nombre o sin cruce con el CSV Sentinel se mantienen en gris. En el dataset actual hay 18 parcelas de este tipo; no representan bajo estrés, sino ausencia de datos/predicción.

Para la pestaña climática, el CSV esperado es:

```csv
forecast_date,nombre_parcela,temp_min_c,temp_max_c,temp_mean_c,precipitation_mm,humidity_pct,wind_m_s,et0_mm
2026-05-12,A1,18.2,31.4,24.7,0.0,58,3.1,4.8
```

También puede descargarse automáticamente desde Open-Meteo usando los centroides de las parcelas:

```bash
python scripts/download_climate_forecast.py --forecast-days 7
```

El script guarda `data/raw/climate_forecast.csv`, que la web leerá directamente.

En este equipo, si usas el Python de Anaconda:

```powershell
py -3.13 -m streamlit run web/app.py
```

Después abre:

```text
http://localhost:8501
```

Para cerrar la web, vuelve a la terminal donde está corriendo Streamlit y pulsa:

```text
Ctrl + C
```

Si quedó ejecutándose en segundo plano, puedes localizar el proceso y cerrarlo desde PowerShell:

```powershell
Get-Process python,py
Stop-Process -Id NUMERO_DEL_PID
```

## Modelos incluidos

- Random Forest: baseline tabular fuerte con scikit-learn.
- XGBoost: baseline tabular opcional si está instalado.
- LSTM, GRU, CNN-LSTM y TCN: modelos PyTorch de secuencias por parcela si PyTorch está instalado.
- ConvLSTM: aproximación tabular que reordena features en una rejilla pseudo-espacial.
- Chronos-Bolt: baseline temporal local de persistencia, reemplazable por Chronos real cuando esté disponible.
- TFT: aproximación ligera en PyTorch con gating de variables y atención temporal.
- Prophet: baseline interpretable sobre la serie agregada semanal media.
- SARIMAX: baseline estadístico sobre la serie agregada media.

## Limitaciones

- `stress_index` es una etiqueta derivada, no una medición agronómica directa.
- Las fechas de Sentinel no son perfectamente regulares.
- El dataset actual no contiene imágenes completas; CNN-LSTM y ConvLSTM son aproximaciones tabulares. Si se incorporan tiles o tensores espaciales reales, ConvLSTM debería adaptarse a esa entrada.
- La predicción a 5 o 7 días depende de cómo se construye `target_stress_5d` o `target_stress_7d`.
- Si no hay variables meteorológicas explícitas en el CSV, los modelos trabajan con las variables disponibles.
- El sistema no debe considerarse operativo en campo sin validación agronómica adicional.
