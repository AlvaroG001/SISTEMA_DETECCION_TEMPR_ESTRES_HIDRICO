# Prediccion de estres hidrico por parcela a 7 dias

Este repositorio es un MVP tecnico para predecir el `stress_index` de parcelas agricolas aproximadamente a 7 dias vista. Usa el CSV ya construido con datos Sentinel por `parcela + fecha`, genera variables temporales y espectrales, entrena modelos comparables y muestra resultados en una web sencilla.

## Dataset

Los datos de entrada estan en `data/raw/`:

- `sentinel_stress_by_parcel_20160101_to_20260510.csv`: dataset principal por parcela y fecha.
- `Jerovia - Parcelas - Coordenadas (1).xlsx`: coordenadas de parcelas para generar `parcels.geojson`.

La variable objetivo original es `stress_index`, una etiqueta derivada entre 0 y 1. Debe interpretarse como una aproximacion o "silver truth", no como una medicion directa de campo.

El script de preparacion calcula:

- `ndvi = (B8A - B4) / (B8A + B4)`
- `ndmi = (B8A - B12) / (B8A + B12)`
- `msi = B12 / B8A`
- variables de calendario, lags, diferencias, medias moviles y dias desde la observacion anterior.

Para el objetivo supervisado se crea `target_stress_7d`, buscando por parcela la observacion futura mas cercana a `fecha + 7 dias`, con tolerancia controlada. Las filas sin futuro util se eliminan del entrenamiento.

## Instalacion

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

En este equipo tambien se puede usar el Python de Anaconda con:

```bash
py -3.13 scripts/prepare_dataset.py
```

## Preparacion de datos

```bash
python scripts/prepare_dataset.py
```

Genera:

- `data/processed/dataset_modeling.csv`
- `data/processed/parcels.geojson`

Si el Excel no permite generar el GeoJSON, se crea `data/processed/parcels_geojson_warning.txt` y el resto del proyecto sigue funcionando.

## Entrenamiento

Cada modelo puede ejecutarse por separado:

```bash
python scripts/train_random_forest.py
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
```

Si el dataset ya esta preparado:

```bash
python scripts/train_all_models.py --skip-prepare
```

Los scripts con dependencias opcionales registran un JSON explicativo si no pueden ejecutarse. Esto permite que el pipeline completo no se rompa por falta de dependencias avanzadas.

En este MVP, Chronos-Bolt, ConvLSTM y TFT se ejecutan con versiones adaptadas al dataset disponible:

- Chronos-Bolt usa un baseline temporal local de persistencia por parcela cuando no esta instalada la libreria fundacional.
- ConvLSTM transforma cada vector de caracteristicas en una pequena rejilla pseudo-espacial, porque el CSV no contiene imagenes reales.
- TFT usa una version ligera en PyTorch con seleccion/gating de variables y atencion temporal, evitando la dependencia pesada `pytorch-forecasting`.

## Resultados

Los resultados se guardan de forma homogenea:

- `outputs/models/`: modelos entrenados.
- `outputs/predictions/`: CSV con `parcela_id`, `nombre_parcela`, `fecha`, `target_date`, `y_true`, `y_pred`, `model_name`, `split`.
- `outputs/metrics/`: JSON por modelo y `model_comparison.csv`.
- `outputs/plots/`: graficas real vs predicho.

## Web

```bash
streamlit run web/app.py
```

La web permite seleccionar un modelo con predicciones guardadas, ver metricas, mapa de parcelas coloreado por estres predicho, historico por parcela y tabla de ultimas predicciones.

En este equipo, si usas el Python de Anaconda:

```powershell
py -3.13 -m streamlit run web/app.py
```

Despues abre:

```text
http://localhost:8501
```

Para cerrar la web, vuelve a la terminal donde esta corriendo Streamlit y pulsa:

```text
Ctrl + C
```

Si quedo ejecutandose en segundo plano, puedes localizar el proceso y cerrarlo desde PowerShell:

```powershell
Get-Process python,py
Stop-Process -Id NUMERO_DEL_PID
```

## Modelos incluidos

- Random Forest: baseline tabular fuerte con scikit-learn.
- XGBoost: baseline tabular opcional si esta instalado.
- LSTM, GRU, CNN-LSTM y TCN: modelos PyTorch de secuencias por parcela si PyTorch esta instalado.
- ConvLSTM: aproximacion tabular que reordena features en una rejilla pseudo-espacial.
- Chronos-Bolt: baseline temporal local de persistencia, reemplazable por Chronos real cuando este disponible.
- TFT: aproximacion ligera en PyTorch con gating de variables y atencion temporal.
- Prophet: baseline interpretable sobre la serie agregada semanal media.
- SARIMAX: baseline estadistico sobre la serie agregada media.

## Limitaciones

- `stress_index` es una etiqueta derivada, no una medicion agronomica directa.
- Las fechas de Sentinel no son perfectamente regulares.
- El dataset actual no contiene imagenes completas; CNN-LSTM y ConvLSTM son aproximaciones tabulares. Si se incorporan tiles o tensores espaciales reales, ConvLSTM deberia adaptarse a esa entrada.
- La prediccion a 7 dias depende de como se construye `target_stress_7d`.
- Si no hay variables meteorologicas explicitas en el CSV, los modelos trabajan con las variables disponibles.
- El sistema no debe considerarse operativo en campo sin validacion agronomica adicional.
