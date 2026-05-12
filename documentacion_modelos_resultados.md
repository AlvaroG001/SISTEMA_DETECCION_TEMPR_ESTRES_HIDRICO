# Documentación de modelos, pruebas y conclusiones

## 1. Objetivo de las pruebas

El objetivo de esta fase ha sido comparar distintos modelos de predicción para estimar el índice de estrés hídrico de cada parcela a dos horizontes temporales:

- Predicción a 5 días.
- Predicción a 7 días.

La intención no era elegir el modelo más complejo, sino identificar qué tipo de enfoque funciona mejor con el dataset disponible y dejar una comparación clara, reproducible y fácil de explicar.

Para ello se ha preparado un pipeline común donde todos los modelos:

- usan la misma estructura de datos de entrada;
- respetan una división temporal de entrenamiento, validación y test;
- generan predicciones en el mismo formato;
- guardan métricas comparables;
- pueden visualizarse desde la web.

## Glosario de siglas

En la documentación se utilizan las siguientes siglas:

| Sigla con significado | Uso en el proyecto |
|---|---|
| MVP (Minimum Viable Product / Producto Mínimo Viable) | Primera versión funcional del sistema. |
| CSV (Comma-Separated Values / Valores separados por comas) | Formato de los ficheros de datos y predicciones. |
| MAE (Mean Absolute Error / Error absoluto medio) | Métrica principal para comparar modelos; cuanto menor, mejor. |
| RMSE (Root Mean Squared Error / Raíz del error cuadrático medio) | Métrica que penaliza más los errores grandes; cuanto menor, mejor. |
| R2 (coeficiente de determinación) | Mide cuánto explica el modelo de la variabilidad de los datos; cuanto más cerca de 1, mejor. |
| LSTM (Long Short-Term Memory) | Red neuronal recurrente para series temporales. |
| GRU (Gated Recurrent Unit) | Red neuronal recurrente más ligera que LSTM (Long Short-Term Memory). |
| CNN (Convolutional Neural Network) | Red neuronal convolucional. |
| CNN-LSTM (Convolutional Neural Network + Long Short-Term Memory) | Modelo que combina convoluciones y memoria temporal. |
| ConvLSTM (Convolutional Long Short-Term Memory) | Variante de LSTM (Long Short-Term Memory) con estructura convolucional. |
| TCN (Temporal Convolutional Network) | Red convolucional para secuencias temporales. |
| TFT (Temporal Fusion Transformer) | Arquitectura avanzada para predicción temporal con atención y selección de variables. |
| SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors) | Modelo estadístico clásico de series temporales. |
| JSON (JavaScript Object Notation) | Formato estructurado usado para guardar métricas y salidas explicativas. |
| GeoJSON (Geographic JavaScript Object Notation) | Formato geográfico usado para pintar las parcelas. |
| XGBoost (Extreme Gradient Boosting) | Modelo tabular basado en boosting. |
| NDVI (Normalized Difference Vegetation Index) | Índice de vegetación calculado con bandas Sentinel. |
| NDMI (Normalized Difference Moisture Index) | Índice relacionado con humedad de la vegetación. |
| MSI (Moisture Stress Index) | Índice asociado a estrés hídrico. |
| ET0 (evapotranspiración de referencia) | Variable climática usada como referencia de demanda evaporativa. |

## 2. Modelos evaluados

Se han probado modelos de distintas familias para comparar enfoques simples, modelos temporales y modelos estadísticos.

### Random Forest

Random Forest se ha usado como baseline tabular principal. Es un modelo robusto, relativamente rápido y adecuado cuando las relaciones entre variables no son puramente lineales.

En las pruebas ha sido el modelo con mejor rendimiento global. Esto lo convierte en el candidato principal para el MVP (Minimum Viable Product / Producto Mínimo Viable).

### XGBoost (Extreme Gradient Boosting)

XGBoost (Extreme Gradient Boosting) se ha incluido como segundo baseline tabular fuerte. Suele funcionar bien en problemas estructurados y permite capturar interacciones complejas entre variables.

En los resultados queda muy cerca de Random Forest, especialmente en el horizonte de 7 días. Es una buena alternativa si se quiere seguir ajustando hiperparámetros.

### LSTM (Long Short-Term Memory)

LSTM (Long Short-Term Memory) se ha probado como modelo secuencial para capturar dependencias temporales dentro de cada parcela.

Funciona correctamente, pero no supera a los modelos tabulares. Esto indica que, para este dataset, el histórico ya queda suficientemente resumido con las variables generadas y los modelos tabulares lo aprovechan mejor.

### GRU (Gated Recurrent Unit)

GRU (Gated Recurrent Unit) se ha incluido como alternativa más ligera a LSTM (Long Short-Term Memory). Tiene una lógica parecida, pero con una arquitectura más simple.

Su rendimiento es razonable, aunque tampoco supera a Random Forest ni XGBoost (Extreme Gradient Boosting). Puede ser interesante si se quiere mantener un modelo temporal más ligero.

### CNN-LSTM (Convolutional Neural Network + Long Short-Term Memory)

CNN-LSTM (Convolutional Neural Network + Long Short-Term Memory) se ha implementado como una aproximación tabular. La parte convolucional trabaja sobre la secuencia de variables y después una LSTM (Long Short-Term Memory) modela la evolución temporal.

Ha obtenido resultados competitivos dentro de los modelos de deep learning, pero sigue quedando por detrás de los modelos tabulares.

### TCN (Temporal Convolutional Network)

TCN (Temporal Convolutional Network) usa convoluciones temporales para modelar la serie de observaciones. Es una alternativa a los modelos recurrentes.

En las pruebas no ha sido de los mejores modelos. Sirve como comparación, pero no parece el enfoque más fuerte para este MVP (Minimum Viable Product / Producto Mínimo Viable).

### ConvLSTM (Convolutional Long Short-Term Memory)

ConvLSTM (Convolutional Long Short-Term Memory) se ha dejado como una versión adaptada al dataset actual. Como no se trabaja con imágenes completas, las variables se reorganizan como una rejilla pseudo-espacial.

Funciona como prueba técnica, pero no debe interpretarse como un ConvLSTM (Convolutional Long Short-Term Memory) plenamente espacial. Su utilidad real aumentaría si en el futuro se incorporan imágenes o tensores por parcela.

### TFT (Temporal Fusion Transformer)

TFT (Temporal Fusion Transformer) se ha implementado como una versión ligera inspirada en Temporal Fusion Transformer, usando selección de variables y atención temporal.

Es útil como comparación avanzada, pero en esta versión no supera a los modelos tabulares. Puede ser una línea futura si se amplía el dataset o se añaden más covariables temporales.

### Chronos-Bolt

Chronos-Bolt se ha incluido como baseline temporal ejecutable mediante una estrategia local de persistencia, usando el último valor observado como referencia.

No se han cargado pesos reales de un modelo fundacional Chronos. Por tanto, en esta versión sirve como referencia temporal simple y no como evaluación completa de Chronos-Bolt.

### Prophet

Prophet se ha probado sobre una serie agregada semanal. Es un modelo interpretable y útil como referencia estadística.

Su rendimiento es bajo frente a los modelos por parcela porque pierde detalle individual y trabaja sobre una serie agregada.

### SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors)

SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors) se ha incluido como baseline estadístico clásico, también sobre una serie agregada.

Al igual que Prophet, sirve como referencia interpretable, pero no compite bien con los modelos que trabajan directamente por parcela.

## 3. Cómo se han comparado los modelos

Todos los modelos se han evaluado con métricas de regresión:

- MAE (Mean Absolute Error / Error absoluto medio).
- RMSE (Root Mean Squared Error / Raíz del error cuadrático medio).
- R2 (coeficiente de determinación).

La métrica principal para comparar ha sido MAE (Mean Absolute Error / Error absoluto medio), porque mide el error medio absoluto y es fácil de interpretar. RMSE (Root Mean Squared Error / Raíz del error cuadrático medio) se usa como apoyo porque penaliza más los errores grandes. R2 (coeficiente de determinación) ayuda a ver la capacidad explicativa general del modelo.

La división de datos se ha hecho de forma temporal:

- entrenamiento con fechas antiguas;
- validación con fechas intermedias;
- test con fechas más recientes.

Esto es importante porque evita evaluar el modelo con información mezclada del futuro.

## 4. Resultados principales

En las pruebas realizadas, los mejores modelos han sido:

| Horizonte | Mejor modelo | Resultado aproximado |
|---|---|---|
| 5 días | Random Forest | MAE (Mean Absolute Error / Error absoluto medio) ≈ 0.052 |
| 7 días | Random Forest | MAE (Mean Absolute Error / Error absoluto medio) ≈ 0.058 |

XGBoost (Extreme Gradient Boosting) queda como segunda mejor opción:

| Horizonte | Modelo | Resultado aproximado |
|---|---|---|
| 5 días | XGBoost (Extreme Gradient Boosting) | MAE (Mean Absolute Error / Error absoluto medio) ≈ 0.063 |
| 7 días | XGBoost (Extreme Gradient Boosting) | MAE (Mean Absolute Error / Error absoluto medio) ≈ 0.061 |

Los modelos secuenciales y avanzados funcionan, pero en general quedan por detrás de Random Forest y XGBoost (Extreme Gradient Boosting).

Los modelos agregados, como Prophet y SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors), son los que peor rendimiento ofrecen en esta comparación.

## 5. Conclusiones de las pruebas

La primera conclusión es que los modelos tabulares funcionan mejor para este MVP (Minimum Viable Product / Producto Mínimo Viable). Random Forest y XGBoost (Extreme Gradient Boosting) aprovechan muy bien la información histórica y las variables ya preparadas.

La segunda conclusión es que el horizonte de 5 días suele ser más fácil de predecir que el de 7 días. Esto tiene sentido porque la predicción está más cerca de la última observación disponible y hay menos incertidumbre temporal.

La tercera conclusión es que los modelos de deep learning no aportan una mejora clara en esta versión. Aunque LSTM (Long Short-Term Memory), GRU (Gated Recurrent Unit), CNN-LSTM (Convolutional Neural Network + Long Short-Term Memory), TCN (Temporal Convolutional Network), TFT (Temporal Fusion Transformer) y ConvLSTM (Convolutional Long Short-Term Memory) se ejecutan correctamente, su complejidad no se traduce en mejores métricas.

La cuarta conclusión es que los modelos agregados pierden demasiada información. Prophet y SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors) son útiles para tener una referencia interpretable, pero no son los más adecuados si el objetivo es predecir parcela por parcela.

La quinta conclusión es que el modelo recomendado para la demo y para la primera versión del sistema es Random Forest. Es el más preciso, estable y fácil de explicar. XGBoost (Extreme Gradient Boosting) queda como alternativa fuerte.

## 6. Modelo recomendado

Para el MVP (Minimum Viable Product / Producto Mínimo Viable) se recomienda usar:

```text
Random Forest
```

Motivos:

- obtiene el menor error en test;
- funciona bien tanto a 5 como a 7 días;
- es más fácil de entrenar y mantener que los modelos deep learning;
- no necesita dependencias pesadas;
- produce resultados estables;
- es sencillo de explicar en una memoria técnica.

Como segunda opción se recomienda:

```text
XGBoost (Extreme Gradient Boosting)
```

Motivos:

- obtiene resultados muy cercanos a Random Forest;
- es un baseline tabular muy competitivo;
- podría mejorar con ajuste de hiperparámetros.

## 7. Líneas futuras

Las mejoras más relevantes serían:

- ajustar hiperparámetros de Random Forest y XGBoost (Extreme Gradient Boosting);
- probar validación temporal por campañas agrícolas;
- incorporar predicción climática real como covariable de entrada;
- entrenar modelos específicos por grupo de parcelas;
- evaluar si ConvLSTM (Convolutional Long Short-Term Memory) o CNN-LSTM (Convolutional Neural Network + Long Short-Term Memory) mejoran cuando existan imágenes o tensores espaciales reales;
- comparar Chronos-Bolt real cargando pesos del modelo fundacional.

## 8. Resumen final

El sistema ha conseguido entrenar y comparar múltiples modelos bajo una misma estructura. Las pruebas muestran que, para el dataset actual, la mejor solución no es la más compleja, sino la más robusta.

Random Forest es el modelo recomendado para la versión actual. XGBoost (Extreme Gradient Boosting) queda como alternativa principal. Los modelos temporales profundos se mantienen como comparación y posible evolución futura, pero no superan a los modelos tabulares en las pruebas actuales.

La predicción a 5 días presenta mejores resultados que la predicción a 7 días, por lo que ambos horizontes son útiles: 5 días para mayor precisión y 7 días para anticipación algo mayor.
