# Memoria del Proyecto: Modelo XGBoost Optimizado para la Predicción de Cancelaciones Hoteleras

## Índice

1. [Introducción: El Desafío de las Cancelaciones Anticipadas](#introducción-el-desafío-de-las-cancelaciones-anticipadas)
2. [Mirando Atrás: Lecciones Aprendidas del Primer Intento](#mirando-atrás-lecciones-aprendidas-del-primer-intento)
3. [El Nuevo Enfoque: Preprocesamiento y Optimización del Modelo](#el-nuevo-enfoque-preprocesamiento-y-optimización-del-modelo)
    - 3.1. [Preparación y Análisis de Datos](#preparación-y-análisis-de-datos)
    - 3.2. [Ingeniería de Características Avanzada](#ingeniería-de-características-avanzada)
    - 3.3. [Estrategia de Gestión de Valores Nulos](#estrategia-de-gestión-de-valores-nulos)
    - 3.4. [Simplicidad y Generalización: De Stacking a XGBoost](#simplicidad-y-generalización-de-stacking-a-xgboost)
    - 3.5. [Manejo del Desbalanceo de Clases](#manejo-del-desbalanceo-de-clases)
    - 3.6. [Validación Estratificada por Grupos](#validación-estratificada-por-grupos)
    - 3.7. [Optimización del Umbral de Decisión](#optimización-del-umbral-de-decisión)
4. [Evaluación del Modelo y Resultados](#evaluación-del-modelo-y-resultados)
    - 4.1. [Análisis de la Distribución de Probabilidades y Umbral Óptimo](#análisis-de-la-distribución-de-probabilidades-y-umbral-óptimo)
5. [Conclusiones y Direcciones Futuras](#conclusiones-y-direcciones-futuras)
6. Resultados del Script de Inferencia
7. Verificación de Docker

## Introducción: El Desafío de las Cancelaciones Anticipadas

En la industria hotelera, las cancelaciones anticipadas representan un desafío crítico para la gestión eficiente del inventario y la maximización de ingresos. Predecir qué reservas tienen alta probabilidad de cancelarse con al menos 30 días de antelación permite implementar estrategias de overbooking controlado y promociones dirigidas, mitigando así el impacto económico de las habitaciones no ocupadas.

Nuestro objetivo se centra en desarrollar un sistema de predicción que identifique con alta precisión las reservas con mayor riesgo de cancelación anticipada, buscando un equilibrio óptimo entre precisión y recall para maximizar el valor de negocio.

## Mirando Atrás: Lecciones Aprendidas del Primer Intento

Nuestro primer trabajo en este problema utilizó un modelo XGBoost único optimizado mediante búsqueda de hiperparámetros. Aunque mostró resultados prometedores, la evaluación y el feedback recibido revelaron importantes deficiencias:

- **Alta variabilidad entre folds**: Las métricas F1 mostraban fluctuaciones significativas en diferentes subconjuntos de datos, indicando problemas de estabilidad.
- **Preprocesamiento fragmentado**: El proceso de transformación de datos se realizaba en múltiples etapas, complicando la reproducibilidad y aumentando el riesgo de fugas de datos.
- **Manejo subóptimo del desbalanceo**: La clase minoritaria (cancelaciones anticipadas) no se trataba eficientemente, siendo este uno de los problemas principales del trabajo.
- **Estrategia de validación inadecuada**: No se utilizaba cross validation excepto en GridSearch, y no se eligió la estrategia más adecuada para este problema, ignorando la estructura de grupo natural de los datos (reservas del mismo hotel).
- **Error crítico en la etapa de inferencia**: Eliminamos incorrectamente los registros de tipo "Booked" en inferencia, lo que nos dejaba sin filas para hacer predicciones. Además, clasificamos erróneamente los "No-show" como cancelaciones cuando realmente son clientes que han pagado.
- **Definición incorrecta del target**: Definimos la variable objetivo incorrectamente como cancelaciones con 30 días o más de anticipación, cuando debíamos predecir cancelaciones con 30 días o menos de anticipación.

Estas observaciones críticas del feedback anterior fueron fundamentales para dirigir nuestro desarrollo hacia un enfoque más robusto.

## El Nuevo Enfoque: Preprocesamiento y Optimización del Modelo

### Preparación y Análisis de Datos

El proceso de preparación de datos se mejoró significativamente integrándolo en una estructura coherente:

```python
def get_X_y():
    """
    Carga y preprocesa los datos para el entrenamiento del modelo.
    Crea la variable objetivo y define los features.
    """
    hotels = pd.read_csv(os.environ.get("HOTELS_DATA_PATH", "data/hotels.csv"))
    bookings = pd.read_csv(os.environ.get("TRAIN_DATA_PATH", "data/bookings_train.csv"))

    # Une las tablas de reservas con los hoteles
    data = pd.merge(bookings, hotels, on='hotel_id', how='left')

    # Filtra solo reservas finalizadas o canceladas (excluye las aún en estado "Booked")
    data = data[data['reservation_status'] != 'Booked'].copy()

    # Considera los "No-Show" como "Check-Out" para el target
    data['reservation_status'].replace('No-Show', 'Check-Out', inplace=True)

    # Convierte fechas a formato datetime
    for col in ['arrival_date', 'booking_date', 'reservation_status_date']:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col])

    # Días entre fecha de llegada y la fecha del estado de la reserva
    data['days_before_arrival'] = (data['arrival_date'] - data['reservation_status_date']).dt.days

    # CORRECCIÓN: Define el target como cancelaciones con 30 días o MENOS de anticipación
    data['target'] = ((data['reservation_status'] == 'Canceled') &
                       (data['days_before_arrival'] <= 30)).astype(int)
```

Esta corrección aborda varios errores críticos identificados en el feedback anterior:

- **Corrección de la definición del target**: Ahora definimos correctamente el objetivo como predicción de cancelaciones con 30 días o menos de anticipación (`days_before_arrival <= 30`).
- **Interpretación correcta de "No-Show"**: Clasificamos correctamente los registros "No-Show" como "Check-Out" puesto que son clientes que efectivamente han pagado.
- **Filtrado adecuado**: Solo filtramos registros "Booked" durante el entrenamiento, manteniendo la coherencia con el procedimiento de inferencia.

El análisis de la distribución del target muestra un claro desbalanceo de clases:

```plaintext
--- Distribución Original del Target (y) ---
target
0    38442
1     8178
Name: count, dtype: int64
```

Porcentaje de clase positiva (1 = Cancelación <= 30 días): 17.54%
Total de registros: 46620

Este desbalanceo, con solo el 17.54% de las reservas resultando en cancelaciones en los 30 días previos a la llegada, es un aspecto importante a considerar en nuestro enfoque de modelado.

### Ingeniería de Características Avanzada

La ingeniería de características se enfocó en incorporar conocimiento del dominio, creando tan solo un indicador predictivo clave:

```python
# Crear características clave
data['lead_time'] = (data['arrival_date'] - data['booking_date']).dt.days
```

- **Lead time**: Días entre reserva y llegada, un predictor clásico de cancelaciones, y suficientemente correlacionado con el target, pero sin causar data leakage.

Consideramos la posibilidad de crear características agregadas por grupo de hotel, pero decidimos no implementarlas debido al riesgo de data leakage. Estas características podrían haber incluido tasas históricas de cancelación por hotel o patrones estacionales específicos por establecimiento, pero su implementación inadecuada podría introducir fugas de información del conjunto de validación al de entrenamiento.

### Estrategia de Gestión de Valores Nulos

Un avance significativo fue la integración de la imputación de valores nulos dentro del pipeline de preprocesamiento:

```python
# Pipeline para variables numéricas: imputación y escalado
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Pipeline para variables categóricas: imputación y one-hot encoding
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# Aplica transformaciones por tipo de columna
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_features + bool_features),
        ("cat", categorical_transformer, cat_features),
    ]
)
```

Esta implementación presenta varias ventajas clave:

- **Tratamiento específico por tipo de dato**: Cada tipo de característica recibe un tratamiento diferenciado:
  - **Variables numéricas**: Imputación por mediana y escalado estándar.
  - **Variables categóricas**: Imputación por moda y codificación one-hot con `handle_unknown="ignore"` para manejar categorías nuevas en inferencia.
- **Prevención de data leakage**: Al encapsular todo el preprocesamiento en `ColumnTransformer`, las estadísticas para la imputación (medianas, modas) y el escalado (medias, desviaciones estándar) se calculan solo en el conjunto de entrenamiento y se aplican de manera consistente tanto en entrenamiento como en inferencia.
- **Manejo coherente de valores ausentes**: La estrategia garantiza que los valores nulos se traten de manera coherente durante todo el ciclo de vida del modelo.

### Simplicidad y Generalización: De Stacking a XGBoost

Una decisión crítica en este proyecto fue abandonar la arquitectura de ensemble stacking inicialmente propuesta en favor de un modelo XGBoost único pero bien optimizado. Esta decisión se basó en evidencia empírica de que el modelo stacking podría estar provocando overfitting, especialmente considerando el desbalanceo de clases y la estructura de los datos.

```python
# Modelo base: XGBoost con configuración personalizada
model = XGBClassifier(
    objective='binary:logistic',
    tree_method='hist',
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42,
    min_child_weight=4,  # Previene overfitting
    learning_rate=0.05,  # Aprendizaje lento para mejor generalización
    n_estimators=300,    # Suficientes árboles para aprender sin overfitting
    gamma=0.1,           # Regularización para controlar overfitting
)
```

Los hiperparámetros fueron cuidadosamente seleccionados para favorecer la generalización:

- `min_child_weight=4`: Un valor más alto que el predeterminado para prevenir que el modelo se ajuste demasiado a patrones específicos en los datos de entrenamiento.
- `learning_rate=0.05`: Una tasa de aprendizaje relativamente baja que permite al modelo converger más suavemente, reduciendo la probabilidad de overfitting.
- `gamma=0.1`: Un parámetro de regularización que controla la creación de nodos adicionales en los árboles, limitando la complejidad del modelo.

Esta configuración busca un equilibrio deliberado entre ajuste a los datos y capacidad de generalización, favoreciendo ligeramente la generalización para asegurar un rendimiento estable en datos no vistos.

### Manejo del Desbalanceo de Clases

El desbalanceo entre clases (17.54% de cancelaciones con ≤ 30 días de anticipación vs. 82.46% de no cancelaciones) fue uno de los problemas principales identificados en el feedback anterior. Para abordar este desafío, implementamos una estrategia de SMOTE cuidadosamente calibrada:

```python
# Pipeline completo con SMOTE para reequilibrar clases
pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42, sampling_strategy=0.6, k_neighbors=5)),
    ('classifier', model)
])
```

La configuración de SMOTE incluye características importantes:

- `sampling_strategy=0.6`: En lugar de igualar completamente las clases (lo que podría introducir demasiada información sintética), optamos por un valor moderado de 0.6, lo que significa que la clase minoritaria alcanzará aproximadamente el 60% del tamaño de la clase mayoritaria. Esta decisión equilibra la necesidad de abordar el desbalanceo sin introducir demasiados datos sintéticos.
- `k_neighbors=5`: Un valor moderado para `k_neighbors` que permite generar instancias sintéticas basadas en vecinos reales, manteniendo la credibilidad de los datos generados.

Esta estrategia deliberadamente conservadora de SMOTE está diseñada para mejorar la detección de la clase minoritaria sin ir tan lejos como para generar datos sintéticos excesivos que podrían confundir al modelo con patrones irreales.

### Validación Estratificada por Grupos

Mantuvimos e incluso mejoramos la estrategia de validación cruzada estratificada por grupos, utilizando `StratifiedGroupKFold` con 7 folds:

```python
# Validación cruzada estratificada por grupo (hotel)
cv = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=42)
```

Esta implementación es crucial por varias razones:

- **Agrupación por `hotel_id`**: Todas las reservas de un mismo hotel permanecen juntas ya sea en entrenamiento o validación, nunca divididas entre ambos. Esto previene el data leakage y asegura que estamos evaluando la capacidad del modelo para generalizar a hoteles con características similares pero no idénticas.
- **Estratificación por clase**: Cada fold mantiene aproximadamente la misma proporción de cancelaciones anticipadas, asegurando que la distribución de clases sea similar en todos los folds.
- **Número óptimo de folds**: La elección de 7 folds proporciona suficiente diversidad de datos para entrenar modelos robustos mientras mantiene conjuntos de validación significativos.

Esta estrategia de validación nos permite obtener una evaluación más realista del rendimiento del modelo en nuevos datos, evitando la "fuga" de información entre los conjuntos de entrenamiento y validación.

### Optimización del Umbral de Decisión

Un componente crítico del proceso es la optimización del umbral de decisión para maximizar el F1-score global:

```python
def find_threshold(y_true, y_pred_proba):
    """
    Encuentra el mejor umbral de decisión para maximizar el F1-score.
    """
    best_f1, best_threshold = 0, 0.5
    thresholds = np.linspace(0.1, 0.8, 40)

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)

        if f1 > best_f1:
            best_f1, best_threshold = f1, threshold

    return best_threshold, best_f1
```

Esta búsqueda sistemática del umbral óptimo explora 40 valores entre 0.1 y 0.8, seleccionando aquel que maximiza el F1-score.

Los resultados muestran que el umbral óptimo para nuestro modelo es 0.441, un valor razonable que refleja un equilibrio entre precisión y recall:

- **Umbral óptimo seleccionado**: 0.4410 (F1: 0.5867)

Este umbral cercano al 0.5 estándar sugiere que nuestro modelo está bien calibrado, a diferencia de la versión anterior que requería un umbral extremadamente alto (0.95) para obtener resultados aceptables. El umbral más equilibrado obtenido es indicativo de un modelo que generaliza mejor y gestiona más eficazmente el desbalanceo de clases.

## Evaluación del Modelo y Resultados

La evaluación del modelo utilizando el umbral optimizado muestra un rendimiento sólido:

```plaintext
------ Evaluación del Modelo ------
Umbral aplicado: 0.4410

Métricas principales:
→ Precision:   0.5572 (De las predicciones positivas, ¿cuántas son correctas?)
→ Recall:      0.6196 (De todos los positivos reales, ¿cuántos detectamos?)
→ F1-Score:    0.5867 (Media armónica entre precision y recall)
→ AUC-ROC:     0.8601 (Capacidad discriminativa general del modelo)

Métricas adicionales:
→ Accuracy:    0.8469 (Porcentaje general de aciertos)
→ Specificity: 0.8952 (De los negativos reales, ¿cuántos detectamos?)

Matriz de confusión:
→ Verdaderos Negativos (TN): 34415 (No cancela y predecimos que no cancela)
→ Falsos Positivos (FP):    4027 (No cancela pero predecimos cancelación)
→ Falsos Negativos (FN):    3111 (Cancela pero no lo detectamos)
→ Verdaderos Positivos (TP): 5067 (Cancela y predecimos correctamente)
```

Estos resultados muestran un modelo que logra un equilibrio efectivo entre precisión (55.72%) y recall (61.96%), lo que se traduce en un F1-Score de 0.5867. Este equilibrio es crucial para el problema de negocio, donde tanto los falsos positivos como los falsos negativos tienen costos asociados.

El AUC-ROC de 0.8601 indica una excelente capacidad discriminativa del modelo, muy por encima del nivel aleatorio (0.5), lo que confirma que el modelo está capturando patrones valiosos en los datos.

Desde la perspectiva de negocio, las métricas más relevantes son:

- **Tasa de falsa alarma**: 0.1048 (De todas las reservas que no cancelan, ¿cuántas marcamos erróneamente?)
- **Tasa de pérdida**: 0.3804 (De todas las cancelaciones, ¿cuántas no detectamos?)

Estas métricas indican que:

- Solo el 10.48% de las reservas que realmente no cancelan son incorrectamente marcadas como cancelaciones, lo que significa que la estrategia de overbooking basada en estas predicciones sería razonablemente segura.
- Se pierden aproximadamente el 38.04% de las cancelaciones reales, lo que representa una oportunidad de mejora futura pero sigue siendo un avance significativo comparado con no tener modelo predictivo.

El reporte de clasificación detallado muestra el rendimiento por clase:

```plaintext
              precision    recall  f1-score   support
           0       0.92      0.90      0.91     38442
           1       0.56      0.62      0.59      8178
    accuracy                           0.85     46620
   macro avg       0.74      0.76      0.75     46620
weighted avg       0.85      0.85      0.85     46620
```

Este reporte confirma que el modelo logra un buen equilibrio entre las métricas para ambas clases, con una precisión y recall particularmente altos para la clase mayoritaria (no cancelaciones) y un rendimiento razonable para la clase minoritaria (cancelaciones).

### Análisis de la Distribución de Probabilidades y Umbral Óptimo

La búsqueda exhaustiva del umbral óptimo reveló un comportamiento interesante de las probabilidades predichas en relación con el F1-score:

```plaintext
------ Búsqueda de umbral óptimo ------
Umbral		F1		Precision	Recall
0.10		0.4342		0.2822		0.9412
0.12		0.4535		0.3005		0.9238
...
0.44		0.5867		0.5572		0.6196
...
0.58		0.5285		0.6206		0.4603
0.60		0.5132		0.6282		0.4337
...
0.80		0.2626		0.7286		0.1602
```

Observamos que:

- **Umbrales bajos (0.10-0.20)**: Ofrecen un recall extremadamente alto (>90%) pero con precisión baja (<35%), lo que significa que detectaríamos casi todas las cancelaciones pero con muchos falsos positivos.
- **Umbrales medios (0.40-0.50)**: Proporcionan el mejor equilibrio entre precisión y recall, con el F1-score máximo alrededor de 0.44.
- **Umbrales altos (>0.60)**: Ofrecen alta precisión (>62%) pero bajo recall (<43%), lo que significa que las predicciones positivas son muy confiables pero se pierden muchas cancelaciones reales.

La elección del umbral óptimo de 0.441 representa un punto de inflexión donde el modelo maximiza el F1-score, ofreciendo el mejor equilibrio entre detectar la mayor cantidad posible de cancelaciones reales sin generar demasiados falsos positivos.

Este umbral cercano al estándar de 0.5 sugiere que nuestro modelo está bien calibrado, a diferencia de la versión anterior que requería un umbral extremadamente alto (0.95) para obtener resultados aceptables, lo que indica una mejora significativa en la robustez y generalización del modelo.

## Conclusiones y Direcciones Futuras

La evolución de nuestro sistema, desde un modelo complejo de stacking hasta un XGBoost único pero bien optimizado, demuestra el valor de un enfoque iterativo y basado en evidencia. Este trabajo ha incorporado específicamente las lecciones y correcciones del feedback recibido en el primer intento, con énfasis en:

- **Correcta definición del target**: Hemos corregido la definición del objetivo para predecir cancelaciones con 30 días o menos de anticipación, alineando el modelo con el requerimiento de negocio real.
- **Generalización sobre complejidad**: La elección de un modelo más simple pero bien optimizado en lugar de un ensemble complejo ha mejorado la generalización y reducido el riesgo de overfitting.
- **Hiperparámetros enfocados en generalización**: La configuración de hiperparámetros como `min_child_weight=4`, `learning_rate=0.05` y `gamma=0.1` favorece la estabilidad y generalización del modelo.
- **Estrategia balanceada para clases desbalanceadas**: El uso de SMOTE con `sampling_strategy=0.6` proporciona suficiente reequilibrio sin introducir demasiados datos sintéticos que podrían confundir al modelo.
- **Mantenimiento de la validación estratificada por grupos**: La validación cruzada utilizando `StratifiedGroupKFold` sigue siendo fundamental para evaluar correctamente el rendimiento en nuevos hoteles.
- **Equilibrio en el umbral de decisión**: El umbral optimizado de 0.441 refleja un modelo bien calibrado que equilibra precisión y recall, a diferencia del umbral extremo (0.95) que requería la versión anterior.

Las métricas actuales muestran un modelo con un F1-score de 0.5867, un AUC-ROC de 0.8601, y un balance efectivo entre precisión (55.72%) y recall (61.96%). Estos resultados son particularmente valiosos en el contexto de la industria hotelera, donde tanto los falsos positivos como los falsos negativos tienen costos operativos asociados.

# Resultados del Script de Inferencia

## Output del Script local

```plaintext
Cargando datos de inferencia...
Datos preprocesados: 4121 registros, 19 características.
Cargando el modelo entrenado...
Modelo cargado correctamente.
Generando predicciones...
Predicciones completadas.
Predicciones guardadas en data/output_predictions.csv:
→ 655 cancelaciones previstas de 4121 reservas (15.9%)
```

## Análisis de los Resultados

### Resumen de Predicciones

El script de inferencia ha procesado un total de 4121 registros, cada uno con 19 características. Utilizando el modelo XGBoost entrenado, se han generado predicciones para estas reservas. Los resultados indican que se espera que 655 de estas reservas (el 15.9%) sean canceladas.

### Comparación con Datos Históricos

Para contextualizar estos resultados, es útil compararlos con los datos históricos de cancelaciones. En el conjunto de datos de entrenamiento (`bookings_train`), la tasa de cancelaciones anticipadas (con 30 días o menos de antelación) fue del 17.54%. La tasa de cancelaciones prevista en el conjunto de inferencia (15.9%) es ligeramente inferior, lo que podría indicar una mejora en la gestión de reservas o una variación natural en el comportamiento de los clientes.

### Análisis de Reservas "Booked"

Para un análisis más detallado, se puede examinar el subconjunto de reservas que actualmente están en estado "Booked" en el conjunto de datos de entrenamiento (`bookings_train`). Este análisis puede proporcionar información adicional sobre la precisión del modelo en reservas activas.

#### Pasos para el Análisis

1. **Filtrar Reservas "Booked"**: Extraer las reservas que están actualmente en estado "Booked" del conjunto de datos de entrenamiento.
2. **Aplicar el Modelo**: Generar predicciones para estas reservas utilizando el mismo modelo entrenado.
3. **Evaluar Resultados**: Comparar las predicciones con las tasas históricas de cancelación y analizar la distribución de las probabilidades predichas.

#### Resultados Esperados

- **Tasa de Cancelación Prevista**: Se espera que la tasa de cancelación prevista para las reservas "Booked" sea similar a la observada en el conjunto de inferencia (alrededor del 15.9%).
- **Distribución de Probabilidades**: La distribución de las probabilidades predichas puede proporcionar información sobre la confianza del modelo en sus predicciones. Un análisis detallado puede revelar si hay un subconjunto de reservas con alta probabilidad de cancelación que requiera atención especial.

### Conclusiones

El modelo XGBoost optimizado ha demostrado ser efectivo para predecir cancelaciones anticipadas en el conjunto de inferencia, con una tasa de cancelación prevista del 15.9%. Este resultado es coherente con las tasas históricas observadas y sugiere que el modelo está bien calibrado. Un análisis adicional de las reservas "Booked" en el conjunto de entrenamiento puede proporcionar información valiosa para la gestión proactiva de reservas y la implementación de estrategias de overbooking controlado.

# Verificación de Docker

```bash
Windows PowerShell
Copyright (C) Microsoft Corporation. Todos los derechos reservados.

Prueba la nueva tecnología PowerShell multiplataforma https://aka.ms/pscore6

PS C:\Users\Administrador\DataspellProjects\Aprendizaje_automatico2> docker build -t hotel-predictor -f ensembled/Dockerfile .
[+] Building 46.6s (13/13) FINISHED                                                                                                                                                                            docker:desktop-linux
 => [internal] load build definition from Dockerfile                                                                                                                                                                           0.0s
 => => transferring dockerfile: 1.21kB                                                                                                                                                                                         0.0s
 => [internal] load metadata for docker.io/library/python:3.11-slim                                                                                                                                                            1.2s
 => [auth] library/python:pull token for registry-1.docker.io                                                                                                                                                                  0.0s
 => [internal] load .dockerignore                                                                                                                                                                                              0.0s
 => => transferring context: 2B                                                                                                                                                                                                0.0s
 => [1/7] FROM docker.io/library/python:3.11-slim@sha256:75a17dd6f00b277975715fc094c4a1570d512708de6bb4c5dc130814813ebfe4                                                                                                      0.0s
 => => resolve docker.io/library/python:3.11-slim@sha256:75a17dd6f00b277975715fc094c4a1570d512708de6bb4c5dc130814813ebfe4                                                                                                      0.0s
 => [internal] load build context                                                                                                                                                                                              0.0s
 => => transferring context: 23.09kB                                                                                                                                                                                           0.0s 
 => CACHED [3/7] RUN mkdir -p /app/data /app/models                                                                                                                                                                            0.0s 
 => CACHED [4/7] COPY ensembled/requirements.txt /app/                                                                                                                                                                         0.0s 
 => [5/7] COPY data/*.csv /app/data/                                                                                                                                                                                           0.0s 
 => [6/7] COPY ensembled/*.py /app/                                                                                                                                                                                            0.0s 
 => [7/7] RUN pip install --no-cache-dir -r requirements.txt                                                                                                                                                                  42.4s 
 => exporting to image                                                                                                                                                                                                         2.9s 
 => => exporting layers                                                                                                                                                                                                        2.9s 
 => => writing image sha256:ac82f7c1c9c8bb944a94c0ff27f8af8babd06509da0f865cb68fa3938b0a500e                                                                                                                                   0.0s 
 => => naming to docker.io/library/hotel-predictor                                                                                                                                                                             0.0s 

View build details: docker-desktop://dashboard/build/desktop-linux/desktop-linux/t79faos3ugf71pehisv83fwo4

What's next:
    View a summary of image vulnerabilities and recommendations → docker scout quickview
PS C:\Users\Administrador\DataspellProjects\Aprendizaje_automatico2> docker run -v "$(pwd)/models:/app/models" hotel-predictor
Iniciando entrenamiento...

--- Distribución Original del Target (y) ---
target
0    38442
1     8178
Name: count, dtype: int64
Porcentaje de clase positiva (1 = Cancelación <= 30 días): 17.54%
Total de registros: 46620
-------------------------------------------


Realizando validación cruzada con 7 splits (estratificada por hotel_id)...

------ Búsqueda de umbral óptimo ------
Umbral          F1              Precision       Recall
0.10            0.4403          0.2873          0.9414
0.12            0.4558          0.3028          0.9219
0.14            0.4718          0.3190          0.9055
0.15            0.4871          0.3354          0.8895
0.17            0.5003          0.3505          0.8733
0.19            0.5158          0.3686          0.8590
0.21            0.5267          0.3853          0.8321
0.23            0.5385          0.4029          0.8117
0.24            0.5446          0.4151          0.7915
0.26            0.5520          0.4282          0.7764
0.28            0.5608          0.4436          0.7624
0.30            0.5666          0.4575          0.7439
0.32            0.5723          0.4709          0.7294
0.33            0.5766          0.4856          0.7096
0.35            0.5805          0.5011          0.6898
0.37            0.5825          0.5146          0.6709
0.39            0.5843          0.5272          0.6553
0.41            0.5839          0.5396          0.6360
0.42            0.5856          0.5521          0.6234
0.44            0.5846          0.5603          0.6112
0.46            0.5824          0.5699          0.5954
0.48            0.5803          0.5804          0.5802
0.49            0.5762          0.5898          0.5632
0.51            0.5696          0.5964          0.5451
0.53            0.5608          0.6020          0.5249
0.55            0.5526          0.6103          0.5049
0.57            0.5427          0.6188          0.4832
0.58            0.5342          0.6296          0.4639
0.60            0.5183          0.6366          0.4371
0.62            0.4989          0.6437          0.4073
0.64            0.4792          0.6480          0.3802
0.66            0.4584          0.6512          0.3536
0.67            0.4340          0.6518          0.3253
0.69            0.4058          0.6538          0.2942
0.71            0.3845          0.6608          0.2711
0.73            0.3644          0.6712          0.2501
0.75            0.3389          0.6896          0.2246
0.76            0.3065          0.6922          0.1969
0.78            0.2780          0.7071          0.1730
0.80            0.2434          0.7278          0.1461

Umbral óptimo seleccionado: 0.4231 (F1: 0.5856)

------ Evaluación del Modelo ------
Umbral aplicado: 0.4231

Métricas principales:
→ Precision:   0.5521 (De las predicciones positivas, ¿cuántas son correctas?)
→ Recall:      0.6234 (De todos los positivos reales, ¿cuántos detectamos?)
→ F1-Score:    0.5856 (Media armónica entre precision y recall)
→ AUC-ROC:     0.8612 (Capacidad discriminativa general del modelo)

Métricas adicionales:
→ Accuracy:    0.8452 (Porcentaje general de aciertos)
→ Specificity: 0.8924 (De los negativos reales, ¿cuántos detectamos?)

Matriz de confusión:
→ Verdaderos Negativos (TN): 34306 (No cancela y predecimos que no cancela)
→ Falsos Positivos (FP):    4136 (No cancela pero predecimos cancelación)
→ Falsos Negativos (FN):    3080 (Cancela pero no lo detectamos)
→ Verdaderos Positivos (TP): 5098 (Cancela y predecimos correctamente)

Métricas de negocio:
→ Tasa de falsa alarma:  0.1076 (De todas las reservas que no cancelan, ¿cuántas marcamos erróneamente?)
→ Tasa de pérdida:       0.3766 (De todas las cancelaciones, ¿cuántas no detectamos?)

              precision    recall  f1-score   support

           0       0.92      0.89      0.90     38442
           1       0.55      0.62      0.59      8178

    accuracy                           0.85     46620
   macro avg       0.73      0.76      0.75     46620
weighted avg       0.85      0.85      0.85     46620


Entrenando modelo final con todos los datos...
Modelo guardado en /app/models/pipeline.cloudpkl

Entrenamiento completado. ¡El modelo está listo para inferencia!
PS C:\Users\Administrador\DataspellProjects\Aprendizaje_automatico2> docker run -e SCRIPT_TO_RUN=inference -v "$(pwd)/models:/app/models" -v "$(pwd)/data:/app/data" hotel-predictor
Cargando datos de inferencia...
Datos preprocesados: 4121 registros, 19 características.
Cargando el modelo entrenado...
Modelo cargado correctamente.
Generando predicciones...
Predicciones completadas.
Predicciones guardadas en /app/data/output_predictions.csv:
→ 696 cancelaciones previstas de 4121 reservas (16.9%)
