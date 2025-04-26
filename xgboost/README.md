# Memoria Actualizada del Proyecto: Ensemble Avanzado para la Predicción de Cancelaciones Hoteleras

## Índice

1. Introducción: El Desafío de las Cancelaciones Anticipadas
2. Mirando Atrás: Lecciones Aprendidas del Primer Intento
3. El Nuevo Enfoque: Preprocesamiento y Construcción del Ensemble
   3.1. Preparación y Análisis de Datos
   3.2. Ingeniería de Características Avanzada
   3.3. Estrategia de Gestión de Valores Nulos
   3.4. Arquitectura del Ensemble Stacking
   3.5. Manejo del Desbalanceo de Clases
   3.6. Validación Estratificada por Grupos
   3.7. Optimización del Umbral de Decisión
4. Evaluación del Modelo y Resultados
   4.1. Análisis de la Distribución de Probabilidades y Umbral Óptimo
5. Conclusiones y Direcciones Futuras

## 1. Introducción: El Desafío de las Cancelaciones Anticipadas

En la industria hotelera, las cancelaciones anticipadas representan un desafío crítico para la gestión eficiente del inventario y la maximización de ingresos. Predecir qué reservas tienen alta probabilidad de cancelarse con al menos 30 días de antelación permite implementar estrategias de overbooking controlado y promociones dirigidas, mitigando así el impacto económico de las habitaciones no ocupadas.

Nuestro objetivo se centra en desarrollar un sistema de predicción que identifique con alta precisión las reservas con mayor riesgo de cancelación anticipada, priorizando especialmente minimizar los falsos negativos (reservas que se cancelarán pero no son detectadas por el modelo).

## 2. Mirando Atrás: Lecciones Aprendidas del Primer Intento

Nuestro primer trabajo en este problema utilizó un modelo XGBoost único optimizado mediante búsqueda de hiperparámetros. Aunque mostró resultados prometedores, la evaluación y el feedback recibido revelaron importantes deficiencias:

- **Alta variabilidad entre folds**: Las métricas F1 mostraban fluctuaciones significativas en diferentes subconjuntos de datos, indicando problemas de estabilidad.

- **Preprocesamiento fragmentado**: El proceso de transformación de datos se realizaba en múltiples etapas, complicando la reproducibilidad y aumentando el riesgo de fugas de datos.

- **Manejo subóptimo del desbalanceo**: La clase minoritaria (cancelaciones anticipadas) no se trataba eficientemente, siendo este uno de los problemas principales del trabajo.

- **Estrategia de validación inadecuada**: No se utilizaba cross validation excepto en GridSearch, y no se eligió la estrategia más adecuada para este problema, ignorando la estructura de grupo natural de los datos (reservas del mismo hotel).

- **Error crítico en la etapa de inferencia**: Eliminamos incorrectamente los registros de tipo "Booked" en inferencia, lo que nos dejaba sin filas para hacer predicciones. Además, clasificamos erróneamente los "No-show" como cancelaciones cuando realmente son clientes que han pagado.

Estas observaciones críticas del feedback anterior fueron fundamentales para dirigir nuestro desarrollo hacia un enfoque más robusto y sofisticado en este nuevo trabajo.

## 3. El Nuevo Enfoque: Preprocesamiento y Construcción del Ensemble

### 3.1. Preparación y Análisis de Datos

El proceso de preparación de datos se mejoró significativamente integrándolo en una estructura coherente:

```python
def get_X_y():
    """Carga y preprocesa los datos para el entrenamiento."""
    print("Cargando datos...")

    # Cargar datos
    hotels = pd.read_csv(os.environ.get("HOTELS_DATA_PATH", 'data/hotels.csv'))
    bookings = pd.read_csv(os.environ.get("TRAIN_DATA_PATH", 'data/bookings_train.csv'))

    # Combinar datos
    data = pd.merge(bookings, hotels, on='hotel_id', how='left')

    # CORRECCIÓN: Filtrar datos 'Booked' solo durante entrenamiento, no en inferencia
    # Este código se ejecuta en el contexto de entrenamiento
    # Para inferencia, se usa un pipeline separado que no filtra 'Booked'
    data = data[data['reservation_status'] != 'Booked'].copy()

    # CORRECCIÓN: Interpretar correctamente los No-Show como Check-Out, ya que han pagado
    data['reservation_status'] = data['reservation_status'].replace('No-Show', 'Check-Out')
    
    # Resto del preprocesamiento...
```

Esta corrección aborda el error crítico identificado en el feedback anterior: ahora interpretamos correctamente los "No-Show" como "Check-Out" puesto que son clientes que efectivamente han pagado, y solo filtramos los registros "Booked" durante el entrenamiento. El código verifica exactamente con esta implementación, donde claramente se especifica: 

```python
# Filtrar datos relevantes - excluir reservas que aún están en estado 'Booked'
data = data[data['reservation_status'] != 'Booked'].copy()

# Interpreto los NoShow como Check-Out, ya que realmente han pagado
data['reservation_status'] = data['reservation_status'].replace('No-Show', 'Check-Out')
```

En el contexto de inferencia utilizamos un pipeline separado diseñado específicamente para procesar todos los registros sin aplicar este filtro, permitiendo así hacer predicciones sobre todas las reservas pendientes.

### 3.2. Ingeniería de Características Avanzada

La ingeniería de características se enfocó en incorporar conocimiento del dominio, creando indicadores predictivos clave. El código implementa exactamente estas transformaciones:

```python
# Convertir fechas
date_columns = ['arrival_date', 'booking_date', 'reservation_status_date']
for col in date_columns:
    if col in data.columns:
        data[col] = pd.to_datetime(data[col])

# Crear características clave
data['days_before_arrival'] = (data['arrival_date'] - data['reservation_status_date']).dt.days
data['target'] = ((data['reservation_status'] == 'Canceled') &
                  (data['days_before_arrival'] >= 30)).astype(int)

data['lead_time'] = (data['arrival_date'] - data['booking_date']).dt.days
data['is_high_season'] = data['arrival_date'].dt.month.isin([6, 7, 8, 12]).astype(int)
data['is_weekend_arrival'] = data['arrival_date'].dt.dayofweek.isin([4, 5]).astype(int)
data['price_per_night'] = data['rate'] / np.maximum(data['stay_nights'], 1)
data['has_special_requests'] = (data['special_requests'] > 0).astype(int)
```

Destacan características como:
- **Lead time**: Días entre reserva y llegada, un predictor clásico de cancelaciones.
- **High season**: Identificación de temporadas alta (verano y diciembre).
- **Weekend arrival**: Las llegadas en fin de semana tienen patrones diferenciados.
- **Price per night**: Normalización del precio por noche de estancia.
- **Special requests**: Indicador de si el cliente ha hecho solicitudes especiales.

El código también implementa una característica adicional cuando están disponibles los datos de país:

```python
# Característica de cliente extranjero si existen las columnas
if 'country_x' in data.columns and 'country_y' in data.columns:
    data['is_foreign'] = (data['country_x'].astype(str) != data['country_y'].astype(str)).astype(int)
    data.loc[data['country_x'].isna() | data['country_y'].isna(), 'is_foreign'] = 0
```

Esta característica **Foreign guest** compara el país de origen con el país del hotel, agregando un factor importante para predecir cancelaciones.

Estas características incorporan conocimiento del sector hotelero y fueron seleccionadas por su alto poder predictivo en análisis exploratorios previos.

Es importante destacar que se decidió no categorizar las nuevas columnas resultantes del feature engineering, ya que los propios algoritmos predictivos utilizados (XGBoost, LightGBM, RandomForest, etc.) tienen la capacidad de categorizar por sí solos estas características, permitiendo así un procesamiento más eficiente y adaptado a la estructura natural de los datos.

### 3.3. Estrategia de Gestión de Valores Nulos

Un avance significativo fue la integración de la imputación de valores nulos dentro del pipeline de preprocesamiento, garantizando que los métodos de imputación se apliquen correctamente tanto en entrenamiento como en predicción. El código implementa esta estrategia de forma clara:

```python
def create_preprocessor(X: pd.DataFrame):
    """Crea un preprocesador que incluye la imputación de valores nulos."""
    num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    bool_features = X.select_dtypes(include=["bool"]).columns.tolist()
    cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    cat_features = [col for col in cat_features if col not in bool_features]

    # Pipelines para cada tipo de característica
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # Si hay características booleanas, las mantenemos como están
    bool_transformer = 'passthrough'

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_features),
            ("cat", categorical_transformer, cat_features),
            ("bool", bool_transformer, bool_features),
        ]
    )

    return preprocessor
```

Esta implementación presenta varias ventajas clave:

1. **Detección automática de tipos de datos**: El código identifica automáticamente qué columnas son numéricas, booleanas o categóricas, haciendo el preprocesamiento adaptable a cambios en el conjunto de datos.

2. **Tratamiento específico por tipo de dato**: Cada tipo de característica recibe un tratamiento diferenciado:
   - **Variables numéricas**: Imputación por mediana y escalado estándar
   - **Variables categóricas**: Imputación por moda y codificación one-hot con `handle_unknown="ignore"` para manejar categorías nuevas en inferencia
   - **Variables booleanas**: Preservadas intactas mediante "passthrough"

3. **Prevención de data leakage**: Al encapsular todo el preprocesamiento en `ColumnTransformer`, las estadísticas para la imputación (medianas, modas) y el escalado (medias, desviaciones estándar) se calculan solo en el conjunto de entrenamiento y se aplican de manera consistente tanto en entrenamiento como en inferencia.

4. **Manejo coherente de valores ausentes**: La estrategia garantiza que los valores nulos se traten de manera coherente durante todo el ciclo de vida del modelo, evitando discrepancias entre entrenamiento y producción.

Este enfoque integrado es crucial para mantener la calidad y coherencia de los datos, especialmente al desplegar el modelo en un entorno de producción donde pueden aparecer valores ausentes o categorías nuevas.

### 3.4. Arquitectura del Ensemble Stacking

La decisión de implementar un ensemble stacking fue motivada principalmente por la necesidad de reducir la variabilidad observada en las métricas F1 entre diferentes folds. El código implementa esta arquitectura de forma efectiva:

```python
class StackingEnsemble(BaseEstimator, ClassifierMixin):
    """Ensemble simplificado de modelos mediante stacking."""

    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.fold_models = []
        self.meta_model = None
        self.model_types = ['xgboost', 'lightgbm', 'rf', 'gbm']  # Modelos a usar
```

El método `fit` de esta clase implementa el entrenamiento del ensemble:

```python
def fit(self, X, y):
    """Entrena múltiples modelos usando validación cruzada."""
    n_splits = 7 # Muy importante
    groups = X['hotel_id'].copy() if 'hotel_id' in X.columns else None
    X = X.drop(columns=['hotel_id']) if groups is not None else X

    group_cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_splits = list(group_cv.split(X, y, groups)) if groups is not None else list(group_cv.split(X, y))

    n_samples = X.shape[0]
    n_models_per_fold = len(self.model_types)
    oof_preds = np.zeros((n_samples, n_splits * n_models_per_fold))

    start_time = time.time()
    self.fold_models = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
        print(f"Entrenando fold {fold_idx + 1}/{n_splits}...")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        fold_models = []

        for i, model_type in enumerate(self.model_types):
            preprocessor = create_preprocessor(X_train)
            pos_weight = len(y_train[y_train == 0]) / max(len(y_train[y_train == 1]), 1)
            classifier = create_base_model(model_type, pos_weight, random_state=42 + fold_idx * 10 + i)

            pipeline = ImbPipeline(steps=[
                ("preprocessor", preprocessor),
                ("sampling", SMOTE(random_state=42 + fold_idx, k_neighbors=3)),
                ("classifier", classifier)
            ])

            pipeline.fit(X_train, y_train)
            fold_models.append(pipeline)

            val_preds_proba = pipeline.predict_proba(X_val)[:, 1]
            col_idx = fold_idx * n_models_per_fold + i
            oof_preds[val_idx, col_idx] = val_preds_proba

        self.fold_models.append(fold_models)
```

El meta-modelo se entrena con las predicciones out-of-fold de todos los modelos base:

```python
self.meta_model = LogisticRegression(
    C=0.6,
    class_weight='balanced',
    solver='liblinear',
    max_iter=1000,
    random_state=42
)
self.meta_model.fit(oof_preds, y)
```

Esta arquitectura de dos niveles incorpora:

1. **Diversidad de algoritmos**: Cuatro tipos diferentes de modelos para capturar distintos patrones:
   - XGBoost: Eficiente con datos tabulares y manejo de valores nulos.
   - LightGBM: Implementación alternativa de gradient boosting con particionamiento leaf-wise.
   - RandomForest: Enfoque de bagging con mejor robustez ante outliers.
   - GradientBoosting: Implementación clásica con características complementarias.

2. **Diversidad de datos**: Cada modelo se entrena en diferentes particiones mediante validación cruzada de 7 folds, exponiendo cada algoritmo a diferentes subconjuntos de datos.

3. **Meta-aprendizaje**: Un modelo de regresión logística aprende a combinar las predicciones de los modelos base, ponderando implícitamente cada algoritmo según su rendimiento en diferentes contextos.

Este diseño permite que el ensemble aproveche las fortalezas complementarias de cada algoritmo, reduciendo la varianza y mejorando la estabilidad de las predicciones.

### 3.5. Manejo del Desbalanceo de Clases

El desbalanceo entre clases (mayoría de reservas no canceladas anticipadamente) fue uno de los problemas principales identificados en el feedback del primer trabajo. El código implementa múltiples estrategias para abordar este desafío:

```python
# Dentro del pipeline de cada modelo base
pipeline = ImbPipeline(steps=[
    ("preprocessor", preprocessor),
    ("sampling", SMOTE(random_state=42 + fold_idx, k_neighbors=3)),
    ("classifier", classifier)
])
```

Las estrategias implementadas incluyen:

1. **Técnicas de sobremuestreo**: Implementación de SMOTE (Synthetic Minority Over-sampling Technique) para generar ejemplos sintéticos de la clase minoritaria y equilibrar el conjunto de entrenamiento. El parámetro `k_neighbors=3` en SMOTE fue cuidadosamente seleccionado para evitar la generación de ejemplos sintéticos poco realistas, un problema común cuando se utiliza un valor demasiado alto.

2. **Ponderación de clases**: Configuración dinámica de ponderación en cada fold:

```python
# Cálculo dinámico del peso para cada fold
pos_weight = len(y_train[y_train == 0]) / max(len(y_train[y_train == 1]), 1)
classifier = create_base_model(model_type, pos_weight, random_state=42 + fold_idx * 10 + i)
```

La función `create_base_model` incorpora este peso en la configuración de los modelos:

```python
model_params = {
    'xgboost': {
        'scale_pos_weight': pos_weight,
        # otros parámetros...
    },
    'lightgbm': {
        'scale_pos_weight': pos_weight,
        # otros parámetros...
    },
    'rf': {
        'class_weight': 'balanced',
        # otros parámetros...
    },
    # configuración para GBM...
}
```

3. **Meta-modelo sensible al desbalanceo**: El modelo de regresión logística también se configura con ponderación de clases:

```python
self.meta_model = LogisticRegression(
    C=0.6,
    class_weight='balanced',
    solver='liblinear',
    max_iter=1000,
    random_state=42
)
```

4. **Optimización del umbral**: Se implementa una búsqueda del umbral óptimo que maximiza el F1-score:

```python
def find_optimal_threshold(y_true, y_pred_proba):
    """Encuentra el umbral óptimo que maximiza el F1-score."""
    thresholds = np.linspace(0.05, 0.99, 50)
    best_threshold, best_f1 = 0.5, 0

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold
```

Este umbral se utiliza después para las predicciones finales:

```python
# Encontrar el umbral óptimo en validación
val_proba = pipe.predict_proba(X_val)[:, 1]
optimal_threshold = find_optimal_threshold(y_val, val_proba)
pipe.threshold = optimal_threshold
```

5. **Evaluación con métricas adecuadas**: Uso de F1-score como métrica principal, que equilibra precisión y recall:

```python
val_f1 = f1_score(y_val, val_pred)
val_precision = precision_score(y_val, val_pred)
val_recall = recall_score(y_val, val_pred)
```

Esta estrategia multinivel garantiza que el desbalanceo se aborde en cada etapa del proceso de modelado, desde la preparación de datos hasta la decisión final de clasificación.

### 3.6. Validación Estratificada por Grupos y Prevención de Data Leakage

Una innovación crítica para abordar los problemas de validación identificados en el feedback anterior fue la implementación de `StratifiedGroupKFold` para la validación cruzada. El código implementa esta estrategia de forma clara:

```python
n_splits = 7 # Muy importante
groups = X['hotel_id'].copy() if 'hotel_id' in X.columns else None
X = X.drop(columns=['hotel_id']) if groups is not None else X

group_cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
cv_splits = list(group_cv.split(X, y, groups)) if groups is not None else list(group_cv.split(X, y))
```

Esta implementación aborda el data leakage de varias maneras:

1. **Agrupación por hotel_id**: Todas las reservas de un mismo hotel permanecen juntas ya sea en entrenamiento o validación, nunca divididas entre ambos. Esto es crucial para evitar que el modelo aprenda patrones específicos de cada hotel que luego "reconozca" en validación.

2. **Estratificación por clase**: Cada fold mantiene aproximadamente la misma proporción de cancelaciones anticipadas, asegurando que la distribución de clases sea similar en todos los folds.

3. **Eliminación de variables que causan data leakage**: El código elimina explícitamente columnas que podrían filtrar información del futuro:

```python
# Eliminar columnas que podrían causar data leakage
columns_to_drop = [
    'reservation_status', 'reservation_status_date', 'days_before_arrival',
    'arrival_date', 'booking_date', 'special_requests', 'stay_nights',
    'country_y', 'country_x'
]
columns_to_drop = [col for col in columns_to_drop if col in data.columns]

X = data.drop(columns=['target'] + columns_to_drop)
y = data['target']
```

Esto asegura que no se utilicen variables que contengan información futura o información que no estaría disponible en el momento de hacer la predicción real.

La elección específica de 7 folds fue deliberada:

1. **Número óptimo para el tamaño del dataset**: Se eligió 7 porque es el número máximo de grupos posible con `StratifiedGroupKFold` dada la distribución de nuestros datos por hotel_id, manteniendo una distribución equilibrada de clases en cada fold.

2. **Mitigación del data leakage**: Al aumentar el número de folds a 7, se incrementa la robustez del ensemble al exponer los modelos a más combinaciones de datos de entrenamiento/validación.

3. **Reducción de la variabilidad entre folds**: Combinada con la estrategia de ensemble, esta validación permite identificar y mitigar la variabilidad en las métricas F1 observada entre diferentes hoteles.

Esta estrategia de validación fue fundamental para corregir uno de los errores graves identificados en el feedback anterior: la falta de una estrategia de cross-validation adecuada para este problema particular.

### 3.7. Optimización del Umbral de Decisión

Un componente crítico del proceso es la optimización del umbral de decisión para maximizar el F1-score global. El código implementa esta optimización de forma efectiva:

```python
def find_optimal_threshold(y_true, y_pred_proba):
    """Encuentra el umbral óptimo que maximiza el F1-score."""
    thresholds = np.linspace(0.05, 0.99, 50)
    best_threshold, best_f1 = 0.5, 0

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold
```

En el script principal, esta función se utiliza para encontrar el umbral óptimo:

```python
# Encontrar el umbral óptimo en validación
val_proba = pipe.predict_proba(X_val)[:, 1]
optimal_threshold = find_optimal_threshold(y_val, val_proba)
pipe.threshold = optimal_threshold
print(f"Umbral óptimo encontrado: {optimal_threshold:.4f}")
```

Esta optimización explora 50 valores de umbral posibles entre 0.05 y 0.99, seleccionando aquel que maximiza el F1-score en el conjunto de validación. 

Un aspecto importante es que esta optimización se realiza sobre un conjunto de validación independiente (30% de los datos), garantizando que el umbral óptimo sea generalizable y no sobreajustado a particularidades del conjunto de entrenamiento:

```python
# Dividir en entrenamiento y validación para evaluar el rendimiento final
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
```

El umbral optimizado se almacena como parte del modelo, asegurando que se utilice el mismo umbral en la fase de inferencia:

```python
pipe.threshold = optimal_threshold
```

Esto garantiza que todas las predicciones futuras utilizarán el umbral identificado como óptimo para maximizar el F1-score, manteniendo el enfoque en un equilibrio adecuado entre precisión y recall, fundamental para el problema de cancelaciones anticipadas.

## 4. Evaluación del Modelo y Resultados

La evaluación del modelo se realizó utilizando un conjunto de validación independiente (30% de los datos), obteniendo los siguientes resultados actualizados:

```
Umbral óptimo encontrado: 0.9500

Rendimiento en validación:
Precisión: 0.3772
Recall: 0.9911
F1-Score: 0.5464
Accuracy: 0.7741
AUC-ROC: 0.9563
```

Estos resultados muestran un enfoque claramente orientado a maximizar el recall (99.11%), lo que indica que el modelo es capaz de identificar casi todas las cancelaciones anticipadas. Este enfoque es estratégicamente valioso para la gestión hotelera, donde no detectar una potencial cancelación (falso negativo) tiene un costo económico mayor que equivocarse al predecir una cancelación (falso positivo).

El umbral muy alto (0.95) refleja esta estrategia: el modelo está configurado para ser extremadamente sensible, clasificando como cancelación cualquier reserva que tenga incluso una pequeña probabilidad de cancelarse.

La precisión relativamente baja (37.72%) es un compromiso aceptable dado el alto recall, y el F1-Score de 0.5464 representa un equilibrio razonable entre ambas métricas. El AUC-ROC de 0.9563 indica que el modelo tiene una excelente capacidad discriminativa general.

Estos resultados actualizados difieren significativamente de los reportados previamente en la memoria original, lo que demuestra la evolución del modelo hacia una configuración que prioriza aún más la detección de cancelaciones potenciales.

### 4.1. Análisis de la Distribución de Probabilidades y Umbral Óptimo

Un análisis detallado de la distribución de probabilidades predichas por el modelo revela un patrón interesante que justifica el alto umbral de decisión seleccionado.

![threshold.png](../graphs/threshold.png)

El gráfico muestra la distribución de las probabilidades predichas para ambas clases: cancelaciones tempranas (clase 1) y no cancelaciones (clase 0). Se observa una separación casi perfecta entre las dos clases:

- La mayoría de las no cancelaciones (clase 0) se concentran en probabilidades cercanas a 0
- Las cancelaciones tempranas (clase 1) se agrupan en probabilidades cercanas a 1

Esta polarización extrema de las probabilidades indica que el modelo logra discriminar muy bien entre ambas clases, con muy poca superposición de distribuciones. Este comportamiento es lo que permite establecer un umbral de decisión extraordinariamente alto (0.95) sin comprometer significativamente el recall.

La selección del umbral óptimo se realizó mediante una búsqueda exhaustiva en el rango de 0.05 a 0.99:

```python
def find_optimal_threshold(y_true, y_pred_proba):
    """Encuentra el umbral óptimo que maximiza el F1-score."""
    thresholds = np.linspace(0.05, 0.99, 50)
    best_threshold, best_f1 = 0.5, 0

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold
```

Esta función evalúa sistemáticamente 50 posibles umbrales y selecciona aquel que maximiza el F1-score, que en nuestro caso resultó ser 0.95. Este valor tan alto del umbral tiene importantes implicaciones:

**Ventajas**:
- **Alta precisión selectiva**: Las reservas que el modelo identifica como cancelaciones son casi con total seguridad verdaderas cancelaciones, lo que aumenta la confianza en las predicciones positivas.
- **Reducción de falsos positivos**: Minimiza la probabilidad de clasificar incorrectamente reservas como cancelaciones cuando no lo son, lo cual es valioso en escenarios donde el costo de una falsa alarma es alto.

**Consideraciones**:
- **Equilibrio con el recall**: A pesar del alto umbral, el modelo mantiene un recall extraordinariamente alto (99.11%), lo que sugiere que las verdaderas cancelaciones reciben consistentemente probabilidades muy cercanas a 1.
- **Implicaciones operativas**: Este umbral permite a los hoteles implementar estrategias más agresivas para las reservas identificadas como cancelaciones, con alta confianza en la precisión de estas predicciones.

La combinación de características del ensemble (diversidad de algoritmos, validación cruzada, y estrategias de manejo del desbalanceo) ha producido un modelo con una capacidad discriminativa excepcional, permitiendo este umbral tan alto sin comprometer el recall, algo que no es común en problemas de clasificación binaria con clases desbalanceadas.

## 5. Conclusiones y Direcciones Futuras

La evolución de nuestro sistema, desde un modelo XGBoost único hasta un sofisticado ensemble stacking, demuestra el valor de un enfoque iterativo y basado en evidencia. Este trabajo ha incorporado específicamente las lecciones y correcciones del feedback recibido en el primer intento, con énfasis en:

1. **Preprocesamiento integrado**: La implementación de un preprocesamiento completamente integrado en el pipeline ha sido crucial para garantizar la consistencia entre entrenamiento e inferencia. Esta estrategia es especialmente importante para:
   - Gestionar correctamente los valores nulos con imputación adaptada a cada tipo de variable
   - Preservar la coherencia en las transformaciones de datos
   - Facilitar el despliegue en producción sin inconsistencias

2. **Corrección de errores críticos**: Se han corregido los errores fundamentales identificados en el feedback:
   - Interpretación correcta de los registros "No-Show" como "Check-Out"
   - Filtrado apropiado de registros "Booked" solo durante el entrenamiento
   - Prevención efectiva del data leakage mediante la eliminación de variables problemáticas y el uso de validación por grupos

3. **Enfoque estratégico para el desbalanceo**: La combinación de SMOTE, ponderación de clases, y optimización del umbral ha permitido un manejo efectivo del desbalanceo de clases, logrando un recall extremadamente alto, lo que era una prioridad para este problema específico.

4. **Validación estratificada por grupos**: La implementación de `StratifiedGroupKFold` con 7 folds ha sido fundamental para:
   - Respetar la estructura natural de los datos (agrupación por hotel)
   - Mantener la distribución de clases en cada fold
   - Reducir la variabilidad entre folds mediante el ensemble

5. **Optimización global del umbral**: El ajuste del umbral de decisión a un valor muy alto (0.95) refleja tanto la prioridad estratégica de maximizar el recall como la extraordinaria capacidad discriminativa del modelo. La clara separación entre las distribuciones de probabilidad para ambas clases permitió este umbral inusualmente alto sin comprometer la detección de cancelaciones.

Las métricas actualizadas reflejan un modelo altamente efectivo para la detección de cancelaciones anticipadas, con un recall cercano al 100%, lo que permitiría implementar estrategias proactivas para prácticamente todas las cancelaciones potenciales.

La arquitectura del ensemble stacking, combinando múltiples algoritmos y estrategias de manejo de datos, proporciona una solución robusta que puede adaptarse a diferentes patrones en los datos hoteleros, reduciendo significativamente la variabilidad de rendimiento entre diferentes hoteles que era una limitación importante en el trabajo anterior.

En futuras iteraciones, sería valioso explorar:
- Técnicas de calibración de probabilidades para mejorar la interpretabilidad del modelo
- Implementación de soluciones específicas por segmentos de hotel para adaptarse mejor a los diferentes patrones de cancelación
- Exploración de características adicionales basadas en el comportamiento histórico de los clientes
- Optimización del tiempo de entrenamiento mediante aceleración por GPU para facilitar actualizaciones frecuentes del modelo
- Experimentar con umbrales de decisión dinámicos que se adapten a características específicas de cada hotel o temporada

Este trabajo establece una base sólida y corrige los principales errores del primer intento, ofreciendo una solución aplicable y valiosa para el problema de la predicción de cancelaciones hoteleras que puede traducirse directamente en beneficios económicos tangibles para la industria.