# Proyecto de Predicción de Cancelaciones Hoteleras con XGBoost-GPU

## Índice
- [Descripción General](#descripción-general)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Configuración del Entorno](#configuración-del-entorno)
- [Arquitectura de la Solución](#arquitectura-de-la-solución)
- [Flujo de Trabajo](#flujo-de-trabajo)
   - [Carga y Fusión de Datos](#carga-y-fusión-de-datos)
   - [Preprocesamiento e Ingeniería de Características](#preprocesamiento-e-ingeniería-de-características)
   - [Preparación de Características para Modelado](#preparación-de-características-para-modelado)
   - [Manejo del Desbalanceo de Clases](#manejo-del-desbalanceo-de-clases)
   - [Filtrado de Características](#filtrado-de-características)
   - [Optimización de Hiperparámetros con GPU](#optimización-de-hiperparámetros-con-gpu)
   - [Evaluación y Selección del Umbral Óptimo](#evaluación-y-selección-del-umbral-óptimo)
- [Pipeline Personalizado](#pipeline-personalizado)
- [Resultados Obtenidos](#resultados-obtenidos)
- [Requisitos del Sistema](#requisitos-del-sistema)
- [Ejecución del Proyecto](#ejecución-del-proyecto)
- [Conclusiones](#conclusiones)

## Descripción General

Este proyecto implementa un modelo de machine learning para predecir cancelaciones de reservas hoteleras que ocurren con 30 días o más de antelación a la fecha de llegada. Utiliza XGBoost con aceleración GPU para optimizar el rendimiento de entrenamiento y permite escalar automáticamente a CPU cuando no hay GPUs disponibles.

El problema abordado es relevante para la industria hotelera, ya que identificar con anticipación las posibles cancelaciones permite implementar estrategias de mitigación, optimizar la capacidad y maximizar los ingresos.

## Estructura del Proyecto

```
xgboost/
├── Dockerfile         # Configuración del contenedor Docker con soporte CUDA
├── requirements.txt   # Dependencias del proyecto
├── bost_train.py      # Script principal de entrenamiento del modelo
├── README.md          # Este archivo
data/
├── hotels.csv         # Datos de hoteles
├── bookings_train.csv # Datos de reservas para entrenamiento
models/
└── xgboost_model.pkl  # Modelo entrenado serializado
```

## Configuración del Entorno

El proyecto está configurado para ejecutarse en un contenedor Docker con soporte para GPU NVIDIA mediante CUDA. Esto facilita la replicabilidad y escalabilidad del entrenamiento.

### Docker Compose

```yaml
services:
  xgboost-training:
    build:
      context: .
      dockerfile: xgboost/Dockerfile
    image: xgboost-gpu
    volumes:
      - ./models:/app/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

### Dockerfile

```dockerfile
# Use NVIDIA CUDA base image with Python
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY xgboost/requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the application code and data
COPY data/ /app/data/
COPY xgboost/ /app/xgboost/

# Create models directory
RUN mkdir -p /app/models

# Set working directory to script location
WORKDIR /app

# Command to run the script
CMD ["python3", "xgboost/bost_train.py"]
```

### Dependencias

```
pandas>=1.3.0
numpy>=1.20.0
scikit-learn>=1.0.0
xgboost>=1.6.0
imbalanced-learn>=0.9.0
pytest>=7.0.0
cupy-cuda12x  # Add CuPy for GPU support
```

## Arquitectura de la Solución

La solución implementa una arquitectura end-to-end para el entrenamiento de modelos ML con las siguientes características clave:

1. **Procesamiento híbrido GPU/CPU**: Detecta automáticamente la disponibilidad de GPUs y distribuye las cargas de trabajo correspondientes.
2. **Optimización de hiperparámetros paralela**: Distribuye la búsqueda de parámetros entre múltiples GPUs cuando están disponibles.
3. **Pipeline personalizado**: Implementa un pipeline personalizado que encapsula preprocesamiento, filtrado de características y umbral de predicción.
4. **Manejo de desbalanceo de clases**: Integra técnicas de submuestreo y sobremuestreo (RandomUnderSampler y SMOTE).
5. **Selección automática de características**: Implementa filtrado basado en importancia para reducir la dimensionalidad.

## Flujo de Trabajo

### Carga y Fusión de Datos

El proceso comienza cargando dos conjuntos de datos:
- **hotels.csv**: Contiene información sobre los hoteles (12 registros, 8 variables)
- **bookings_train.csv**: Contiene información sobre las reservas (50741 registros, 15 variables)

```python
def load_data():
    hotels = pd.read_csv('/app/data/hotels.csv')
    bookings = pd.read_csv('/app/data/bookings_train.csv')
    return hotels, bookings

def merge_data(hotels, bookings):
    merged = pd.merge(bookings, hotels, on='hotel_id', how='left')
    filtered = merged[~merged['reservation_status'].isin(['Booked', np.nan])].copy()
    hotel_ids = filtered['hotel_id'].copy()
    return filtered, hotel_ids
```

Se combinan ambos conjuntos mediante un left join por `hotel_id` y se filtran reservas con estado 'Booked' o NaN, manteniendo solo registros relevantes para el análisis de cancelación.

### Preprocesamiento e Ingeniería de Características

Esta etapa es crítica y transforma los datos crudos en características predictivas:

```python
def preprocess_data(data):
    data = data.copy()

    # Convertir columnas de fecha a datetime
    date_columns = ['arrival_date', 'booking_date', 'reservation_status_date']
    for col in date_columns:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col])

    # Definir objetivo: Cancelaciones con al menos 30 días de anticipación
    data['days_before_arrival'] = (data['arrival_date'] - data['reservation_status_date']).dt.days
    data['target'] = ((data['reservation_status'] == 'Canceled') & (data['days_before_arrival'] >= 30)).astype(int)
    
    # [resto del código de ingeniería de características]
```

Las características generadas se pueden categorizar en:

1. **Características Temporales**:
   - `lead_time`: Días entre reserva y llegada
   - `lead_time_category`: Categorización de lead_time ('last_minute', 'short', 'medium', 'long', 'very_long')
   - `is_high_season`: Indicador para llegadas en temporada alta (meses 6, 7, 8, 12)
   - `is_weekend_arrival`: Indicador para llegadas en fin de semana
   - Descomposición de fechas: `arrival_month`, `arrival_dayofweek`, `booking_month`

2. **Características de Precio**:
   - `price_per_night`: Tarifa dividida por duración de estancia
   - `price_per_person`: Tarifa dividida por número de huéspedes
   - `price_deviation`: Desviación porcentual del precio respecto a la media del hotel

3. **Características de Duración**:
   - `stay_duration_category`: Categorización de duración de estancia

4. **Características de Solicitudes Especiales**:
   - `has_special_requests`: Indicador binario
   - `special_requests_ratio`: Solicitudes divididas por número de huéspedes

5. **Características de Localización**:
   - `is_foreign`: Indicador si país del cliente es diferente al país del hotel

6. **Características Interactivas**:
   - `price_length_interaction`: precio por noche × duración de estancia
   - `lead_price_interaction`: lead time × precio por noche

### Preparación de Características para Modelado

En esta etapa se crean pipelines separados para el procesamiento de variables numéricas y categóricas:

```python
def prepare_features(data):
    X = data.drop(columns=['target'])
    y = data['target']

    # Identificar características categóricas y numéricas
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Pipeline para características categóricas
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Pipeline para características numéricas
    numerical_transformer = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', StandardScaler())
    ])

    # Combinar transformadores
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

    return X, y, preprocessor
```

Este enfoque es fundamental porque:
- Para variables numéricas, se implementa imputación KNN seguida de escalado estándar
- Para variables categóricas, se implementa imputación por moda seguida de codificación one-hot
- El objeto `ColumnTransformer` mantiene la consistencia en la aplicación de estas transformaciones

### Manejo del Desbalanceo de Clases

El dataset presenta un fuerte desbalanceo con aproximadamente 86.3% de clase negativa (no cancelaciones) y 13.7% de clase positiva (cancelaciones). Para abordar este desbalanceo, se aplica una estrategia combinada:

```python
# En la función custom_parallel_search:
rus = RandomUnderSampler(sampling_strategy=0.30, random_state=42)
X_train_under, y_train_under = rus.fit_resample(X_train_transformed, y_train)

smote = SMOTE(sampling_strategy=0.65, random_state=42, k_neighbors=5)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_under, y_train_under)
```

Esta estrategia es un enfoque de dos etapas:
1. Primero se reduce la clase mayoritaria usando `RandomUnderSampler` hasta una proporción de 0.30
2. Luego se aumenta la clase minoritaria usando `SMOTE` hasta una proporción de 0.65

Este enfoque combinado preserva la variabilidad de la clase mayoritaria mientras genera ejemplos sintéticos de calidad para la clase minoritaria.

### Filtrado de Características

Para reducir la dimensionalidad y mejorar el rendimiento, se implementa una selección de características basada en importancia:

```python
def filter_zero_importance_features(model, feature_names, X_train_transformed, X_test_transformed):
    importance_scores = model.feature_importances_

    # Usar percentil para seleccionar características
    importance_threshold = np.percentile(importance_scores, 15)  # Mantener el top 85%
    important_feature_indices = np.where(importance_scores > importance_threshold)[0]

    # Asegurar mantener al menos un número mínimo de características
    min_features = max(10, int(X_train_transformed.shape[1] * 0.5))
    if len(important_feature_indices) < min_features:
        important_feature_indices = np.argsort(importance_scores)[-min_features:]

    X_train_array = np.array(X_train_transformed)
    X_test_array = np.array(X_test_transformed)

    X_train_filtered = X_train_array[:, important_feature_indices]
    X_test_filtered = X_test_array[:, important_feature_indices]

    return X_train_filtered, X_test_filtered, important_feature_indices
```

El enfoque implementado:
- Utiliza el percentil 15 como umbral adaptativo (mantiene el top 85% de características)
- Garantiza un número mínimo de características (al menos el 50% o 10, lo que sea mayor)
- Reduce significativamente la dimensionalidad después del one-hot encoding (típicamente de ~200 a ~50 características)

### Optimización de Hiperparámetros con GPU

Una de las características distintivas del proyecto es la optimización de hiperparámetros que aprovecha la aceleración por GPU:

```python
def custom_parallel_search(X_train, X_test, y_train, y_test, preprocessor, num_iterations=100):
    # Comprobar disponibilidad de GPU
    has_gpu = initialize_gpu_for_xgboost()
    
    # [...]
    
    # Comprobar GPUs y núcleos de CPU disponibles
    try:
        num_gpus = len(os.popen('nvidia-smi -L').read().strip().split('\n'))
    except:
        num_gpus = 0  # Por defecto 0 si no se detecta
    
    # [...]
    
    # Si no se detecta GPU, ejecutar todo en CPU
    if num_gpus == 0:
        print("No se detectaron GPUs, ejecutando todas las tareas en CPU")
        gpu_tasks = []
        cpu_tasks = param_sets
    else:
        # Asignar la mayoría de las tareas a GPU para maximizar su uso
        gpu_task_ratio = 1.0  # 100% de tareas en GPU cuando esté disponible
        gpu_tasks_count = int(len(param_sets) * gpu_task_ratio)
        
        # [...]
```

El proceso de optimización:
1. Detecta automáticamente la presencia de GPUs en el sistema
2. Distribuye las tareas de búsqueda entre GPUs disponibles (cuando existen)
3. Utiliza valores de referencia para parametrizar rangos de búsqueda eficientes
4. Realiza entrenamiento paralelo para maximizar la eficiencia
5. Selecciona el mejor modelo basado en el F1-score en datos de validación

### Evaluación y Selección del Umbral Óptimo

Para maximizar el F1-score, el proyecto implementa una función para encontrar el umbral óptimo de clasificación:

```python
def find_optimal_threshold(y_true, y_pred_proba):
    thresholds = np.linspace(0.1, 0.9, 400)  # Aumentar el rango del umbral
    best_threshold, best_f1 = 0.5, 0

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1, best_threshold = f1, threshold

    return best_threshold, best_f1
```

Este enfoque:
- Evalúa 400 umbrales potenciales en el rango [0.1, 0.9]
- Selecciona el umbral que maximiza el F1-score
- Típicamente encuentra valores óptimos alrededor de 0.8, lo que mejora significativamente el rendimiento respecto al umbral predeterminado de 0.5

## Pipeline Personalizado

Para facilitar la implementación en producción, se crea una clase de pipeline personalizada:

```python
class FilteredPipeline:
    def __init__(self, preprocessor, model, important_indices, best_threshold):
        self.preprocessor = preprocessor
        self.model = model
        self.important_indices = important_indices
        self.best_threshold = best_threshold

    def predict_proba(self, X):
        X_transformed = self.preprocessor.transform(X)
        X_filtered = X_transformed[:, self.important_indices]
        return self.model.predict_proba(X_filtered)

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.best_threshold).astype(int)
```

Esta clase:
- Encapsula el preprocesador, modelo, índices de características importantes y umbral óptimo
- Implementa métodos `predict_proba` y `predict` compatibles con la API de scikit-learn
- Asegura que se apliquen todas las transformaciones necesarias durante la inferencia
- Aplica automáticamente el umbral óptimo al hacer predicciones
- Facilita la serialización y deserialización mediante pickle

## Resultados Obtenidos

El modelo entrenado muestra un excelente rendimiento en la predicción de cancelaciones de reservas hoteleras con 30 días o más de antelación:

| Métrica | Valor Típico |
|---------|--------------|
| Exactitud | 0.961        |
| F1-Score | 0.863        |
| Precisión | 0.844        |
| Recall | 0.882        |
| AUC ROC | 0.987        |
| Umbral Óptimo | 0.86         |

Las características más predictivas generalmente incluyen:
- `lead_time` y su categorización
- `lead_price_interaction`
- Variables relacionadas con el precio y su desviación

## Requisitos del Sistema

Para ejecutar el proyecto de forma óptima:

- **Hardware**:
   - GPU compatible con CUDA (opcional pero recomendado)
   - Al menos 8GB de RAM

- **Software**:
   - Docker y Docker Compose
   - Controladores NVIDIA y NVIDIA Container Toolkit (para aceleración GPU)

## Ejecución del Proyecto

Para entrenar el modelo:

```bash
# Construir y ejecutar el contenedor Docker
docker-compose up xgboost-training

# Alternativamente, ejecutar directamente (con GPU)
python3 xgboost/bost_train.py
```

El modelo entrenado se guardará en `models/xgboost_model.pkl`.

Para utilizar el modelo en inferencia:

```python
import pickle

# Cargar el modelo serializado
with open('models/xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Realizar predicciones en nuevos datos
predictions = model.predict(new_data)
```

## Conclusiones

Este proyecto implementa un pipeline completo de machine learning para predecir cancelaciones de reservas hoteleras, abordando los principales desafíos:

1. **Preprocesamiento robusto**: Implementa manejo adecuado de valores faltantes y transformaciones de variables
2. **Ingeniería de características avanzada**: Genera características predictivas a partir de datos crudos
3. **Manejo del desbalanceo**: Aplica estrategias combinadas para equilibrar las clases
4. **Optimización eficiente**: Utiliza aceleración GPU para buscar los mejores hiperparámetros
5. **Ajuste fino**: Optimiza el umbral de decisión para maximizar el F1-score

El enfoque híbrido GPU/CPU permite una escalabilidad natural desde equipos personales hasta entornos empresariales con múltiples GPUs, mientras que el pipeline personalizado facilita la transición del modelo a producción.

Las métricas obtenidas demuestran la efectividad del modelo para identificar cancelaciones con suficiente antelación, lo que permite a los hoteles implementar estrategias proactivas de gestión de capacidad y maximización de ingresos.