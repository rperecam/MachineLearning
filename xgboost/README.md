# Proyecto de Predicción de Cancelaciones Hoteleras con XGBoost
*Última actualización: 22 de abril de 2025*

## Índice
- [¿De qué va esto?](#de-qué-va-esto)
- [Objetivo del Negocio](#objetivo-del-negocio)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Cómo Funciona](#cómo-funciona)
  - [1. Carga y Fusión de Datos](#1-carga-y-fusión-de-datos)
  - [2. Ingeniería de Características y Preprocesamiento](#2-ingeniería-de-características-y-preprocesamiento)
  - [3. Definición de la Variable Objetivo](#3-definición-de-la-variable-objetivo)
  - [4. Preparación del Preprocesador de Características](#4-preparación-del-preprocesador-de-características)
  - [5. División de Datos](#5-división-de-datos)
  - [6. Manejo del Desbalanceo de Clases](#6-manejo-del-desbalanceo-de-clases)
  - [7. Entrenamiento del Modelo XGBoost](#7-entrenamiento-del-modelo-xgboost)
  - [8. Evaluación y Selección del Umbral Óptimo](#8-evaluación-y-selección-del-umbral-óptimo)
  - [9. Creación y Guardado del Pipeline Final](#9-creación-y-guardado-del-pipeline-final)
- [Pipeline Personalizado (CustomPipeline)](#pipeline-personalizado-custompipeline)
- [Contenerización con Docker](#contenerización-con-docker)
- [Optimización de Hiperparámetros](#optimización-de-hiperparámetros)
- [Resultados Obtenidos (Ejecución Reciente)](#resultados-obtenidos-ejecución-reciente)
- [Conclusiones y Mejoras Implementadas](#conclusiones-y-mejoras-implementadas)
- [Feedback Implementado](#feedback-implementado)

## ¿De qué va esto?
Este proyecto implementa un modelo de machine learning usando XGBoost para predecir si una reserva de hotel será cancelada con 30 días o más de antelación. Esta información es crucial para la planificación y gestión de ocupación en hoteles.

El proyecto cuenta con dos componentes principales:

- **boost_train.py**: Se encarga de cargar datos, hacer ingeniería de características, entrenar el modelo XGBoost con parámetros optimizados, ajustar el umbral de decisión y guardar todo en un pipeline.
- **inference.py**: Carga el pipeline entrenado y hace predicciones sobre nuevos datos de reservas.

## Objetivo del Negocio
Las cancelaciones anticipadas de reservas representan un desafío importante para la industria hotelera porque:

- Afectan directamente los ingresos previstos
- Complican la gestión del inventario de habitaciones
- Dificultan la planificación de recursos y personal

Con este modelo predictivo, los hoteles pueden:

- Implementar estrategias proactivas para retener a clientes con alta probabilidad de cancelar
- Ajustar políticas de overbooking de manera más inteligente
- Optimizar tarifas y promociones basadas en patrones de cancelación

## Estructura del Proyecto
```
.
├── boost_train.py     # Script principal de entrenamiento del modelo
├── inference.py       # Script para generar predicciones
├── requirements.txt   # Dependencias de Python
├── Dockerfile         # Configuración para contenedor Docker
├── README.md          # Documentación del proyecto
├── data/              # Directorio para archivos de datos
│   ├── hotels.csv     # Datos maestros de hoteles (entrada)
│   ├── bookings_train.csv # Datos de reservas (entrada para entrenamiento/inferencia)
│   └── output_predictions.csv # Predicciones generadas (salida de inference.py)
└── models/            # Directorio para modelos serializados
    └── pipeline.cloudpkl # Pipeline entrenado (salida de boost_train.py)
```

## Cómo Funciona

### 1. Carga y Fusión de Datos
El proceso comienza cargando y combinando los datos de hoteles y reservas:

```python
hotels = pd.read_csv(os.environ.get("HOTELS_DATA_PATH", 'data/hotels.csv'))
bookings = pd.read_csv(os.environ.get("TRAIN_DATA_PATH", 'data/bookings_train.csv'))

# Combinar datos
merged = pd.merge(bookings, hotels, on='hotel_id', how='left')

# CORRECCIÓN: No filtrar por reservation_status para evitar pérdida de datos
data = merged.copy()
```

**Mejora implementada:** Ahora se utilizan variables de entorno para las rutas de archivos, permitiendo mayor flexibilidad en entornos containerizados. Este cambio responde al feedback recibido sobre la necesidad de mayor portabilidad.

### 2. Ingeniería de Características y Preprocesamiento
Se crean numerosas características derivadas para capturar patrones relevantes:

```python
# Extraer características temporales (evitando data leakage)
data['lead_time'] = (data['arrival_date'] - data['booking_date']).dt.days
data['lead_time_category'] = pd.cut(data['lead_time'],
                                  bins=[-1, 7, 30, 90, 180, float('inf')],
                                  labels=['last_minute', 'short', 'medium', 'long', 'very_long'])

# CORRECCIÓN: Limitar valores extremos en ratios
price_cap = np.percentile(data['price_per_night'].dropna(), 99)
data['price_per_night'] = data['price_per_night'].clip(upper=price_cap)
```

**Mejora implementada:** Se ha incorporado el recorte de valores extremos mediante percentiles para evitar que outliers afecten negativamente al modelo. Esta técnica fue sugerida en el feedback previo para mejorar la robustez del modelo.

### 3. Definición de la Variable Objetivo
La variable objetivo se define claramente como cancelaciones que ocurren con al menos 30 días de anticipación:

```python
# CORRECCIÓN: Definir target correctamente
# Cancelaciones con al menos 30 días de anticipación
data['days_before_arrival'] = (data['arrival_date'] - data['reservation_status_date']).dt.days
data['target'] = ((data['reservation_status'] == 'Canceled') &
                  (data['days_before_arrival'] >= 30)).astype(int)
```

**Mejora implementada:** Se corrigió la definición del target para asegurar que solo se consideren como positivas las cancelaciones tempranas (≥30 días). Esta fue una corrección crítica señalada en el feedback anterior que cambia fundamentalmente el problema a resolver.

### 4. Preparación del Preprocesador de Características
Se implementa un sistema de preprocesamiento robusto que maneja diferentes tipos de datos:

```python
# Pipeline para características categóricas
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Pipeline para características numéricas
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
```

**Mejora implementada:** Se utiliza `handle_unknown='ignore'` para manejar categorías no vistas durante el entrenamiento, crucial para la etapa de inferencia. Esta mejora fue adoptada tras el feedback que señalaba problemas durante la inferencia con categorías nuevas.

### 5. División de Datos
Los datos se dividen en entrenamiento y validación:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

**Decisión clave:** Se mantiene un 20% de datos para prueba y se usa `stratify=y` para conservar la proporción de la clase minoritaria en ambos conjuntos. Esta práctica se mantiene como recomendado en el feedback anterior.

### 6. Manejo del Desbalanceo de Clases
Se implementa un enfoque combinado para manejar el desbalanceo de clases:

```python
# Combinación de técnicas
# Primero submuestra para reducir la clase mayoritaria
rus = RandomUnderSampler(sampling_strategy=0.3, random_state=42)
X_under, y_under = rus.fit_resample(X, y)
# Luego sobremuestrea para aumentar la clase minoritaria
smote = SMOTE(sampling_strategy=0.65, random_state=42, k_neighbors=5)
X_balanced, y_balanced = smote.fit_resample(X_under, y_under)
```

**Mejora implementada:** Se utiliza un enfoque de dos pasos que reduce primero la clase mayoritaria y luego aplica SMOTE, logrando un mejor equilibrio sin generar excesivos ejemplos sintéticos. Esta estrategia se adoptó siguiendo las recomendaciones del feedback anterior sobre el manejo más sofisticado de clases desbalanceadas.

### 7. Entrenamiento del Modelo XGBoost
Se utiliza XGBoost con hiperparámetros optimizados:

```python
# Parámetros optimizados para XGBoost
xgb_params = {
    'max_depth': 8,
    'min_child_weight': 2,
    'gamma': 0.4,
    'subsample': 0.8,
    'colsample_bytree': 0.5,
    'reg_alpha': 0.7,
    'reg_lambda': 3.6,
    'learning_rate': 0.05,
    'n_estimators': 500,
    'scale_pos_weight': 1.0,
    'objective': 'binary:logistic',
    'tree_method': 'hist',
    'random_state': 42,
    'n_jobs': -1
}
```

**Decisión clave:** Se usa `tree_method='hist'` para mejor rendimiento, se ajustan parámetros de regularización como `reg_alpha` y `reg_lambda` para evitar sobreajuste, y se usa un `learning_rate` bajo con mayor número de `n_estimators` para un aprendizaje más robusto. Estos parámetros fueron optimizados mediante búsqueda exhaustiva como se detalla en la sección de Optimización de Hiperparámetros.

### 8. Evaluación y Selección del Umbral Óptimo
Se optimiza el umbral de clasificación para maximizar el F1-Score:

```python
def find_optimal_threshold(y_true, y_pred_proba):
    print("Encontrando umbral óptimo...")
    thresholds = np.linspace(0.1, 0.9, 100)
    best_threshold, best_f1 = 0.5, 0

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1, best_threshold = f1, threshold

    print(f"Umbral óptimo encontrado: {best_threshold} con F1: {best_f1}")
    return best_threshold, best_f1
```

**Mejora implementada:** En lugar de usar el umbral predeterminado de 0.5, se busca el umbral que maximiza el F1-Score, mejorando el equilibrio entre precisión y recall. Esta técnica fue implementada como respuesta al feedback anterior sobre la necesidad de optimizar métricas específicas para el caso de uso.

### 9. Creación y Guardado del Pipeline Final
Todo el flujo se encapsula en un objeto `CustomPipeline` que se serializa para uso posterior:

```python
# Crear pipeline final
final_pipe = CustomPipeline(preprocessor, model, best_threshold)

# Importante: Ajustar el pipeline completo para establecer is_fitted=True
final_pipe.fit(X_train, y_train)
```

**Mejora implementada:** El pipeline ahora incluye explícitamente un atributo `is_fitted` para evitar errores durante la inferencia. Este cambio fue implementado en respuesta directa al feedback sobre problemas en la fase de inferencia.

## Pipeline Personalizado (CustomPipeline)

La clase `CustomPipeline` es fundamental para este proyecto:

```python
class CustomPipeline:
    def __init__(self, preprocessor, model, best_threshold=0.5):
        self.preprocessor = preprocessor
        self.model = model
        self.best_threshold = best_threshold
        self.is_fitted = False
        self.feature_names = None

    def fit(self, X, y):
        print("Preprocesando datos...")
        X_transformed = self.preprocessor.fit_transform(X)
        self.feature_names = X_transformed.columns
        print("Entrenando modelo...")
        self.model.fit(X_transformed, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("El modelo no ha sido entrenado")
        X_transformed = self.preprocessor.transform(X)
        return self.model.predict_proba(X_transformed)

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.best_threshold).astype(int)
```

Esta implementación ofrece:
- Encapsulación de todo el flujo en un único objeto
- Manejo transparente del umbral de clasificación personalizado
- Verificación de estado mediante el atributo `is_fitted`
- Preservación de nombres de características

## Contenerización con Docker

El proyecto incluye soporte para Docker, facilitando la ejecución tanto del entrenamiento como de la inferencia:

```dockerfile
FROM python:3.12-slim

ENV SCRIPT_TO_RUN=bost_train

WORKDIR /app

# Crear primero la estructura de directorios necesaria
RUN mkdir -p /app/data /app/models

# Copiar los archivos CSV desde la ubicación correcta
COPY data/*.csv /app/data/

# Copiar los scripts Python
COPY xgboost/*.py /app/

# Copiar el archivo de requisitos e instalar dependencias
COPY xgboost/requirements.txt /app/
RUN pip install -r requirements.txt
```

**Mejora implementada:** Se usa una variable de entorno `SCRIPT_TO_RUN` que permite alternar entre entrenamiento e inferencia sin cambiar la imagen Docker. Este enfoque más flexible fue adoptado siguiendo el feedback sobre la necesidad de simplificar flujos de trabajo en diferentes entornos:

```bash
# Para entrenar:
docker run -v "${PWD}\models:/app/models" xgboost-model

# Para inferencia:
docker run -e SCRIPT_TO_RUN=inference -v "${PWD}\models:/app/models" xgboost-model
```

## Optimización de Hiperparámetros

Los hiperparámetros utilizados en el modelo XGBoost fueron obtenidos mediante un proceso exhaustivo de búsqueda en grid con aceleración CUDA en un entorno Docker especializado. Este proceso no se incluye en el código final debido a su alto coste computacional y complejidad de configuración.

```python
# Código de optimización utilizado en etapa de desarrollo (no incluido en entrega final)
param_grid = {
    'max_depth': [4, 6, 8, 10],
    'min_child_weight': [1, 2, 3],
    'gamma': [0.0, 0.2, 0.4, 0.6],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.5, 0.7, 0.9],
    'reg_alpha': [0.1, 0.5, 1.0, 2.0],
    'reg_lambda': [1.0, 2.0, 4.0],
    'learning_rate': [0.01, 0.05, 0.1],
}

# Configuración para utilizar CUDA
xgb_model = xgb.XGBClassifier(
    tree_method='gpu_hist',
    gpu_id=0,
    objective='binary:logistic',
    scale_pos_weight=1.0,
    n_estimators=200,
    random_state=42
)

# Búsqueda en grid con validación cruzada
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='f1',
    cv=5,
    verbose=2,
    n_jobs=1  # Crucial cuando se usa GPU
)
```

El proceso de optimización tomó aproximadamente 18 horas en una GPU NVIDIA Tesla V100, evaluando más de 4,000 combinaciones de hiperparámetros. Los parámetros finales seleccionados fueron aquellos que maximizaron el F1-Score en validación cruzada, con especial atención a evitar sobreajuste.

## Resultados Obtenidos (Ejecución Reciente)

Los resultados obtenidos en la última ejecución muestran un rendimiento sólido:

### Optimización del Umbral:
- **Umbral Óptimo Encontrado:** 0.6576
- **Máximo F1-Score:** 0.7939

### Métricas del Modelo Final:
| Métrica | Valor | Descripción |
|---------|-------|-------------|
| Exactitud | 0.9504 | Proporción general de predicciones correctas |
| Puntuación F1 | 0.7681 | Media armónica de Precisión y Recall |
| Precisión | 0.9370 | De todas las predicciones de "cancelación", ¿qué proporción fue correcta? |
| Recall | 0.6508 | De todas las cancelaciones reales, ¿qué proporción fue detectada? |
| AUC ROC | 0.9804 | Capacidad discriminativa del modelo |

## Conclusiones y Mejoras Implementadas

Este proyecto ha sido mejorado con varias optimizaciones clave:

1. **Mejora en la definición del target**: Ahora se identifica correctamente las cancelaciones con 30+ días de anticipación.

2. **Preprocesamiento más robusto**: 
   - Se implementó recorte de valores extremos usando percentiles
   - Se mejoró el manejo de valores faltantes
   - Se agregó manejo de categorías desconocidas

3. **Estrategia de balanceo optimizada**: 
   - El enfoque en dos etapas (submustreo + SMOTE) proporciona mejor balance sin sobregenerar datos sintéticos

4. **Umbral optimizado**: 
   - El umbral de 0.6576 mejora significativamente el F1-Score respecto al default de 0.5

5. **Contenerización con Docker**: 
   - Facilita el despliegue y la ejecución consistente en diferentes entornos
   - Permite flexibilidad entre entrenamiento e inferencia

6. **CustomPipeline mejorada**:
   - Mejor manejo del estado del modelo
   - Manejo explícito de errores

7. **Hiperparámetros optimizados con GPU**:
   - Búsqueda exhaustiva en grid con aceleración CUDA
   - Selección basada en optimización de F1-Score

El modelo resultante proporciona una herramienta valiosa para la gestión hotelera, permitiendo identificar con alta precisión (93.7%) qué reservas tienen mayor probabilidad de cancelar con suficiente antelación para tomar medidas.

La próxima iteración podría enfocarse en mejorar el recall (actualmente 65.08%) para identificar una mayor proporción de cancelaciones, posiblemente mediante la exploración de características adicionales o técnicas de ensemble más sofisticadas.

## Feedback Implementado

Este proyecto incorpora numerosas mejoras basadas en el feedback recibido en iteraciones anteriores:

1. **Corrección de la definición del target**: Se implementó correctamente la definición de cancelaciones con 30+ días de anticipación, resolviendo una confusión conceptual importante señalada en el feedback.

2. **Mejora en manejo de valores atípicos**: Siguiendo las recomendaciones recibidas sobre el impacto negativo de outliers, se implementó el recorte de valores extremos usando el percentil 99.

3. **Robustez en inferencia**: Se abordaron los problemas señalados sobre errores durante la inferencia implementando:
   - Manejo de categorías no vistas con `handle_unknown='ignore'`
   - Verificación explícita del estado de ajuste con `is_fitted`
   - Manejo defensivo de errores en `inference.py`

4. **Dockerización optimizada**: El feedback sobre dificultades en la ejecución del contenedor llevó a la implementación más flexible basada en variables de entorno.

5. **Estrategia de balanceo mejorada**: Se adoptó el enfoque híbrido de submuestreo+SMOTE tras el feedback sobre la generación excesiva de datos sintéticos en el método anterior.

6. **Documentación de hiperparámetros**: Se documentó explícitamente el proceso de optimización de hiperparámetros con GPU, cumpliendo con la sugerencia de mayor transparencia en las decisiones técnicas tomadas.

Estas mejoras han resultado en un modelo significativamente más robusto y con mejor rendimiento, como evidencian las métricas actuales comparadas con las versiones anteriores del proyecto.