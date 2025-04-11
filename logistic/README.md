# Memoria Detallada: Modelo Predictivo de Cancelación de Reservas Hoteleras

## 1. Introducción

Este documento presenta un análisis detallado del desarrollo de un modelo de regresión logística para la predicción de cancelaciones de reservas hoteleras, con énfasis particular en las cancelaciones tardías (30 días antes de la fecha de llegada). El modelo se enmarca dentro de un sistema completo que incluye procesos de preprocesamiento, entrenamiento, evaluación y despliegue, siguiendo una arquitectura modular y escalable.

## 2. Análisis Detallado del Código de Entrenamiento (train.py)

### 2.1. Mapeo de Continentes (`ContinentMapper`)

```python
class ContinentMapper(BaseEstimator, TransformerMixin):
    def __init__(self, country_col='country_x', continent_col='continent_customer', unknown='Desconocido'):
        self.country_col = country_col
        self.continent_col = continent_col
        self.unknown = unknown
        self.mapping = {
            'SPA': 'Europa', 'FRA': 'Europa', 'POR': 'Europa', ...
            'USA': 'América del Norte', 'MEX': 'América del Norte', ...
            'BRA': 'América del Sur', 'ARG': 'América del Sur', ...
            # Otros mapeos de países a continentes
        }
```

**Función y propósito:**
- Transforma códigos ISO de países (3 letras) a sus respectivos continentes.
- Crea una nueva variable categórica de menor cardinalidad (`continent_customer`).
- Asigna un valor predeterminado (`Desconocido`) para países no mapeados.

**Análisis técnico detallado:**
- Hereda de `BaseEstimator` y `TransformerMixin` para integrarse con scikit-learn.
- El método `fit()` no realiza operaciones, solo retorna `self` para cumplir con la API.
- El método `transform()` aplica el mapeo mediante un diccionario predefinido.
- Permite personalizar las columnas de entrada/salida mediante parámetros.

**Justificación:**
- **Reducción de dimensionalidad**: Disminuye significativamente la cantidad de categorías (de ≈200 países a 6 continentes).
- **Tratamiento del ruido**: Agrupa países con pocos datos, evitando el overfitting en categorías poco representadas.
- **Interpretabilidad**: Las diferencias geográficas a nivel de continente son más interpretables y estables.
- **Eficiencia computacional**: Reduce la complejidad al aplicar codificación one-hot posteriormente.

### 2.2. Ingeniería de Características (`FeatureEngineer`)

```python
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, date_cols=['arrival_date', 'booking_date'],
                 asset_cols=['pool_and_spa', 'restaurant', 'parking'],
                 country_cols=['country_x', 'country_y']):
        self.date_cols = date_cols
        self.asset_cols = asset_cols
        self.country_cols = country_cols
```

**Características generadas y su relevancia:**

1. **`lead_time`** (días entre reserva y llegada):
   - **Cálculo**: `(X_copy['arrival_date'] - X_copy['booking_date']).dt.days`
   - **Relevancia**: Variable crítica que captura el comportamiento de planificación.
   - **Hipótesis de negocio**: Las reservas hechas con mucha antelación tienen mayor riesgo de cancelación por:
     - Mayor probabilidad de cambio en planes del viajero.
     - Mayor probabilidad de encontrar ofertas mejores.
     - Reservas especulativas (bloquear precio sin compromiso firme).

2. **`num_assets`** (conteo de amenidades/instalaciones):
   - **Cálculo**: Suma de columnas binarias de instalaciones (`pool_and_spa`, `restaurant`, `parking`).
   - **Relevancia**: Captura el valor percibido y categoría del hotel.
   - **Hipótesis de negocio**: Hoteles con más amenidades podrían tener:
     - Menor tasa de cancelación debido al valor añadido.
     - Políticas de cancelación más estrictas.
     - Distinto perfil de cliente (negocios vs. turismo).

3. **`is_foreign`** (indicador de cliente internacional):
   - **Cálculo**: Compara `country_x` (país del cliente) con `country_y` (país del hotel).
   - **Relevancia**: Captura la naturaleza del viaje (doméstico vs. internacional).
   - **Hipótesis de negocio**: Viajeros internacionales podrían:
     - Tener mayor riesgo de cancelación por problemas de visado, cambios en restricciones de viaje.
     - Planificar con mayor antelación, aumentando la ventana de posible cancelación.
     - Tener mayor sensibilidad a eventos geopolíticos o cambiarios.

**Implementación técnica:**
- Manejo seguro de fechas con `pd.to_datetime` y `errors='coerce'`.
- Verificación de la existencia de columnas antes de usarlas.
- Valores por defecto para casos donde faltan datos.

**Ventajas como baseline:**
- **Conocimiento de dominio incorporado**: Transforma datos crudos en características con sentido de negocio.
- **Eficiencia**: Genera variables derivadas sin aumentar excesivamente la dimensionalidad.
- **Flexibilidad**: Parametrizable para adaptarse a diferentes esquemas de datos.
- **Robustez**: Maneja casos de columnas faltantes o valores nulos.

### 2.3. Manejo de Outliers (`OutlierCapper`)

```python
class OutlierCapper(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, factor=1.5):
        self.columns = columns
        self.factor = factor
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}
```

**Funcionamiento detallado:**
- Implementa la técnica del rango intercuartílico (IQR) para detectar y tratar valores atípicos.
- En la fase de `fit()`:
  1. Calcula Q1 (percentil 25) y Q3 (percentil 75) para cada columna numérica.
  2. Determina IQR = Q3 - Q1.
  3. Establece límites: 
     - Inferior: Q1 - factor*IQR
     - Superior: Q3 + factor*IQR
- En la fase de `transform()`:
  1. Reemplaza valores por debajo del límite inferior con el valor del límite inferior.
  2. Reemplaza valores por encima del límite superior con el valor del límite superior.

**Parámetros importantes:**
- `factor`: Controla la agresividad del recorte (valor estándar: 1.5).
  - Valores más altos son más conservadores (mantienen más valores extremos).
  - Valores más bajos son más agresivos (tratan más valores como outliers).
- `columns`: Permite especificar qué columnas tratar, evitando modificar variables donde los outliers podrían ser significativos.

**Análisis técnico:**
- Utiliza funciones vectorizadas de NumPy (`np.where`) para eficiencia computacional.
- Almacena los límites en el objeto para mantener consistencia entre training e inferencia.
- Se integra en el pipeline como un paso formal, no como preprocesamiento ad-hoc.

**Ventajas como baseline:**
- **Robustez estadística**: Utiliza IQR, más robusto que métodos basados en desviaciones estándar para distribuciones no normales.
- **Preservación de datos**: No elimina observaciones, manteniendo el tamaño de la muestra.
- **Suavizado de distribuciones**: Reduce el impacto de valores extremos sin distorsionar relaciones generales.
- **Mejora de convergencia**: Facilita el entrenamiento de modelos sensibles a outliers.

### 2.4. Ingeniería de la Variable Objetivo

```python
def engineer_target(df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    # ...
    df_eng['is_canceled'] = df_eng['reservation_status'].isin(['Canceled', 'No-Show']).astype(int)
    df_eng['days_to_arrival'] = (df_eng['arrival_date'] - df_eng['reservation_status_date']).dt.days
    df_eng['cancelled_last_30_days'] = (df_eng['is_canceled'] == 1) & (df_eng['days_to_arrival'] <= 30)
    y = df_eng['cancelled_last_30_days'].fillna(False).astype(int)
    # ...
```

**Proceso de construcción del target:**
1. Identificación de cancelaciones:
   - Considera como canceladas las reservas con estado 'Canceled' o 'No-Show'.
   - Convierte a variable binaria `is_canceled` (1: cancelada, 0: no cancelada).

2. Cálculo de temporalidad:
   - Calcula `days_to_arrival`: diferencia entre fecha de llegada y fecha de cambio de estado.
   - Valores negativos indican cambios después de la fecha de llegada programada.
   - Valores positivos indican días de antelación al cambiar el estado.

3. Definición del target específico:
   - `cancelled_last_30_days`: reservas canceladas (1) dentro de los últimos 30 días antes de la llegada.
   - Representa cancelaciones tardías, críticas para la operación hotelera.

**Análisis de la definición del target:**
- Se centra en cancelaciones de alto impacto (tardías) que:
  - Son más difíciles de recuperar con nuevas reservas.
  - Tienen mayor impacto financiero (menor tiempo para mitigar pérdidas).
  - Pueden estar sujetas a diferentes patrones que cancelaciones tempranas.

- Equilibrio entre especificidad y generalidad:
  - Suficientemente específico: foco en problema de negocio concreto.
  - Suficientemente general: permite aplicación en diversos contextos hoteleros.

**Ventajas como baseline:**
- **Enfoque en problema de negocio**: Alineado con necesidades operativas reales.
- **Simplicidad interpretativa**: Target binario claro y accionable.
- **Flexibilidad temporal**: El umbral de 30 días podría ajustarse según necesidades del negocio.

### 2.5. Procesamiento de Datos Numéricos y Categóricos

```python
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, make_column_selector(dtype_include=np.number)),
        ('cat', categorical_transformer, make_column_selector(dtype_include='object'))
    ],
    remainder='drop'
)
```

**Procesamiento de variables numéricas:**
1. **Imputación con mediana**:
   - Ventajas: Robustez frente a distribuciones sesgadas y outliers.
   - Alternativas descartadas: Media (sensible a outliers), constante (no aprovecha distribución).

2. **Estandarización**:
   - Proceso: Transformación a distribución con media 0 y desviación estándar 1.
   - Importancia: Fundamental para modelos basados en distancias o penalizaciones (como regresión logística).
   - Beneficios: Mejora convergencia, estabiliza coeficientes, equilibra influencia de variables.

**Procesamiento de variables categóricas:**
1. **Imputación con moda**:
   - Estrategia: Reemplaza valores faltantes con la categoría más frecuente.
   - Justificación: Preserva la distribución original y no introduce categorías artificiales.

2. **Codificación one-hot**:
   - Configuración: `handle_unknown='ignore'` para manejar categorías no vistas en entrenamiento.
   - `sparse_output=False` para generar matrices densas (más eficientes en este contexto).
   - No se usa codificación ordinal al no existir relación de orden entre categorías.

**Implementación con `ColumnTransformer`:**
- Uso de `make_column_selector` para selección automática por tipo de dato.
- Configuración `remainder='drop'` para excluir columnas no procesadas explícitamente.

**Ventajas como baseline:**
- **Automatización**: Detección automática de tipos evita especificación manual de columnas.
- **Consistencia**: Tratamiento unificado y reproducible.
- **Encapsulación**: Todo el preprocesamiento se serializa con el modelo.
- **Modularidad**: Facilidad para modificar estrategias específicas sin afectar otras partes.

### 2.6. Selección de Características

El pipeline implementa dos técnicas complementarias de selección de características:

#### 2.6.1. Filtrado por Varianza

```python
('variance_threshold', VarianceThreshold(threshold=self.variance_threshold))
```

**Funcionamiento detallado:**
- Elimina características con varianza inferior al umbral especificado (0.001 por defecto).
- Identifica y elimina:
  - Variables constantes (var=0), que no aportan información predictiva.
  - Variables cuasi-constantes (var≈0), con valor casi siempre igual, como flags raramente activados.
  - Columnas one-hot con distribución muy desbalanceada.

**Análisis técnico:**
- Método no supervisado: no utiliza la variable objetivo.
- Complementario a métodos basados en correlación.
- Aplicado después del preprocesamiento para considerar también variables transformadas.

**Ventajas como filtro inicial:**
- **Eficiencia computacional**: Cálculo simple que reduce dimensionalidad antes de métodos más costosos.
- **Reducción de ruido**: Elimina variables con poca información discriminativa.
- **Prevención de multicolinealidad**: Ayuda a eliminar columnas redundantes creadas en one-hot encoding.

#### 2.6.2. Selección de Mejores K Características

```python
('feature_selection', SelectKBest(score_func=f_classif, k=self.k_best))
```

**Funcionamiento detallado:**
- Aplica test ANOVA (F-value) para evaluar la relación entre cada característica y el target.
- Selecciona las `k` características con mayor valor estadístico F (22 en la configuración utilizada).
- Evalúa la significancia de la diferencia en medias entre grupos (cancelación vs. no cancelación).

**Análisis técnico:**
- Método supervisado: considera explícitamente la relación con el target.
- Apropiado para clasificación binaria (análisis de varianza entre clases).
- Aplicado después de SMOTE para evaluar las características con clases balanceadas.

**Ventajas específicas:**
- **Interpretabilidad**: Identifica explícitamente características con mayor poder predictivo.
- **Reducción de dimensionalidad**: Limita el modelo final a un conjunto manejable de variables.
- **Mitigación de sobreajuste**: Elimina variables que podrían capturar ruido en lugar de señal.
- **Mejora de generalización**: Se enfoca en variables con relación estadísticamente significativa.

### 2.7. Manejo del Desbalanceo de Clases con SMOTE

```python
('smote', SMOTE(random_state=self.random_state))
```

**Funcionamiento detallado:**
1. Identificación de la clase minoritaria (cancelaciones).
2. Para cada instancia de la clase minoritaria:
   - Encuentra sus k vecinos más cercanos en la misma clase.
   - Selecciona aleatoriamente uno de estos vecinos.
   - Genera un punto sintético en algún lugar del segmento entre la instancia y el vecino.
3. Añade suficientes puntos sintéticos hasta igualar la cantidad de la clase mayoritaria.

**Configuración específica:**
- Implementado dentro del pipeline después del preprocesamiento.
- Aplicado con semilla aleatoria fija para reproducibilidad.
- Utiliza parámetros por defecto para k-vecinos (k=5).

**Análisis técnico:**
- Aborda el problema de desbalance creando nuevas instancias sintéticas.
- Genera puntos en el espacio de características, no simplemente duplica ejemplos existentes.
- Más sofisticado que técnicas de sobremuestreo aleatorio que pueden provocar sobreajuste.

**Ventajas frente a alternativas:**
- **Vs. Submuestreo**: No pierde información de la clase mayoritaria.
- **Vs. Ponderación de clases**: No simplemente ajusta la importancia, sino que enriquece el espacio de características.
- **Vs. ADASYN**: Más simple y generalmente suficiente como baseline.
- **Vs. Ajuste de umbrales**: Aborda el problema en la fase de entrenamiento, no solo en la decisión final.

### 2.8. Algoritmo de Clasificación y Optimización

```python
if self.model_type == 'logistic':
    pipeline_steps.append(('classifier', LogisticRegression(random_state=self.random_state, solver='liblinear', max_iter=2000)))
    self.param_grid = {
        'classifier__C': [0.001, 0.01, 0.1, 1, 10],
        'classifier__penalty': ['l1', 'l2']
    }
```

**Algoritmo principal: Regresión Logística**
- **Configuración base**:
  - `solver='liblinear'`: Optimizador eficiente para conjuntos de datos pequeños/medianos.
  - `max_iter=2000`: Número elevado de iteraciones para garantizar convergencia.
  - `random_state`: Semilla fija para reproducibilidad.

- **Hiperparámetros optimizados**:
  1. **Parámetro C** (inverso de la regularización):
     - Rango explorado: [0.001, 0.01, 0.1, 1, 10]
     - Valores pequeños: mayor regularización, modelo más simple.
     - Valores grandes: menor regularización, modelo más complejo.
  
  2. **Tipo de regularización** (`penalty`):
     - `l1` (Lasso): Tiende a producir coeficientes exactamente cero, realizando selección de características implícita.
     - `l2` (Ridge): Penaliza coeficientes grandes, distribuyendo el impacto entre variables correlacionadas.

**Algoritmo alternativo: SGD Classifier**
- Implementación más eficiente para conjuntos de datos grandes.
- Configuración equivalente a regresión logística mediante `loss='log_loss'`.
- Exploración de diferentes niveles (`alpha`) y tipos de regularización.

**Búsqueda de hiperparámetros:**
```python
cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
grid_search = GridSearchCV(self.pipeline, self.param_grid, cv=cv, scoring='f1', n_jobs=-1, verbose=1)
```

- **Validación cruzada estratificada**: Mantiene distribución de clases en todas las particiones.
- **Métrica de optimización**: F1-score (equilibrio entre precisión y recall).
- **Paralelización**: Utiliza todos los núcleos disponibles (`n_jobs=-1`).

**Resultado de la optimización:**
```
Mejores parámetros: {'classifier__C': 0.1, 'classifier__penalty': 'l2'}
```
- Regularización moderada (C=0.1) con penalización L2.
- Indica preferencia por un modelo más simple que distribuye la importancia entre variables correlacionadas.

**Ventajas como baseline:**
- **Eficiencia computacional**: Entrenamiento rápido incluso con datos moderadamente grandes.
- **Interpretabilidad**: Coeficientes directamente relacionados con el logaritmo de odds ratios.
- **Calibración de probabilidades**: Produce probabilidades bien calibradas sin postprocesamiento.
- **Flexibilidad**: Diferentes configuraciones de regularización permiten ajustar el balance sesgo-varianza.

### 2.9. Evaluación del Modelo y Análisis de Métricas

```python
self.metrics = {
    'f1_score': f1_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred),
    'recall': recall_score(y_test, y_pred),
    'accuracy': accuracy_score(y_test, y_pred),
    'roc_auc': roc_auc_score(y_test, y_pred_proba),
    'pr_auc': auc(precision_recall_curve(y_test, y_pred_proba)[1], precision_recall_curve(y_test, y_pred_proba)[0])
}
```

**Resultados obtenidos:**
```
f1_score: 0.4460
precision: 0.3142
recall: 0.7684
accuracy: 0.6535
roc_auc: 0.7497
pr_auc: 0.3332
```

**Análisis detallado de las métricas:**

1. **F1-Score (0.4460)**:
   - Media armónica entre precisión y recall.
   - Valor moderado que refleja el compromiso entre identificar correctamente cancelaciones y minimizar falsos positivos.
   - Interpretación: El modelo tiene un rendimiento equilibrado, aunque no óptimo.

2. **Precisión (0.3142)**:
   - Proporción de predicciones positivas que son correctas.
   - Valor relativamente bajo: por cada 100 reservas que el modelo predice como cancelaciones, solo ~31 realmente se cancelan.
   - Implicación: Tendencia a sobreestimar cancelaciones (muchos falsos positivos).

3. **Recall/Sensibilidad (0.7684)**:
   - Proporción de positivos reales detectados.
   - Valor alto: el modelo identifica correctamente ~77% de todas las cancelaciones tardías.
   - Fortaleza: Buena capacidad para detectar la mayoría de los casos problemáticos.

4. **Exactitud (0.6535)**:
   - Proporción total de predicciones correctas.
   - Valor moderado: ~65% de todas las reservas son clasificadas correctamente.
   - Contexto: Métrica menos informativa en problemas desbalanceados.

5. **ROC-AUC (0.7497)**:
   - Área bajo la curva ROC (discriminación).
   - Valor bueno: el modelo distingue razonablemente bien entre cancelaciones y no cancelaciones.
   - Interpretación: 0.75 indica discriminación aceptable (>0.5 aleatorio, 1.0 perfecto).

6. **PR-AUC (0.3332)**:
   - Área bajo la curva precisión-recall.
   - Más informativa que ROC-AUC para clases desbalanceadas.
   - Valor moderado-bajo: refleja la dificultad del problema y el desbalance.

**Balance y compromiso en las métricas:**
- El modelo prioriza la sensibilidad (recall) sobre la precisión.
- Configuración defensiva: prefiere identificar más cancelaciones potenciales, aun a costa de falsos positivos.
- Justificación empresarial: El costo de no detectar una cancelación tardía (pérdida de ingreso) podría ser mayor que el costo de un falso positivo (acciones preventivas innecesarias).

**Interpretación para el negocio:**
- El modelo puede ser útil para identificar reservas con alto riesgo de cancelación tardía.
- Limitada capacidad para determinar con certeza cuáles se cancelarán (precisión ~31%).
- Adecuado para dirigir acciones de prevención y mitigación, no para decisiones automáticas definitivas.
- Las probabilidades proporcionadas permiten ajustar el umbral según la estrategia (más preventiva o más selectiva).

## 3. Código de Inferencia y Despliegue (Versión Resumida)

### 3.1. Inferencia (inference.py)

El archivo `inference.py` implementa la lógica para aplicar el modelo entrenado a nuevos datos, con tres funciones principales:

1. **`load_model()`**: Carga el modelo serializado desde un archivo.
2. **`align_columns()`**: Asegura que los datos de entrada tengan la estructura esperada por el modelo.
3. **`make_predictions()`**: Genera y guarda predicciones de clase y probabilidad.

La implementación utiliza variables de entorno para las rutas de archivos, facilitando la configuración flexible.

### 3.2. Containerización (Dockerfile)

El `Dockerfile` define un contenedor liviano basado en Python 3.12-slim que:
- Instala las dependencias necesarias desde `requirements.txt`.
- Configura las rutas mediante variables de entorno.
- Ejecuta el script de inferencia al inicializarse.

Esta configuración proporciona un entorno aislado y reproducible para desplegar el modelo en producción.

## 4. Conclusiones y Recomendaciones

### 4.1. Evaluación General del Modelo

**Fortalezas del modelo baseline:**
- Capacidad predictiva moderada pero significativamente superior al azar (ROC-AUC ~0.75).
- Alta sensibilidad (~77%) permitiendo identificar la mayoría de cancelaciones tardías.
- Pipeline integral que maneja automáticamente todas las etapas de preprocesamiento.
- Interpretabilidad inherente al algoritmo de regresión logística.

**Limitaciones identificadas:**
- Precisión limitada (~31%) indicando alto número de falsos positivos.
- Valor moderado de F1-score (~0.45) reflejando el compromiso precisión-recall.
- PR-AUC modesto (~0.33) sugiriendo dificultad en el manejo del desbalance de clases.

### 4.2. Aplicaciones Prácticas

El modelo en su estado actual es apropiado para:
- **Identificación temprana de reservas de alto riesgo** para intervenciones preventivas.
- **Segmentación de cartera de reservas** por nivel de riesgo.
- **Estimación de ocupación ajustada por riesgo** para planificación operativa.
- **Punto de partida para modelos más complejos** y específicos.

### 4.3. Recomendaciones para Mejoras Futuras

1. **Refinamiento de características**:
   - Incorporar características de temporalidad (estacionalidad, día de semana).
   - Añadir características de comportamiento histórico del cliente.
   - Explorar interacciones entre variables (ej. lead_time × continente).

2. **Exploración de modelos alternativos**:
   - Árboles de decisión para capturar relaciones no lineales y umbrales naturales.
   - Ensamblados (Random Forest, Gradient Boosting) para mejorar el poder predictivo.
   - Modelos más específicos para diferentes segmentos (hoteles de lujo vs. económicos).

3. **Ajuste de estrategia de balanceo**:
   - Probar diferentes niveles de balanceo (no necesariamente 1:1).
   - Explorar técnicas alternativas como ADASYN o combinaciones de submuestreo y sobremuestreo.

4. **Optimización orientada al negocio**:
   - Ajustar el umbral de clasificación según el costo relativo de falsos positivos vs. falsos negativos.
   - Desarrollar métricas personalizadas que reflejen el impacto económico real de las predicciones.

5. **Operacionalización**:
   - Implementar una API REST para predicciones en tiempo real.
   - Añadir monitoreo de calidad del modelo y drift en producción.
   - Desarrollar estrategias de reentrenamiento periódico.

El modelo baseline establecido proporciona una sólida fundación sobre la cual construir soluciones más sofisticadas, manteniendo el equilibrio entre complejidad e interpretabilidad según evolucionen las necesidades del negocio.
