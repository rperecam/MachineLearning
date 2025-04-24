# Memoria del Proyecto: Predicción de Cancelaciones Anticipadas de Reservas de Hotel con XGBoost y Optimización Avanzada

## Índice

1. [Introducción: El Desafío de las Cancelaciones Anticipadas](#introducción-el-desafío-de-las-cancelaciones-anticipadas)
2. [Mirando Atrás: Lecciones Aprendidas del Primer Intento](#mirando-atrás-lecciones-aprendidas-del-primer-intento)
3. [El Nuevo Enfoque: Un Viaje Detallado Paso a Paso](#el-nuevo-enfoque-un-viaje-detallado-paso-a-paso)
    - 3.1. [Preparando el Terreno: Preprocesamiento y Limpieza de Datos](#preparando-el-terreno-preprocesamiento-y-limpieza-de-datos)
        - 3.1.1. [Carga y Fusión Inicial de Datos](#carga-y-fusión-inicial-de-datos)
        - 3.1.2. [Filtrado Estratégico y Definición del Target](#filtrado-estratégico-y-definición-del-target)
        - 3.1.3. [Limpieza de Datos: Nulos y Tratamiento Específico](#limpieza-de-datos-nulos-y-tratamiento-específico)
        - 3.1.4. [Ingeniería de Características (Feature Engineering)](#ingeniería-de-características-feature-engineering)
        - 3.1.5. [Manejo de Fechas y Descarte de Columnas](#manejo-de-fechas-y-descarte-de-columnas)
    - 3.2. [El Corazón del Problema: Manejo del Desbalanceo de Clases](#el-corazón-del-problema-manejo-del-desbalanceo-de-clases)
        - 3.2.1. [El Intento con RandomOverSampler y sus Limitaciones](#el-intento-con-randomoversampler-y-sus-limitaciones)
        - 3.2.2. [La Solución: SMOTE para Generación Sintética Inteligente](#la-solución-smote-para-generación-sintética-inteligente)
    - 3.3. [Validación Robusta: GroupKFold para Evitar Fugas](#validación-robusta-groupkfold-para-evitar-fugas)
        - 3.3.1. [El Error Común: Validación Cruzada Simple y sus Riesgos](#el-error-común-validación-cruzada-simple-y-sus-riesgos)
        - 3.3.2. [La Implementación Correcta: GroupKFold y la Agrupación por Hotel](#la-implementación-correcta-groupkfold-y-la-agrupación-por-hotel)
    - 3.4. [Puliendo el Modelo: Optimización de Hiperparámetros y Umbral](#puliendo-el-modelo-optimización-de-hiperparámetros-y-umbral)
        - 3.4.1. [Búsqueda Eficiente: RandomizedSearchCV vs. GridSearchCV](#búsqueda-eficiente-randomizedsearchcv-vs-gridsearchcv)
        - 3.4.2. [Ajuste Fino: Encontrando el Umbral Óptimo con F1-Score](#ajuste-fino-encontrando-el-umbral-óptimo-con-f1-score)
    - 3.5. [La Hora de la Verdad: Evaluación del Modelo Final](#la-hora-de-la-verdad-evaluación-del-modelo-final)
    - 3.6. [¿Más es Siempre Mejor? Descartando el Ensamblado de XGBoost](#más-es-siempre-mejor-descartando-el-ensamblado-de-xgboost)
    - 3.7. [Asegurando el Futuro: Guardado y Replicabilidad del Modelo](#asegurando-el-futuro-guardado-y-replicabilidad-del-modelo)
4. [Conclusiones: Un Modelo Robusto Nacido de la Iteración](#conclusiones-un-modelo-robusto-nacido-de-la-iteración)

---

## Introducción: El Desafío de las Cancelaciones Anticipadas

Este proyecto se centra en un problema clave para la gestión hotelera: la predicción de cancelaciones anticipadas de reservas. Definimos "anticipada" como una cancelación realizada con 30 días o más de antelación respecto a la fecha de llegada prevista. ¿Por qué es crucial predecir esto? Porque permite a los hoteles ajustar su inventario, optimizar precios y planificar recursos (personal, suministros, etc.) con mayor eficacia, reduciendo la incertidumbre y maximizando ingresos.

Para abordar este desafío, hemos desarrollado un modelo de Machine Learning basado en XGBoost. Pero no es solo el algoritmo lo que importa, sino todo el proceso que lo rodea. Hemos construido un pipeline completo que integra pasos esenciales:

- **Preprocesamiento Cuidadoso**: Limpieza, imputación y creación de nuevas variables (feature engineering).
- **Manejo del Desbalanceo**: Uso de SMOTE (Synthetic Minority Over-sampling Technique) para dar más peso a la clase minoritaria (las cancelaciones anticipadas) de forma inteligente.
- **Validación Cruzada Robusta**: Implementación de GroupKFold para evitar la "fuga de datos" (data leakage) y obtener una estimación realista del rendimiento del modelo, respetando la estructura agrupada de los datos (reservas por hotel).
- **Optimización de Hiperparámetros**: Búsqueda eficiente con RandomizedSearchCV.
- **Ajuste del Umbral de Decisión**: Selección de un umbral personalizado basado en el F1-Score para equilibrar precisión y sensibilidad, métricas clave en problemas desbalanceados.

Este enfoque no surgió de la nada. Es el resultado de un proceso iterativo, fuertemente influenciado por las lecciones aprendidas (y los errores cometidos) en un intento anterior. A continuación, detallaremos tanto esas lecciones como las mejoras específicas implementadas en esta versión.

---

## Mirando Atrás: Lecciones Aprendidas del Primer Intento

Antes de llegar a la solución actual, hubo una versión previa del proyecto. Analizar críticamente ese primer intento fue fundamental para entender qué debíamos mejorar. Los principales puntos débiles identificados fueron:

- **Validación Cruzada Deficiente**: El trabajo anterior mencionaba el uso de GridSearchCV pero sin una estrategia de validación cruzada (CV) adecuada para la naturaleza de los datos. Probablemente se usó una CV estándar (como K-Fold simple) que no tiene en cuenta agrupaciones naturales en los datos (como múltiples reservas pertenecientes al mismo hotel). Esto es peligroso porque puede llevar a sobreajuste (overfitting): el modelo aprende patrones específicos del conjunto de entrenamiento (incluyendo información "filtrada" del conjunto de validación si no se separa bien) y luego no generaliza bien a datos nuevos y no vistos. El resultado es una falsa sensación de buen rendimiento.

- **Manejo Incorrecto de los Datos**: Se cometieron dos errores importantes:
  - Se eliminaron registros con estado 'Booked' (reservas confirmadas y no canceladas). Esto eliminaba información valiosa sobre las reservas que no se cancelan, dificultando al modelo aprender la diferencia.
  - Se trataron los 'No-Shows' (clientes que no se presentan) como cancelaciones. Aunque un 'No-Show' es un resultado negativo para el hotel, no es una cancelación anticipada. Mezclar estos conceptos distorsiona el objetivo del modelo.

- **Preprocesamiento Simplista**: No se abordó adecuadamente el desbalanceo de clases (probablemente había muchas más reservas no canceladas o canceladas tardíamente que canceladas anticipadamente). Además, el flujo general de preparación de datos carecía de la robustez necesaria.

Este feedback fue una llamada de atención. Nos dimos cuenta de que no bastaba con aplicar un algoritmo potente; la clave estaba en la calidad de los datos, la metodología de validación y un enfoque riguroso en cada paso.

---

## El Nuevo Enfoque: Un Viaje Detallado Paso a Paso

Con las lecciones aprendidas en mente, rediseñamos el proceso por completo. A continuación, desglosamos cada etapa, explicando las decisiones tomadas, los problemas abordados y cómo se reflejan en el código.

### Preparando el Terreno: Preprocesamiento y Limpieza de Datos

Un buen modelo se construye sobre cimientos sólidos: datos limpios y bien preparados. Esta fase fue crucial y mucho más meticulosa que en el intento anterior.

#### Carga y Fusión Inicial de Datos

El proceso comienza cargando los dos conjuntos de datos principales: `hotels.csv` (información sobre los hoteles) y `bookings_train.csv` (datos de las reservas). La fusión se realiza usando `hotel_id` como clave para combinar la información de cada reserva con los detalles de su hotel correspondiente.

```python
# Código relevante en get_X_y()
hotels = pd.read_csv(os.environ.get("HOTELS_DATA_PATH", 'data/hotels.csv'))
bookings = pd.read_csv(os.environ.get("TRAIN_DATA_PATH", 'data/bookings_train.csv'))
data = pd.merge(bookings, hotels, on='hotel_id', how='left')
```

**Decisión/Análisis**: Usar `pd.merge` con `how='left'` asegura que mantenemos todas las reservas y añadimos la información del hotel correspondiente. Si un hotel no existiera en `hotels.csv` (poco probable en un dataset limpio), sus columnas quedarían con valores nulos, que trataríamos más adelante.

#### Filtrado Estratégico y Definición del Target

Aquí corregimos los errores del pasado. Primero, identificamos y tratamos los 'No Show' como datos no relevantes para este problema específico (predecir cancelaciones anticipadas), marcándolos como `NaN` y luego eliminándolos. También filtramos las reservas 'Booked', ya que nuestro objetivo es predecir un tipo específico de cancelación.

```python
# Código relevante en get_X_y()
data['reservation_status'] = data['reservation_status'].replace('No Show', np.nan)
data = data[data['reservation_status'].notna()].copy()
data = data[data['reservation_status'] != 'Booked'].copy()
```

Luego, definimos nuestra variable objetivo (target). Calculamos la diferencia en días entre la fecha de llegada y la fecha del último estado de la reserva (`reservation_status_date`). El target es 1 si la reserva fue 'Canceled' y esta diferencia es de 30 días o más; de lo contrario, es 0.

```python
# Código relevante en get_X_y()
# Convertir fechas primero (ver sección 3.1.5)
data['days_before_arrival'] = (data['arrival_date'] - data['reservation_status_date']).dt.days
data['target'] = ((data['reservation_status'] == 'Canceled') &
                  (data['days_before_arrival'] >= 30)).astype(int)
```

**Decisión/Análisis**: Este filtrado y definición precisa del target son cruciales. Aseguran que el modelo se entrene exactamente para el problema que queremos resolver: distinguir cancelaciones anticipadas (`target=1`) de otras cancelaciones (`target=0`, cancelaciones tardías).

#### Limpieza de Datos: Nulos y Tratamiento Específico

En lugar de eliminar filas con nulos indiscriminadamente (lo que podría causar pérdida de información valiosa), adoptamos una estrategia de imputación:

- Para variables numéricas, rellenamos los valores faltantes con la mediana de la columna. La mediana es más robusta a outliers que la media.
- Para variables categóricas (texto o categorías), rellenamos los nulos con la moda (el valor más frecuente) de la columna.

```python
# Código relevante en get_X_y()
# Bucle para imputar numéricas con mediana
for col in data.select_dtypes(include=['int64', 'float64']).columns:
    if data[col].isna().any():
        data[col] = data[col].fillna(data[col].median())

# Bucle para imputar categóricas con moda
for col in data.select_dtypes(include=['object', 'category']).columns:
    # Asegurarse de no imputar columnas que se van a eliminar
    if col not in columns_to_drop and data[col].isna().any():
        data[col] = data[col].fillna(data[col].mode()[0])
```

**Decisión/Análisis**: La imputación es preferible a la eliminación cuando los datos faltantes no son excesivos. Elegir mediana y moda es una práctica estándar y robusta. Esto contrasta con el manejo inadecuado del trabajo anterior, preservando más datos para el entrenamiento.

#### Ingeniería de Características (Feature Engineering)

No nos conformamos con las variables originales. Creamos nuevas características que podrían capturar patrones útiles para la predicción:

- `lead_time`: Días entre la reserva y la llegada. Intuitivamente, reservas hechas con mucha antelación podrían tener más probabilidad de cancelación.
- `is_high_season`: Indicador binario (1 o 0) si la llegada es en junio, julio, agosto o diciembre. La estacionalidad afecta la demanda y posiblemente las cancelaciones.
- `is_weekend_arrival`: Indicador binario si la llegada es viernes o sábado. El comportamiento de reserva puede diferir los fines de semana.
- `price_per_night`: Precio promedio por noche. Podría correlacionarse con el tipo de cliente o la "seriedad" de la reserva.
- `stay_duration_category`: Duración de la estancia agrupada en categorías (ej. 1 noche, 2-3 noches, etc.). Estancias muy largas o muy cortas podrían tener patrones de cancelación distintos.
- `has_special_requests`: Indicador binario si el cliente hizo alguna petición especial. Podría indicar un mayor compromiso o una necesidad específica.

```python
# Código relevante en get_X_y()
data['lead_time'] = (data['arrival_date'] - data['booking_date']).dt.days
data['is_high_season'] = data['arrival_date'].dt.month.isin([6, 7, 8, 12]).astype(int)
data['is_weekend_arrival'] = data['arrival_date'].dt.dayofweek.isin([4, 5]).astype(int) # Viernes=4, Sábado=5
data['price_per_night'] = data['rate'] / np.maximum(data['stay_nights'], 1) # Evita división por cero
data['stay_duration_category'] = pd.cut(data['stay_nights'],
                                        bins=[-1, 1, 3, 7, 14, float('inf')],
                                        labels=['1_night', '2-3_nights', '4-7_nights', '8-14_nights', '15+_nights'])
data['has_special_requests'] = (data['special_requests'] > 0).astype(int)
```

**Decisión/Análisis**: El feature engineering es un arte. Estas características se basan en el conocimiento del dominio (hostelería) y buscan extraer señales útiles de los datos existentes. Podrían ser cruciales para que el modelo encuentre patrones que no son obvios en las variables originales.

#### Manejo de Fechas y Descarte de Columnas

Las columnas de fecha (`arrival_date`, `booking_date`, `reservation_status_date`) se convierten explícitamente al formato datetime de pandas. Esto es necesario para poder realizar operaciones con ellas, como calcular `lead_time` o `days_before_arrival`.

```python
# Código relevante en get_X_y()
for col in ['arrival_date', 'booking_date', 'reservation_status_date']:
    if col in data.columns:
        data[col] = pd.to_datetime(data[col])
```

Finalmente, eliminamos columnas que no deben entrar al modelo. Esto incluye las columnas de fecha originales (ya usamos la información que contenían), la variable `reservation_status` (que usamos para crear el target), `days_before_arrival` (directamente relacionada con el target, causaría fuga de datos), `special_requests` (usamos `has_special_requests` en su lugar), y `stay_nights` (usamos `stay_duration_category`). También se elimina `hotel_id` de las características (`X`) antes de entrenar, aunque la guardamos por separado (`groups`) para usarla en `GroupKFold`.

```python
# Código relevante en get_X_y()
columns_to_drop = [
    'reservation_status', 'reservation_status_date', 'days_before_arrival',
    'arrival_date', 'booking_date', 'special_requests', 'stay_nights',
    # hotel_id se elimina de X pero se guarda en groups
]
columns_to_drop = [col for col in columns_to_drop if col in data.columns] # Para evitar errores si alguna no existe
X = data.drop(columns=['target'] + columns_to_drop)
y = data['target']
groups = data['hotel_id'].copy()

# ... más tarde, antes de entrenar ...
# Código relevante en __main__
if 'hotel_id' in X_dev.columns:
    X_dev = X_dev.drop(columns=['hotel_id'])
    X_test = X_test.drop(columns=['hotel_id'])
```

**Decisión/Análisis**: Eliminar estas columnas es vital para evitar la fuga de datos (data leakage) – dar al modelo información que no tendría en el momento de hacer una predicción real. Por ejemplo, `days_before_arrival` solo se conoce después de que la cancelación (o no) ocurra. `hotel_id` se quita de `X` porque no queremos que el modelo memorice el comportamiento de hoteles específicos, sino que aprenda patrones generales; sin embargo, es esencial para `GroupKFold`.

### El Corazón del Problema: Manejo del Desbalanceo de Clases

En muchos problemas reales, como la detección de fraude o el diagnóstico médico, la clase de interés (fraude, enfermedad, o en nuestro caso, cancelación anticipada) es mucho menos frecuente que la clase normal. Esto es el desbalanceo de clases. Si no se trata, el modelo tiende a ignorar la clase minoritaria y simplemente predecir siempre la mayoritaria, logrando una alta exactitud (accuracy) pero siendo inútil en la práctica.

#### El Intento con RandomOverSampler y sus Limitaciones

En el trabajo previo (y quizás en una fase inicial de este), se probó `RandomOverSampler`. Esta técnica simplemente duplica aleatoriamente muestras de la clase minoritaria hasta igualar a la mayoritaria.

**Problema**: Aunque equilibra las clases, lo hace de forma "ingenua". Al duplicar exactamente las mismas muestras, no aporta información nueva real y aumenta significativamente el riesgo de sobreajuste a esas muestras específicas de la clase minoritaria. El modelo puede volverse muy bueno reconociendo esas instancias duplicadas, pero no generalizar bien a nuevas instancias minoritarias ligeramente diferentes.

#### La Solución: SMOTE para Generación Sintética Inteligente

Decidimos usar SMOTE (Synthetic Minority Over-sampling Technique). SMOTE es más sofisticado: en lugar de duplicar, crea muestras sintéticas de la clase minoritaria. Lo hace seleccionando una muestra minoritaria y buscando sus vecinos más cercanos (también minoritarios). Luego, genera una nueva muestra sintética en algún punto del segmento de línea que une la muestra original con uno de sus vecinos.

```python
# Código relevante en get_pipeline()
from imblearn.pipeline import Pipeline as ImbPipeline # Importante usar el Pipeline de imblearn
from imblearn.over_sampling import SMOTE

pipeline = ImbPipeline(steps=[
    ("preprocessor", preprocessor), # Primero preprocesar
    ("smote", SMOTE(random_state=42)), # Luego aplicar SMOTE SOLO a datos de entrenamiento (automático con pipeline)
    ("classifier", classifier) # Finalmente, entrenar el clasificador
])
```

**Decisión/Análisis**: SMOTE es generalmente preferible a `RandomOverSampler` porque introduce variabilidad y crea "puentes" entre las muestras minoritarias existentes, ayudando al clasificador a definir regiones de decisión más robustas para la clase minoritaria. Es crucial aplicarlo dentro del pipeline de validación cruzada (usando `ImbPipeline` de `imblearn`) para que SMOTE solo "vea" los datos de entrenamiento de cada fold, evitando así la fuga de datos del conjunto de validación hacia el proceso de oversampling.

### Validación Robusta: GroupKFold para Evitar Fugas

Evaluar el modelo correctamente es tan importante como entrenarlo bien. Aquí es donde corregimos uno de los errores más graves del trabajo anterior.

#### El Error Común: Validación Cruzada Simple y sus Riesgos

Usar una validación cruzada estándar (como `KFold` o `StratifiedKFold`) en datos donde existen grupos inherentes (reservas del mismo hotel) es problemático. Estas técnicas dividen los datos aleatoriamente (o estratificadamente por clase), lo que significa que reservas del mismo hotel podrían acabar tanto en el conjunto de entrenamiento como en el de validación dentro del mismo fold.

**Problema**: El modelo podría aprender características específicas de un hotel durante el entrenamiento y luego ser evaluado positivamente en otras reservas del mismo hotel en el conjunto de validación. Esto infla artificialmente las métricas de rendimiento porque el modelo está, en parte, reconociendo patrones específicos del grupo (hotel) en lugar de generalizar patrones aplicables a cualquier hotel. Es una forma sutil de fuga de datos.

#### La Implementación Correcta: GroupKFold y la Agrupación por Hotel

Para evitar esto, implementamos `GroupKFold`. Esta estrategia de validación cruzada asegura que todas las reservas pertenecientes a un mismo grupo (en nuestro caso, un `hotel_id`) permanezcan juntas, es decir, o todas en el conjunto de entrenamiento o todas en el de validación para un fold determinado. Nunca se dividen las reservas de un mismo hotel entre ambos conjuntos.

```python
# Código relevante en __main__
from sklearn.model_selection import GroupKFold

# ... obtener X_dev, y_dev, groups_dev ...
# 'groups_dev' contiene el hotel_id de cada reserva en el conjunto de desarrollo

group_cv = GroupKFold(n_splits=5) # Crear el objeto GroupKFold

# Pasarlo al RandomizedSearchCV
random_search = RandomizedSearchCV(
    pipe,
    param_distributions=param_dist,
    n_iter=10,
    # Aquí se pasa el generador de splits de GroupKFold
    cv=group_cv.split(X_dev, y_dev, groups_dev),
    scoring=scoring,
    refit='f1',
    # ... otros parámetros ...
)

# Imprimir distribución para verificar (opcional pero útil)
print("\nDistribución de folds con GroupKFold:")
for fold_idx, (train_idx, val_idx) in enumerate(group_cv.split(X_dev, y_dev, groups_dev)):
    print(f"  Fold {fold_idx + 1}: Training: {len(train_idx)} muestras, Validación: {len(val_idx)} muestras")
```

**Decisión/Análisis**: Usar `GroupKFold` con `hotel_id` como grupo es fundamental para obtener una estimación realista y fiable del rendimiento del modelo en hoteles completamente nuevos (no vistos durante el entrenamiento de ese fold). Previene la fuga de datos relacionada con la estructura agrupada y nos da más confianza en que el modelo generalizará bien.

### Puliendo el Modelo: Optimización de Hiperparámetros y Umbral

Una vez que tenemos un pipeline robusto y una estrategia de validación fiable, podemos buscar la mejor configuración para nuestro modelo XGBoost.

#### Búsqueda Eficiente: RandomizedSearchCV vs. GridSearchCV

Los modelos como XGBoost tienen muchos hiperparámetros (parámetros que no se aprenden de los datos, sino que se configuran antes del entrenamiento, como `max_depth`, `learning_rate`, etc.). Encontrar la combinación óptima puede mejorar significativamente el rendimiento.

- `GridSearchCV` prueba todas las combinaciones posibles de los hiperparámetros especificados. Es exhaustivo pero computacionalmente muy costoso, especialmente si el espacio de búsqueda es grande.
- `RandomizedSearchCV` prueba un número fijo (`n_iter`) de combinaciones aleatorias dentro de los rangos especificados para cada hiperparámetro.

```python
# Código relevante en __main__
from sklearn.model_selection import RandomizedSearchCV

# Definir el espacio de búsqueda (distribuciones o listas de valores)
param_dist = {
    'classifier__max_depth': [3, 4, 5],
    'classifier__min_child_weight': [3, 5],
    'classifier__subsample': [0.8, 0.9],
    'classifier__colsample_bytree': [0.7, 0.9],
    'classifier__learning_rate': [0.05], # Podría ser una distribución si quisiéramos más aleatoriedad
    'classifier__n_estimators': [50, 100] # Número de árboles
}

# Configurar RandomizedSearchCV
random_search = RandomizedSearchCV(
    pipe, # Nuestro pipeline completo (preproc + smote + xgb)
    param_distributions=param_dist,
    n_iter=10, # Número de combinaciones a probar
    cv=group_cv.split(X_dev, y_dev, groups_dev), # Usando GroupKFold
    scoring={'f1': make_scorer(f1_score)}, # Métrica a optimizar
    refit='f1', # Re-entrenar el mejor modelo en todo X_dev con los mejores params
    random_state=42, # Para reproducibilidad
    n_jobs=-1, # Usar todos los cores disponibles
    verbose=1 # Mostrar progreso
)

best_model = random_search.fit(X_dev, y_dev).best_estimator_
print(f"Mejores parámetros: {random_search.best_params_}")
```

**Decisión/Análisis**: Elegimos `RandomizedSearchCV` principalmente por eficiencia computacional. A menudo, encuentra combinaciones de hiperparámetros muy buenas (o incluso óptimas) en mucho menos tiempo que `GridSearchCV`, especialmente cuando el número de hiperparámetros a ajustar es elevado. Fijamos `n_iter=10` como un compromiso entre exploración y tiempo. La métrica clave para la optimización (`scoring` y `refit`) es el F1-Score, ideal para problemas desbalanceados.

#### Ajuste Fino: Encontrando el Umbral Óptimo con F1-Score

Los clasificadores binarios como XGBoost suelen devolver una probabilidad (entre 0 y 1) de pertenencia a la clase positiva (en nuestro caso, cancelación anticipada). Por defecto, se usa un umbral de 0.5 para decidir: si la probabilidad es >= 0.5, se predice 1; si no, 0. Sin embargo, este umbral de 0.5 no siempre es el óptimo, especialmente en clases desbalanceadas. Un umbral diferente puede mejorar el equilibrio entre precisión (de las predicciones positivas, cuántas son correctas) y sensibilidad/recall (de todos los positivos reales, cuántos se identificaron).

Creamos una función `find_optimal_threshold` que prueba varios umbrales (entre 0.3 y 0.7 en este caso) sobre las predicciones de probabilidad en el conjunto de desarrollo (`X_dev`) y elige el umbral que maximiza el F1-Score en esos datos.

```python
# Función find_optimal_threshold
def find_optimal_threshold(y_true, y_pred_proba):
    thresholds = np.linspace(0.3, 0.7, 40) # Probar 40 umbrales entre 0.3 y 0.7
    best_threshold, best_f1 = 0.5, 0 # Inicializar

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int) # Aplicar umbral
        f1 = f1_score(y_true, y_pred) # Calcular F1
        if f1 > best_f1: # Si mejora, actualizar
            best_f1 = f1
            best_threshold = threshold

    return best_threshold

# Código relevante en __main__
# Entrenar el mejor modelo encontrado por random_search en todo X_dev
final_model = best_model.fit(X_dev, y_dev)
# Obtener probabilidades en X_dev
y_dev_proba = final_model.predict_proba(X_dev)[:, 1]
# Encontrar el mejor umbral usando y_dev y y_dev_proba
threshold = find_optimal_threshold(y_dev, y_dev_proba)
print(f"\nUmbral óptimo: {threshold:.4f}")

# Aplicar este umbral a las predicciones en el conjunto de TEST
y_test_proba = final_model.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_proba >= threshold).astype(int)
```

**Decisión/Análisis**: Ajustar el umbral es un paso de optimización crucial post-entrenamiento. Al maximizar el F1-Score (que es la media armónica de precisión y recall), buscamos el mejor compromiso posible entre no etiquetar erróneamente las no-cancelaciones como cancelaciones (precisión) y encontrar la mayor cantidad posible de cancelaciones anticipadas reales (recall). Este umbral optimizado se calcula sobre el conjunto de desarrollo (`X_dev`, `y_dev`) y luego se aplica al conjunto de test (`X_test`) para la evaluación final.

### La Hora de la Verdad: Evaluación del Modelo Final

Con el modelo entrenado (usando los mejores hiperparámetros) y el umbral óptimo determinado, llega el momento de evaluar su rendimiento en un conjunto de datos que no ha visto nunca antes: el conjunto de test (`X_test`, `y_test`), que se separó al principio.

Calculamos un conjunto estándar de métricas de clasificación:

- **Exactitud (Accuracy)**: Porcentaje total de predicciones correctas. Engañosa en problemas desbalanceados.
- **Precisión (Precision)**: De todas las predicciones "Cancelación Anticipada", ¿qué porcentaje fue correcto? (`TP / (TP + FP)`)
- **Sensibilidad (Recall/True Positive Rate)**: De todas las cancelaciones anticipadas reales, ¿qué porcentaje detectó el modelo? (`TP / (TP + FN)`)
- **F1-Score**: Media armónica de Precisión y Recall. Buen indicador general para clases desbalanceadas. (`2 * (Precision * Recall) / (Precision + Recall)`)
- **ROC AUC**: Área bajo la curva ROC. Mide la capacidad general del modelo para distinguir entre las dos clases, independientemente del umbral elegido. Un valor cercano a 1 es excelente.

```python
# Código relevante en __main__
# y_test_pred se obtuvo usando el umbral óptimo
print("\nMétricas en conjunto de test:")
print(f"Exactitud:    {accuracy_score(y_test, y_test_pred):.4f}")
print(f"Precisión:    {precision_score(y_test, y_test_pred):.4f}")
print(f"Sensibilidad: {recall_score(y_test, y_test_pred):.4f}")
print(f"F1-Score:     {f1_score(y_test, y_test_pred):.4f}")
# Para ROC AUC usamos las probabilidades, no las predicciones binarizadas
print(f"ROC AUC:      {roc_auc_score(y_test, y_test_proba):.4f}")
```

**Resultados Obtenidos**:

- **Exactitud**: 93.69%
- **Precisión**: 71.03%
- **Sensibilidad**: 91.30%
- **F1-Score**: 79.90%
- **ROC AUC**: 97.90%

**Decisión/Análisis**: Los resultados son muy prometedores. La alta Sensibilidad (91.30%) indica que el modelo detecta la gran mayoría de las cancelaciones anticipadas reales. La Precisión (71.03%) es decente, significando que cuando el modelo predice una cancelación anticipada, acierta aproximadamente 7 de cada 10 veces. El F1-Score (79.90%) confirma un buen equilibrio entre ambas. El ROC AUC (97.90%) es excelente, sugiriendo que el modelo tiene una capacidad discriminativa muy alta entre clases. Aunque la precisión podría mejorarse, la alta sensibilidad es a menudo prioritaria en problemas como este (es preferible investigar unas pocas falsas alarmas que perder muchas cancelaciones reales).

### ¿Más es Siempre Mejor? Descartando el Ensamblado de XGBoost

Durante el desarrollo, exploramos la idea de un modelo ensamblador, combinando las predicciones de tres clasificadores XGBoost entrenados quizás con ligeras variaciones (datos, parámetros). La hipótesis era que combinar varios modelos podría dar un resultado más robusto y preciso que uno solo.

**Experimento**: Se implementó y probó este enfoque.

**Resultado**: No se observó una mejora significativa en la métrica clave (F1-Score) que justificara la complejidad añadida y, sobre todo, el aumento considerable en el tiempo de entrenamiento y predicción.

**Decisión Tomada**: Se decidió descartar el modelo ensamblador. A veces, la simplicidad es preferible si la complejidad adicional no aporta un beneficio claro. Optamos por centrarnos en refinar el pipeline del modelo único, asegurando la calidad de los datos, la validación y la optimización.

### Asegurando el Futuro: Guardado y Replicabilidad del Modelo

Un modelo solo es útil si se puede guardar, cargar y usar para hacer predicciones sobre datos nuevos. Además, es fundamental asegurar la replicabilidad de todo el pipeline.

Utilizamos `cloudpickle` para guardar el objeto completo que incluye no solo el clasificador XGBoost entrenado, sino también todos los pasos de preprocesamiento (imputación, escalado, one-hot encoding) y el paso de SMOTE integrados en el `ImbPipeline`. Además, envolvemos este pipeline en una clase personalizada `ThresholdClassifier` que almacena el umbral óptimo encontrado y lo aplica automáticamente al método `predict`.

```python
# Clase auxiliar para encapsular el umbral
class ThresholdClassifier:
    def __init__(self, classifier, threshold=0.5):
        self.classifier = classifier # El pipeline entrenado
        self.threshold = threshold # El umbral óptimo

    def predict(self, X):
        # Usa el umbral personalizado para la predicción binaria
        return (self.classifier.predict_proba(X)[:, 1] >= self.threshold).astype(int)

    def predict_proba(self, X):
        # Devuelve las probabilidades originales del clasificador base
        return self.classifier.predict_proba(X)

# Código relevante en __main__ y save_pipeline()
import cloudpickle
import os

def save_pipeline(pipe):
    model_path = os.environ.get("MODEL_PATH", "models/xgboost_model.pkl")
    os.makedirs(os.path.dirname(model_path), exist_ok=True) # Crear directorio si no existe
    with open(model_path, mode="wb") as f:
        cloudpickle.dump(pipe, f) # Guardar el objeto completo
    print(f"Modelo guardado en {model_path}")

# ... después de encontrar el umbral ...
threshold_classifier = ThresholdClassifier(final_model, threshold) # Crear instancia con el modelo y umbral
save_pipeline(threshold_classifier) # Guardar este objeto wrapper
```

**Decisión/Análisis**: Guardar el pipeline completo con `cloudpickle` (que es más robusto que el `pickle` estándar para objetos complejos como los pipelines de `scikit-learn/imblearn`) es esencial. Asegura que cuando carguemos el modelo para hacer nuevas predicciones, se apliquen exactamente los mismos pasos de preprocesamiento y el mismo umbral de decisión que se usaron durante la evaluación. Esto garantiza la consistencia y la replicabilidad del flujo de trabajo.

---

## Conclusiones: Un Modelo Robusto Nacido de la Iteración

El modelo final para predecir cancelaciones anticipadas de hotel representa una mejora sustancial respecto a intentos anteriores. Este progreso se basa en una serie de decisiones metodológicas clave, aprendidas a través de la experiencia y la corrección de errores:

- **La Base es Todo**: Un preprocesamiento meticuloso, incluyendo la ingeniería de características relevantes y una imputación sensata de valores nulos, sentó las bases para un modelo más informativo.
- **Equilibrio Inteligente**: Abordar el desbalanceo de clases con SMOTE en lugar de `RandomOverSampler` permitió equilibrar las clases sin simplemente duplicar información, reduciendo el riesgo de sobreajuste.
- **Validación Honesta**: La implementación de `GroupKFold` fue, quizás, la mejora metodológica más crítica. Aseguró una evaluación realista del rendimiento del modelo al prevenir la fuga de datos inherente a la estructura agrupada de las reservas por hotel.
- **Optimización Dirigida**: El uso de `RandomizedSearchCV` permitió una búsqueda eficiente de hiperparámetros, y la optimización del umbral de decisión basada en el F1-Score afinó el modelo para el equilibrio deseado entre precisión y sensibilidad en nuestro contexto de clases desbalanceadas.
- **Enfoque Pragmático**: Se descartaron enfoques más complejos (como el ensamblado) cuando no ofrecieron beneficios claros, favoreciendo un pipeline robusto y eficiente.
- **Replicabilidad**: Guardar el pipeline completo con `cloudpickle`, incluyendo el umbral personalizado, garantiza que el modelo se pueda desplegar y utilizar de forma consistente.

Los resultados finales (F1-Score de ~80%, ROC AUC de ~98%) demuestran la eficacia del enfoque. Este proyecto subraya la importancia de no solo elegir un buen algoritmo, sino de construir cuidadosamente todo el flujo de trabajo, desde la preparación de los datos hasta la validación y el ajuste final, aprendiendo de los errores y refinando el proceso iterativamente.