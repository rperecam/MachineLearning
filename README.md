# Resumen del Proyecto: Predicción de Cancelaciones Hoteleras

## Objetivo del Problema Detallado

El proyecto se centra en abordar el **problema crítico de las cancelaciones de reservas hoteleras** para una cadena específica con establecimientos en España, Francia y Portugal, que identifica esto como un desafío operativo y financiero significativo (fuente 157, 159). El **objetivo principal y específico** es desarrollar y evaluar modelos de Machine Learning capaces de **predecir, con la mayor fiabilidad posible, qué reservas confirmadas serán canceladas dentro de los últimos 30 días previos a la fecha de llegada** (`arrival_date`) (fuentes 163, 178). Este tipo de cancelaciones tardías son especialmente perjudiciales para la planificación y los ingresos (fuente 49).

El contexto implica utilizar datos históricos (reservas de 2016 hasta el 15 de junio de 2017) de los hoteles más antiguos de la cadena para construir un modelo que, eventualmente, pueda aplicarse a hoteles de apertura más reciente sobre los que no se tiene histórico de reservas (fuente 167). La finalidad última es dotar a la cadena hotelera de una herramienta predictiva que permita:

* **Optimizar la gestión de ingresos**: Ajustando estrategias como el overbooking de forma más informada y precisa (fuente 161).
* **Reducir pérdidas financieras y costos operativos**: Implementando acciones preventivas (ej. incentivos para confirmación, reasignación eficiente de habitaciones) al identificar con antelación las reservas con alta probabilidad de cancelación (fuente 162).
* **Mejorar la planificación general**: Facilitando una toma de decisiones más robusta y centrada en datos para fortalecer la posición competitiva (fuente 160).

## Datos Utilizados

Se emplean dos conjuntos de datos principales:

1.  **`hotels.csv`**: Contiene información estática y descriptiva de los 12 hoteles de la cadena (tipo, país, instalaciones como parking/restaurante/piscina, número de habitaciones, valoración media, etc.) a fecha 15 de junio de 2017 (fuentes 169-170).
2.  **`bookings_train.csv`**: Incluye un histórico parcial de reservas (de los hoteles antiguos) realizadas entre 2016 y el 15 de junio de 2017. Detalla información variada de cada reserva: hotel, tipo de alojamiento, segmento de mercado, canal de distribución, tipo de habitación, solicitudes especiales, noches de estancia, precio total (`rate`), número de huéspedes, fechas clave (reserva, llegada, último estado) y el estado final de la reserva (Check-Out, Canceled, No-Show, Booked) (fuentes 158, 171-176).

## Procesos Clave Implementados

El flujo de trabajo se estructura mediante scripts modulares y pipelines de Scikit-learn, abarcando:

1.  **Carga y Fusión de Datos**: Lectura inicial de los CSV y unión mediante `hotel_id` para consolidar la información por reserva (fuente 3, texto sección 3).
2.  **Ingeniería de Características**:
    * Creación de variables con significado de negocio: `lead_time` (días entre reserva y llegada, crítica para cancelaciones), `is_foreign` (viajero internacional vs. doméstico), `price_per_night`, `price_per_person`, `price_deviation` (comparación con precio medio del hotel), componentes temporales (mes/día de llegada/reserva), indicadores de temporada alta/fin de semana, categorías de duración de estancia, número de servicios (`num_assets`), interacciones entre variables (`lead_time * price`), etc. (fuente 14, texto sección 4).
    * Transformadores personalizados como `ContinentMapper` para agrupar países y reducir la dimensionalidad categórica (fuentes 5-13).
3.  **Definición Precisa de la Variable Objetivo (`target`)**: Se construye una variable binaria (`cancelled_last_30_days`) que toma valor 1 si `reservation_status` es 'Canceled' o 'No-Show' Y la cancelación ocurrió 30 días o menos antes de la `arrival_date`, y 0 en otro caso. Esto enfoca el modelo específicamente en las cancelaciones tardías de interés (fuentes 42-49, texto sección 4).
4.  **Preprocesamiento y Transformación de Datos (usando `ColumnTransformer`)**:
    * Limpieza y conversión de tipos (fechas a datetime).
    * **Imputación de Nulos**: Estrategias robustas como mediana (`SimpleImputer`) para numéricos, moda (`SimpleImputer`) para categóricos, o basadas en vecinos como `KNNImputer` (fuente 54-55, texto sección 5).
    * **Manejo de Outliers**: Recorte de valores extremos en variables numéricas mediante `OutlierCapper` (basado en IQR) para evitar su influencia desmedida (fuentes 27-41).
    * **Escalado Numérico**: `StandardScaler` para normalizar la escala de las variables numéricas (media 0, desviación 1) (fuente 54, texto sección 5).
    * **Codificación Categórica**: `OneHotEncoder` (con `handle_unknown='ignore'`) para convertir variables categóricas en formato numérico binario apto para los modelos (fuente 55, texto sección 5).
5.  **Manejo del Desbalanceo de Clases**: Dado que las cancelaciones suelen ser minoritarias, se aplican técnicas en el set de entrenamiento como SMOTE o combinaciones (ej. RandomUnderSampler + SMOTE) para generar un conjunto de datos más equilibrado (sin buscar necesariamente un 1:1) y ayudar al modelo a aprender mejor los patrones de la clase minoritaria (fuente 82, texto sección 6).
6.  **Selección de Características**: Empleo de métodos para reducir la dimensionalidad, eliminar ruido y mejorar la generalización:
    * Filtrado inicial basado en varianza (`VarianceThreshold`) para descartar features sin información (fuentes 68-73).
    * Selección supervisada basada en métricas estadísticas (`SelectKBest` con test ANOVA F) o en la importancia calculada por el propio modelo (`feature_importances_` en XGBoost), reteniendo solo las más predictivas (fuentes 74-81, texto sección 7, 8).
7.  **Entrenamiento, Optimización y Validación de Modelos**:
    * Implementación y comparación de: **Regresión Logística** (baseline), **Regresión Logística con SGD**, **XGBoost** individual y un **Ensemble (VotingClassifier)** de XGBoost.
    * Uso riguroso de **Validación Cruzada** (`StratifiedKFold` para mantener proporciones de clase, o `GroupKFold` por `hotel_id` para evitar fugas de datos entre folds del mismo hotel) durante la optimización y evaluación (fuente 99, texto sección 7).
    * **Optimización de Hiperparámetros**: Búsqueda sistemática (ej. `GridSearchCV`) de los mejores parámetros para cada modelo (regularización, profundidad de árboles, etc.) (fuente 109, texto sección 7, 8).
    * Uso de **Early Stopping** en XGBoost para prevenir el sobreajuste durante el entrenamiento monitorizando una métrica en un set de validación (texto sección 7, 8).
    * **Optimización del Umbral de Decisión**: Ajuste fino del umbral de probabilidad (por defecto 0.5) para clasificar como cancelación, buscando maximizar el F1-score en los datos de prueba (texto sección 7, 8).
8.  **Evaluación Rigurosa**: Medición del rendimiento con métricas clave: **F1-score** (principal métrica objetivo por balancear Precision/Recall en casos desbalanceados), Precision, Recall, Accuracy y ROC-AUC (fuentes 101, 107-119, texto sección 7, 8, 9).
9.  **Construcción y Uso de Pipelines**: Encapsulación de todo el flujo (preprocesamiento, transformadores, balanceo, selección, modelo) en objetos `Pipeline` de Scikit-learn o clases personalizadas. Esto asegura la reproducibilidad, consistencia entre entrenamiento/inferencia y facilita el despliegue (fuente 3, texto sección 10, fuente 186).
10. **Serialización y Preparación para Despliegue**:
    * Guardado del objeto pipeline final (el mejor modelo con su preprocesamiento asociado) en un archivo (`.pkl`) usando `pickle` (texto sección 9, fuente 190).
    * Creación de un `Dockerfile` para empaquetar el modelo y sus dependencias, permitiendo una inferencia batch portable y generalizable en cualquier entorno compatible con Docker (fuentes 131-133, 187-189, 195).

## Modelos Desarrollados

Se construyeron, entrenaron y compararon principalmente los siguientes enfoques:

* Un modelo **Baseline de Regresión Logística** (y su variante con SGD).
* Un modelo **XGBoost** individual, optimizado mediante ajuste de hiperparámetros y selección de características basada en importancia.
* Un modelo **Ensemble (VotingClassifier)** que combina las predicciones (soft voting) de varios modelos XGBoost con configuraciones ligeramente distintas para mejorar la robustez.

## Resultado Final

El entregable principal del proyecto es un **pipeline completo y serializado (`.pkl`)**. Este archivo contiene todos los pasos necesarios de preprocesamiento configurados y el modelo final entrenado (seleccionado como el mejor en base al F1-score). Está listo para ser cargado en otro entorno y utilizado directamente para generar predicciones (probabilidad y clase binaria) sobre si nuevas reservas hoteleras serán canceladas en los últimos 30 días antes de la llegada (texto sección 9, 11).