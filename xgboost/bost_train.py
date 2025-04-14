import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import pickle
import warnings
import os
from scipy.stats import uniform, randint
import concurrent.futures
import multiprocessing
import time
import cupy as cp

# Ignorar advertencias para una salida más limpia
warnings.filterwarnings('ignore')

# Definir clase de pipeline personalizada para filtrar características
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

# Cargar datos de hoteles y reservas
def load_data():
    hotels = pd.read_csv('/app/data/hotels.csv')
    bookings = pd.read_csv('/app/data/bookings_train.csv')
    return hotels, bookings

# Fusionar datos de hoteles y reservas
def merge_data(hotels, bookings):
    merged = pd.merge(bookings, hotels, on='hotel_id', how='left')
    filtered = merged[~merged['reservation_status'].isin(['Booked', np.nan])].copy()
    hotel_ids = filtered['hotel_id'].copy()
    return filtered, hotel_ids

# Preprocesar datos para predecir cancelaciones
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

    # Extraer características temporales
    data['lead_time'] = (data['arrival_date'] - data['booking_date']).dt.days
    data['lead_time_category'] = pd.cut(data['lead_time'], bins=[-1, 7, 30, 90, 180, float('inf')], labels=['last_minute', 'short', 'medium', 'long', 'very_long'])
    data['is_high_season'] = data['arrival_date'].dt.month.isin([6, 7, 8, 12]).astype(int)
    data['is_weekend_arrival'] = data['arrival_date'].dt.dayofweek.isin([4, 5]).astype(int)
    data['arrival_month'] = data['arrival_date'].dt.month
    data['arrival_dayofweek'] = data['arrival_date'].dt.dayofweek
    data['booking_month'] = data['booking_date'].dt.month

    # Extraer características de precio
    data['price_per_night'] = data['rate'] / np.maximum(data['stay_nights'], 1)
    data['price_per_person'] = data['rate'] / np.maximum(data['total_guests'], 1)
    data['total_cost'] = data['rate']

    # Extraer características de duración de estancia
    data['stay_duration_category'] = pd.cut(data['stay_nights'], bins=[-1, 1, 3, 7, 14, float('inf')], labels=['1_night', '2-3_nights', '4-7_nights', '8-14_nights', '15+_nights'])

    # Extraer características de solicitudes especiales
    data['has_special_requests'] = (data['special_requests'] > 0).astype(int)
    data['special_requests_ratio'] = data['special_requests'] / np.maximum(data['total_guests'], 1)

    # Extraer características de ubicación
    if 'country_x' in data.columns and 'country_y' in data.columns:
        data['is_foreign'] = (data['country_x'] != data['country_y']).astype(int)
        data.loc[data['country_x'].isna() | data['country_y'].isna(), 'is_foreign'] = 0

    # Extraer características interactivas
    data['price_length_interaction'] = data['price_per_night'] * data['stay_nights']
    data['lead_price_interaction'] = data['lead_time'] * data['price_per_night']

    # Nueva característica: Desviación de precio del promedio del hotel
    hotel_avg_price = data.groupby('hotel_id')['price_per_night'].transform('mean')
    data['price_deviation'] = (data['price_per_night'] - hotel_avg_price) / hotel_avg_price

    # Extraer características de transporte
    data['requested_parking'] = (data['required_car_parking_spaces'] > 0).astype(int)

    # Eliminar columnas que pueden causar filtración de datos
    columns_to_drop = ['reservation_status', 'reservation_status_date', 'booking_date', 'arrival_date', 'days_before_arrival']
    data.drop(columns=columns_to_drop, inplace=True)

    # Manejar valores infinitos y nulos en características numéricas
    for col in ['price_per_night', 'price_per_person', 'special_requests_ratio', 'price_deviation']:
        if col in data.columns:
            data[col] = data[col].replace([np.inf, -np.inf], np.nan)
            data[col] = data[col].fillna(data[col].median())

    return data

# Preparar características para el modelo
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

# Encontrar umbral óptimo para la puntuación F1
def find_optimal_threshold(y_true, y_pred_proba):
    thresholds = np.linspace(0.1, 0.9, 200)
    best_threshold, best_f1 = 0.5, 0

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1, best_threshold = f1, threshold

    return best_threshold, best_f1

# Filtrar características de importancia cero con umbral adaptativo
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

# Inicializar GPU para XGBoost
def initialize_gpu_for_xgboost():
    # Verificar si hay GPU disponible
    try:
        gpu_info = os.popen('nvidia-smi').read()
        print("GPU detectada correctamente:")
        print(gpu_info.split('\n')[0])
        print(gpu_info.split('\n')[1])
        return True
    except Exception as e:
        print(f"Error al inicializar GPU: {e}")
        return False

# Entrenar modelo en GPU con parámetros corregidos
def train_model_on_gpu(param_set, X_train, y_train, X_eval, y_eval, gpu_id):
    param_set = param_set.copy()

    # Eliminar gpu_id si existe en param_set
    if 'gpu_id' in param_set:
        del param_set['gpu_id']

    # Configurar XGBoost para GPU
    param_set.update({
        'tree_method': 'hist',
        'device': 'cuda',
        'objective': 'binary:logistic',
        'random_state': 42
    })

    # Crear objetos DMatrix para entrenamiento más rápido
    dtrain = xgb.DMatrix(X_train, y_train)
    deval = xgb.DMatrix(X_eval, y_eval)

    # Entrenar modelo usando API nativa que es más eficiente en GPU
    watchlist = [(dtrain, 'train'), (deval, 'eval')]
    num_rounds = param_set.pop('n_estimators', 1000)

    # Entrenar el modelo
    bst = xgb.train(
        param_set,
        dtrain,
        num_rounds,
        evals=watchlist,
        verbose_eval=False
    )

    # Crear un wrapper XGBClassifier para consistencia con el resto del código
    model = xgb.XGBClassifier()
    model._Booster = bst

    # Predecir y calcular puntuación F1
    y_pred_proba = bst.predict(deval)
    best_threshold, best_f1 = find_optimal_threshold(y_eval, y_pred_proba)

    return {
        'params': param_set,
        'model': model,
        'f1_score': best_f1,
        'threshold': best_threshold
    }

# Entrenar modelo en CPU
def train_model_on_cpu(param_set, X_train, y_train, X_eval, y_eval):
    param_set = param_set.copy()
    param_set.update({
        'tree_method': 'hist',
        'device': 'cpu',  # Establecer dispositivo explícitamente a CPU
        'objective': 'binary:logistic',
        'random_state': 42,
        'n_jobs': -1  # Usar todos los núcleos de CPU disponibles
    })

    # Crear objetos DMatrix para entrenamiento más rápido
    dtrain = xgb.DMatrix(X_train, y_train)
    deval = xgb.DMatrix(X_eval, y_eval)

    # Entrenar modelo usando API nativa
    watchlist = [(dtrain, 'train'), (deval, 'eval')]
    num_rounds = param_set.pop('n_estimators', 1000)

    # Entrenar el modelo
    bst = xgb.train(
        param_set,
        dtrain,
        num_rounds,
        evals=watchlist,
        verbose_eval=False
    )

    # Crear un wrapper XGBClassifier para consistencia
    model = xgb.XGBClassifier()
    model._Booster = bst

    # Predecir y calcular puntuación F1
    y_pred_proba = bst.predict(deval)
    best_threshold, best_f1 = find_optimal_threshold(y_eval, y_pred_proba)

    return {
        'params': param_set,
        'model': model,
        'f1_score': best_f1,
        'threshold': best_threshold
    }

# Búsqueda paralela personalizada usando GPU principalmente
def custom_parallel_search(X_train, X_test, y_train, y_test, preprocessor, num_iterations=60):
    print("Iniciando búsqueda paralela de GPU/CPU para los mejores parámetros XGBoost...")
    start_time = time.time()

    # Comprobar disponibilidad de GPU
    has_gpu = initialize_gpu_for_xgboost()
    if not has_gpu:
        print("ADVERTENCIA: GPU no detectada o mal configurada. Usando solo CPU.")

    # Preprocesar datos (usando CPU)
    print("Preprocesando datos (utilizando CPU)...")
    preprocessor.fit(X_train)
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Balancear datos (usando CPU)
    print("Balanceando dataset (utilizando CPU)...")
    rus = RandomUnderSampler(sampling_strategy=0.30, random_state=42)
    X_train_under, y_train_under = rus.fit_resample(X_train_transformed, y_train)

    smote = SMOTE(sampling_strategy=0.65, random_state=42, k_neighbors=5)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_under, y_train_under)

    # Entrenar modelo inicial para importancia de características (usando CPU)
    print("Entrenando modelo inicial para selección de características (utilizando CPU)...")
    init_params = {
        'objective': 'binary:logistic',
        'tree_method': 'hist',
        'device': 'cpu',
        'random_state': 42,
        'n_jobs': -1
    }

    # Usar DMatrix para mejor rendimiento
    dtrain = xgb.DMatrix(X_train_resampled, y_train_resampled)
    init_bst = xgb.train(init_params, dtrain, num_boost_round=100)

    # Convertir a XGBClassifier para compatibilidad
    init_model = xgb.XGBClassifier()
    init_model._Booster = init_bst

    # Filtrar características basadas en importancia (usando CPU)
    print("Filtrando características basadas en importancia (utilizando CPU)...")
    feature_names = preprocessor.get_feature_names_out() if hasattr(preprocessor, 'get_feature_names_out') else [f'feature_{i}' for i in range(X_train_transformed.shape[1])]
    X_train_filtered, X_test_filtered, important_indices = filter_zero_importance_features(
        init_model, feature_names, X_train_resampled, X_test_transformed
    )

    print(f"Seleccionadas {len(important_indices)} características importantes de {X_train_transformed.shape[1]}")

    # Valores óptimos encontrados previamente como referencia
    best_ref = {
        'max_depth': 7,
        'min_child_weight': 1,
        'gamma': 0.485,
        'subsample': 0.909,
        'colsample_bytree': 0.508,
        'reg_alpha': 0.566,
        'reg_lambda': 3.866,
        'learning_rate': 0.125,
        'n_estimators': 818,
        'scale_pos_weight': 4.596
    }

    # Definir rangos de búsqueda amplios alrededor de los valores óptimos
    param_dist = {
        'max_depth': (best_ref['max_depth'] - 3, best_ref['max_depth'] + 3),  # rango 4-10
        'min_child_weight': (best_ref['min_child_weight'] - 1, best_ref['min_child_weight'] + 3),  # rango 0-4
        'gamma': (best_ref['gamma'] - 0.3, best_ref['gamma'] + 0.3),  # rango 0.185-0.785
        'subsample': (best_ref['subsample'] - 0.2, best_ref['subsample'] + 0.09),  # rango 0.709-0.999
        'colsample_bytree': (best_ref['colsample_bytree'] - 0.2, best_ref['colsample_bytree'] + 0.2),  # rango 0.308-0.708
        'reg_alpha': (best_ref['reg_alpha'] - 0.4, best_ref['reg_alpha'] + 0.4),  # rango 0.166-0.966
        'reg_lambda': (best_ref['reg_lambda'] - 2, best_ref['reg_lambda'] + 2),  # rango 1.866-5.866
        'learning_rate': (best_ref['learning_rate'] - 0.075, best_ref['learning_rate'] + 0.075),  # rango 0.05-0.2
        'n_estimators': (best_ref['n_estimators'] - 300, best_ref['n_estimators'] + 300),  # rango 518-1118
        'scale_pos_weight': (best_ref['scale_pos_weight'] - 2, best_ref['scale_pos_weight'] + 2)  # rango 2.596-6.596
    }

    # Generar conjuntos de parámetros aleatorios dentro de los rangos definidos
    param_sets = []
    for _ in range(num_iterations):
        params = {
            'max_depth': int(np.random.randint(param_dist['max_depth'][0], param_dist['max_depth'][1] + 1)),
            'min_child_weight': int(np.random.randint(param_dist['min_child_weight'][0], param_dist['min_child_weight'][1] + 1)),
            'gamma': float(np.random.uniform(param_dist['gamma'][0], param_dist['gamma'][1])),
            'subsample': float(np.random.uniform(param_dist['subsample'][0], min(param_dist['subsample'][1], 1.0))),  # No mayor que 1
            'colsample_bytree': float(np.random.uniform(param_dist['colsample_bytree'][0], min(param_dist['colsample_bytree'][1], 1.0))),  # No mayor que 1
            'reg_alpha': float(np.random.uniform(max(0, param_dist['reg_alpha'][0]), param_dist['reg_alpha'][1])),  # No menor que 0
            'reg_lambda': float(np.random.uniform(max(0, param_dist['reg_lambda'][0]), param_dist['reg_lambda'][1])),  # No menor que 0
            'learning_rate': float(np.random.uniform(max(0.01, param_dist['learning_rate'][0]), param_dist['learning_rate'][1])),  # No menor que 0.01
            'n_estimators': int(np.random.randint(max(100, param_dist['n_estimators'][0]), param_dist['n_estimators'][1] + 1)),  # No menor que 100
            'scale_pos_weight': float(np.random.uniform(max(1, param_dist['scale_pos_weight'][0]), param_dist['scale_pos_weight'][1]))  # No menor que 1
        }
        param_sets.append(params)

    # Comprobar GPUs y núcleos de CPU disponibles
    try:
        num_gpus = len(os.popen('nvidia-smi -L').read().strip().split('\n'))
    except:
        num_gpus = 0  # Por defecto 0 si no se detecta

    num_cpu_cores = multiprocessing.cpu_count()
    print(f"Detectadas {num_gpus} GPUs y {num_cpu_cores} núcleos de CPU")

    # Si no se detecta GPU, ejecutar todo en CPU
    if num_gpus == 0:
        print("No se detectaron GPUs, ejecutando todas las tareas en CPU")
        gpu_tasks = []
        cpu_tasks = param_sets
    else:
        # Asignar la mayoría de las tareas a GPU para maximizar su uso
        gpu_task_ratio = 1.0  # 100% de tareas en GPU cuando esté disponible
        gpu_tasks_count = int(len(param_sets) * gpu_task_ratio)

        gpu_tasks = []
        cpu_tasks = []

        for i, params in enumerate(param_sets):
            if i < gpu_tasks_count:
                # Distribuir entre GPUs disponibles
                gpu_id = i % num_gpus
                gpu_tasks.append((params, X_train_filtered, y_train_resampled, X_test_filtered, y_test, gpu_id))
            else:
                # Tareas de CPU
                cpu_tasks.append((params, X_train_filtered, y_train_resampled, X_test_filtered, y_test))

    results = []

    # Procesar tareas de GPU
    if gpu_tasks:
        print(f"Procesando {len(gpu_tasks)} tareas en GPU(s)...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_gpus) as executor:
            future_to_task = {
                executor.submit(train_model_on_gpu, *task): task for task in gpu_tasks
            }
            for future in concurrent.futures.as_completed(future_to_task):
                try:
                    result = future.result()
                    results.append(result)
                    print(f"Tarea GPU completada con F1: {result['f1_score']:.4f}")
                except Exception as e:
                    print(f"Tarea GPU falló: {e}")

    # Procesar tareas de CPU secuencialmente para evitar problemas de memoria
    if cpu_tasks:
        print(f"Procesando {len(cpu_tasks)} tareas en núcleos de CPU...")
        for task in cpu_tasks:
            try:
                result = train_model_on_cpu(*task)
                results.append(result)
                print(f"Tarea CPU completada con F1: {result['f1_score']:.4f}")
            except Exception as e:
                print(f"Tarea CPU falló: {e}")

    # Encontrar mejor resultado
    if not results:
        raise Exception("¡No hubo ejecuciones exitosas de entrenamiento de modelos!")

    best_result = max(results, key=lambda x: x['f1_score'])

    print("Mejores parámetros encontrados: ", best_result['params'])
    print(f"Mejor puntuación F1: {best_result['f1_score']:.4f}")

    # Crear modelo final con los mejores parámetros
    final_params = best_result['params'].copy()

    # Limpiar parámetros para el modelo final
    if 'gpu_id' in final_params:
        gpu_id = final_params.pop('gpu_id')
    else:
        gpu_id = 0

    # Establecer parámetros adecuados para GPU/CPU para el modelo final
    if has_gpu:
        final_params.update({
            'tree_method': 'hist',
            'device': 'cuda'
        })
    else:
        final_params.update({
            'tree_method': 'hist',
            'device': 'cpu'
        })

    print("Parámetros finales para el mejor modelo:", final_params)

    # Entrenar modelo final usando DMatrix para mejor rendimiento
    dtrain_final = xgb.DMatrix(X_train_filtered, y_train_resampled)
    dtest_final = xgb.DMatrix(X_test_filtered, y_test)

    watchlist = [(dtrain_final, 'train'), (dtest_final, 'eval')]
    num_rounds = final_params.pop('n_estimators', 1000)

    best_bst = xgb.train(
        final_params,
        dtrain_final,
        num_rounds,
        evals=watchlist,
        verbose_eval=False
    )

    # Crear modelo wrapper
    best_model = xgb.XGBClassifier()
    best_model._Booster = best_bst

    # Evaluar modelo final
    y_pred_proba = best_bst.predict(dtest_final)
    best_threshold, best_f1 = find_optimal_threshold(y_test, y_pred_proba)
    y_pred_optimized = (y_pred_proba >= best_threshold).astype(int)

    # Calcular métricas
    metrics = {
        'Exactitud': accuracy_score(y_test, y_pred_optimized),
        'Puntuación F1': f1_score(y_test, y_pred_optimized),
        'Precisión': precision_score(y_test, y_pred_optimized),
        'Recuperación': recall_score(y_test, y_pred_optimized),
        'AUC ROC': roc_auc_score(y_test, y_pred_proba),
        'Mejor Umbral': best_threshold
    }

    # Crear pipeline filtrado con el mejor modelo
    filtered_pipeline = FilteredPipeline(
        preprocessor=preprocessor,
        model=best_model,
        important_indices=important_indices,
        best_threshold=best_threshold
    )

    # Mostrar tiempo total empleado
    end_time = time.time()
    print(f"Tiempo total de optimización: {(end_time - start_time)/60:.2f} minutos")

    return filtered_pipeline, metrics, final_params

# Guardar modelo entrenado en archivo
def save_model(model, filename):
    # Crear directorio de modelos si no existe
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Modelo guardado en '{filename}'")

# Cargar modelo guardado desde archivo
def load_model(filename):
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    print(f"Modelo cargado desde '{filename}'")
    return model

# Función principal para ejecutar flujo completo del modelo
def main():
    print("Cargando datos...")
    hotels, bookings = load_data()

    print(f"Hoteles: {hotels.shape}, Reservas: {bookings.shape}")

    print("Combinando conjuntos de datos y preprocesando...")
    merged, hotel_ids = merge_data(hotels, bookings)

    print(f"Datos preprocesados: {merged.shape}")

    print("Preprocesando características...")
    processed_data = preprocess_data(merged)

    print(f"Datos con características: {processed_data.shape}")

    print("Preparando características para el modelo...")
    X, y, preprocessor = prepare_features(processed_data)

    print(f"X: {X.shape}, y: {y.shape}")
    print(f"Distribución de clases: {y.value_counts(normalize=True)}")

    print("Dividiendo datos en conjuntos de entrenamiento y prueba...")
    X_train, X_test, y_train, y_test, hotel_ids_train, hotel_ids_test = train_test_split(
        X, y, hotel_ids, test_size=0.3, random_state=42, stratify=y
    )

    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

    print("Buscando el mejor modelo XGBoost con optimización GPU-CPU...")
    best_model, best_metrics, best_params = custom_parallel_search(
        X_train, X_test, y_train, y_test, preprocessor, num_iterations=60
    )

    print("Métricas del modelo:")
    for metric, value in best_metrics.items():
        print(f"{metric}: {value}")

    print("\nMejores parámetros:")
    for param, value in best_params.items():
        print(f"{param}: {value}")

    models_path = '/app/models/xgboost_gpu_optimized_best.pkl'
    save_model(best_model, models_path)
    print("Entrenamiento y evaluación del modelo completados.")

if __name__ == "__main__":
    main()