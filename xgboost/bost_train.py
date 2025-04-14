import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GroupKFold, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, make_scorer
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

# Ignorar advertencias para una salida más limpia
warnings.filterwarnings('ignore')

# Definir una clase de pipeline personalizada para filtrar características
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
    # Usar rutas relativas al directorio raíz del proyecto
    hotels = pd.read_csv('/app/data/hotels.csv')
    bookings = pd.read_csv('/app/data/bookings_train.csv')
    return hotels, bookings

# Combinar datos de hoteles y reservas
def merge_data(hotels, bookings):
    merged = pd.merge(bookings, hotels, on='hotel_id', how='left')
    filtered = merged[~merged['reservation_status'].isin(['Booked', np.nan])].copy()
    hotel_ids = filtered['hotel_id'].copy()
    return filtered, hotel_ids

# Preprocesar datos para la predicción de cancelaciones
def preprocess_data(data):
    data = data.copy()

    # Convertir columnas de fecha a datetime
    date_columns = ['arrival_date', 'booking_date', 'reservation_status_date']
    for col in date_columns:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col])

    # Definir el objetivo: Cancelaciones con al menos 30 días de antelación
    data['days_before_arrival'] = (data['arrival_date'] - data['reservation_status_date']).dt.days
    data['target'] = ((data['reservation_status'] == 'Canceled') & (data['days_before_arrival'] >= 30)).astype(int)

    # Extraer características temporales
    data['lead_time'] = (data['arrival_date'] - data['booking_date']).dt.days
    data['lead_time_category'] = pd.cut(data['lead_time'], bins=[-1, 7, 30, 90, 180, float('inf')], labels=['last_minute', 'short', 'medium', 'long', 'very_long'])
    data['is_high_season'] = data['arrival_date'].dt.month.isin([6, 7, 8, 12]).astype(int)
    data['is_weekend_arrival'] = data['arrival_date'].dt.dayofweek.isin([4, 5]).astype(int)

    # Añadir más características para la estacionalidad
    data['arrival_month'] = data['arrival_date'].dt.month
    data['arrival_dayofweek'] = data['arrival_date'].dt.dayofweek
    data['booking_month'] = data['booking_date'].dt.month

    # Extraer características de precio
    data['price_per_night'] = data['rate'] / np.maximum(data['stay_nights'], 1)
    data['price_per_person'] = data['rate'] / np.maximum(data['total_guests'], 1)
    data['total_cost'] = data['rate']

    # Extraer características de duración de la estancia
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

    # Nueva característica: Desviación del precio promedio por hotel
    hotel_avg_price = data.groupby('hotel_id')['price_per_night'].transform('mean')
    data['price_deviation'] = (data['price_per_night'] - hotel_avg_price) / hotel_avg_price

    # Extraer características de transporte
    data['requested_parking'] = (data['required_car_parking_spaces'] > 0).astype(int)

    # Eliminar columnas que puedan causar fugas de datos
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

# Encontrar el umbral óptimo para el F1 score
def find_optimal_threshold(y_true, y_pred_proba):
    thresholds = np.linspace(0.1, 0.9, 200)
    best_threshold, best_f1 = 0.5, 0

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1, best_threshold = f1, threshold

    return best_threshold, best_f1

# Filtrar características de importancia cero con un umbral adaptativo
def filter_zero_importance_features(model, feature_names, X_train_transformed, X_test_transformed):
    importance_scores = model.feature_importances_

    # Usar percentil para seleccionar características
    importance_threshold = np.percentile(importance_scores, 15)  # Mantener el top 85%
    important_feature_indices = np.where(importance_scores > importance_threshold)[0]

    # Asegurarse de mantener al menos un número mínimo de características
    min_features = max(10, int(X_train_transformed.shape[1] * 0.5))
    if len(important_feature_indices) < min_features:
        important_feature_indices = np.argsort(importance_scores)[-min_features:]

    X_train_array = np.array(X_train_transformed)
    X_test_array = np.array(X_test_transformed)

    X_train_filtered = X_train_array[:, important_feature_indices]
    X_test_filtered = X_test_array[:, important_feature_indices]

    return X_train_filtered, X_test_filtered, important_feature_indices

# Función para entrenar un modelo con parámetros específicos en una GPU específica
def train_model_on_gpu(param_set, X_train, y_train, X_eval, y_eval, gpu_id):
    # Establecer dispositivo GPU específico
    param_set = param_set.copy()

    # Eliminar gpu_id si existe en param_set
    if 'gpu_id' in param_set:
        del param_set['gpu_id']

    # Usar sintaxis actualizada de XGBoost GPU (usar device='cuda' en lugar de tree_method='gpu_hist')
    param_set.update({
        'tree_method': 'hist',
        'device': f'cuda:{gpu_id}',  # Usar formato correcto para device
        'objective': 'binary:logistic',
        'random_state': 42
    })

    # Entrenar modelo
    model = xgb.XGBClassifier(**param_set)
    model.fit(
        X_train, y_train,
        eval_set=[(X_eval, y_eval)],
        verbose=False
    )

    # Predecir y calcular F1 score
    y_pred_proba = model.predict_proba(X_eval)[:, 1]
    best_threshold, best_f1 = find_optimal_threshold(y_eval, y_pred_proba)

    return {
        'params': param_set,
        'model': model,
        'f1_score': best_f1,
        'threshold': best_threshold
    }

# Entrenar modelo en CPU - definido a nivel de módulo para hacerlo serializable para ProcessPoolExecutor
def train_model_on_cpu(param_set, X_train, y_train, X_eval, y_eval):
    param_set = param_set.copy()
    param_set.update({
        'tree_method': 'hist',
        'device': 'cpu',
        'objective': 'binary:logistic',
        'random_state': 42,
        'n_jobs': -1     # Usar múltiples núcleos pero no saturar la CPU
    })

    model = xgb.XGBClassifier(**param_set)
    model.fit(X_train, y_train, eval_set=[(X_eval, y_eval)], verbose=False)

    y_pred_proba = model.predict_proba(X_eval)[:, 1]
    best_threshold, best_f1 = find_optimal_threshold(y_eval, y_pred_proba)

    return {
        'params': param_set,
        'model': model,
        'f1_score': best_f1,
        'threshold': best_threshold
    }

# Realizar búsqueda paralela personalizada usando tanto CPU como GPU
def custom_parallel_search(X_train, X_test, y_train, y_test, preprocessor, num_iterations=60):
    print("Iniciando búsqueda paralelizada de GPU/CPU para encontrar los mejores parámetros de XGBoost...")
    start_time = time.time()

    # Preprocesar los datos
    preprocessor.fit(X_train)
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Equilibrar los datos
    rus = RandomUnderSampler(sampling_strategy=0.30, random_state=42)
    X_train_under, y_train_under = rus.fit_resample(X_train_transformed, y_train)

    smote = SMOTE(sampling_strategy=0.65, random_state=42, k_neighbors=5)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_under, y_train_under)

    # Entrenar modelo inicial para importancia de características
    init_model = xgb.XGBClassifier(
        objective='binary:logistic',
        tree_method='hist',
        device='cuda:0',  # Especificar correctamente el dispositivo
        random_state=42
    )

    init_model.fit(X_train_resampled, y_train_resampled)

    # Filtrar características basadas en importancia
    feature_names = preprocessor.get_feature_names_out() if hasattr(preprocessor, 'get_feature_names_out') else [f'feature_{i}' for i in range(X_train_transformed.shape[1])]
    X_train_filtered, X_test_filtered, important_indices = filter_zero_importance_features(
        init_model, feature_names, X_train_resampled, X_test_transformed
    )

    # Definir rangos de búsqueda más específicos basados en los mejores parámetros encontrados
    # Los mejores parámetros encontrados fueron:
    best_found = {
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

    # Definir parámetros de búsqueda más específicos alrededor de los mejores valores encontrados
    param_dist = {
        'max_depth': randint(5, 9),  # Centrado alrededor de 7
        'min_child_weight': randint(1, 3),  # Centrado alrededor de 1
        'gamma': uniform(0.35, 0.3),  # Rango 0.35-0.65, centrado alrededor de 0.485
        'subsample': uniform(0.85, 0.1),  # Rango 0.85-0.95, centrado alrededor de 0.909
        'colsample_bytree': uniform(0.45, 0.12),  # Rango 0.45-0.57, centrado alrededor de 0.508
        'reg_alpha': uniform(0.4, 0.35),  # Rango 0.4-0.75, centrado alrededor de 0.566
        'reg_lambda': uniform(3.4, 1.0),  # Rango 3.4-4.4, centrado alrededor de 3.866
        'learning_rate': uniform(0.1, 0.05),  # Rango 0.1-0.15, centrado alrededor de 0.125
        'n_estimators': randint(700, 940),  # Rango 700-940, centrado alrededor de 818
        'scale_pos_weight': uniform(4.0, 1.2)  # Rango 4.0-5.2, centrado alrededor de 4.596
    }

    # Generar conjuntos de parámetros aleatorios
    param_sets = []
    for _ in range(num_iterations):
        params = {
            'max_depth': int(randint.rvs(5, 9)),
            'min_child_weight': int(randint.rvs(1, 3)),
            'gamma': float(uniform.rvs(0.35, 0.3)),
            'subsample': float(uniform.rvs(0.85, 0.1)),
            'colsample_bytree': float(uniform.rvs(0.45, 0.12)),
            'reg_alpha': float(uniform.rvs(0.4, 0.35)),
            'reg_lambda': float(uniform.rvs(3.4, 1.0)),
            'learning_rate': float(uniform.rvs(0.1, 0.05)),
            'n_estimators': int(randint.rvs(700, 940)),
            'scale_pos_weight': float(uniform.rvs(4.0, 1.2))
        }
        param_sets.append(params)

    # Verificar GPUs y núcleos de CPU disponibles
    try:
        num_gpus = len(os.popen('nvidia-smi -L').read().strip().split('\n'))
    except:
        num_gpus = 1  # Por defecto a 1 si no se puede detectar

    num_cpu_cores = multiprocessing.cpu_count()
    print(f"Detectadas {num_gpus} GPUs y {num_cpu_cores} núcleos de CPU")

    # Distribuir carga de trabajo entre GPU y CPU
    gpu_tasks = []
    cpu_tasks = []

    # Asignar más tareas a las GPUs pero asegurarse de que la CPU también se utilice
    gpu_task_ratio = 0.8  # 80% de tareas en GPU, 20% en CPU para un procesamiento más rápido
    gpu_tasks_count = int(len(param_sets) * gpu_task_ratio)

    for i, params in enumerate(param_sets):
        if i < gpu_tasks_count:
            # Distribuir entre las GPUs disponibles
            gpu_id = i % num_gpus
            gpu_tasks.append((params, X_train_filtered, y_train_resampled, X_test_filtered, y_test, gpu_id))
        else:
            # Tareas de CPU
            cpu_tasks.append((params, X_train_filtered, y_train_resampled, X_test_filtered, y_test))

    results = []

    # Procesar tareas de GPU
    print(f"Procesando {len(gpu_tasks)} tareas en GPU(s)...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_gpus) as executor:
        future_to_task = {
            executor.submit(train_model_on_gpu, *task): task for task in gpu_tasks
        }
        for future in concurrent.futures.as_completed(future_to_task):
            try:
                result = future.result()
                results.append(result)
                print(f"Tarea de GPU completada con F1: {result['f1_score']:.4f}")
            except Exception as e:
                print(f"Tarea de GPU fallida: {e}")

    # Procesar tareas de CPU secuencialmente si hay problemas con el procesamiento paralelo
    print(f"Procesando {len(cpu_tasks)} tareas en núcleos de CPU...")
    for task in cpu_tasks:
        try:
            result = train_model_on_cpu(*task)
            results.append(result)
            print(f"Tarea de CPU completada con F1: {result['f1_score']:.4f}")
        except Exception as e:
            print(f"Tarea de CPU fallida: {e}")

    # Encontrar el mejor resultado
    best_result = max(results, key=lambda x: x['f1_score'])

    print("Mejores Parámetros Encontrados: ", best_result['params'])
    print(f"Mejor F1 Score: {best_result['f1_score']:.4f}")

    # Crear el modelo final con los parámetros encontrados
    final_params = best_result['params'].copy()

    # CORRECCIÓN: No usar 'gpu_id' cuando ya se especifica 'device'
    # Asegurarse de que solo se usa 'device' para especificar GPU
    if 'gpu_id' in final_params:
        del final_params['gpu_id']  # Eliminar gpu_id si existe

    # Asegurar que se especifica correctamente el device
    if 'device' not in final_params or not final_params['device'].startswith('cuda:'):
        final_params['device'] = 'cuda:0'  # Usar la primera GPU por defecto

    if 'tree_method' not in final_params:
        final_params['tree_method'] = 'hist'

    print("Parámetros finales para el mejor modelo:", final_params)

    # Crear modelo con los mejores parámetros
    best_model = xgb.XGBClassifier(**final_params)

    # Entrenar modelo final
    best_model.fit(
        X_train_filtered,
        y_train_resampled,
        eval_set=[(X_test_filtered, y_test)],
        verbose=False
    )

    # Evaluar modelo final
    y_pred_proba = best_model.predict_proba(X_test_filtered)[:, 1]
    best_threshold, best_f1 = find_optimal_threshold(y_test, y_pred_proba)
    y_pred_optimized = (y_pred_proba >= best_threshold).astype(int)

    # Calcular métricas
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred_optimized),
        'F1 Score': f1_score(y_test, y_pred_optimized),
        'Precision': precision_score(y_test, y_pred_optimized),
        'Recall': recall_score(y_test, y_pred_optimized),
        'ROC AUC': roc_auc_score(y_test, y_pred_proba),
        'Best Threshold': best_threshold
    }

    # Crear pipeline filtrado con el mejor modelo
    filtered_pipeline = FilteredPipeline(
        preprocessor=preprocessor,
        model=best_model,
        important_indices=important_indices,
        best_threshold=best_threshold
    )

    # Mostrar tiempo total tomado
    end_time = time.time()
    print(f"Tiempo total de optimización: {(end_time - start_time)/60:.2f} minutos")

    return filtered_pipeline, metrics, final_params

# Guardar el modelo entrenado en un archivo
def save_model(model, filename):
    # Crear directorio de modelos si no existe
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Modelo guardado en '{filename}'")

# Cargar un modelo guardado desde un archivo
def load_model(filename):
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    print(f"Modelo cargado desde '{filename}'")
    return model

# Función principal para ejecutar el flujo de trabajo completo del modelo
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

    print("Encontrando el mejor modelo XGBoost con optimización híbrida GPU-CPU...")
    best_model, best_metrics, best_params = custom_parallel_search(
        X_train, X_test, y_train, y_test, preprocessor, num_iterations=60
    )

    print("Métricas del modelo:")
    for metric, value in best_metrics.items():
        print(f"{metric}: {value}")

    print("\nMejores parámetros:")
    for param, value in best_params.items():
        print(f"{param}: {value}")

    models_path = '/app/models/xgboost_hybrid_optimized_best.pkl'
    save_model(best_model, models_path)
    print("Entrenamiento y evaluación del modelo completos.")

if __name__ == "__main__":
    main()
