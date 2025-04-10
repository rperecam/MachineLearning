import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import f1_score, classification_report, accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.base import BaseEstimator, TransformerMixin
import xgboost as xgb
from imblearn.combine import SMOTETomek
import pickle
import warnings
import os

warnings.filterwarnings('ignore')

# Función simplificada para manejar outliers
def handle_outliers(data, columns, method='iqr', factor=1.5, quantile_low=0.05, quantile_high=0.95):

    data_clean = data.copy()

    for col in columns:
        if col not in data_clean.columns:
            continue

        if method == 'iqr':
            Q1 = data_clean[col].quantile(0.25)
            Q3 = data_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
        elif method == 'quantile':
            lower_bound = data_clean[col].quantile(quantile_low)
            upper_bound = data_clean[col].quantile(quantile_high)
        else:
            raise ValueError(f"Método '{method}' no soportado. Use 'iqr' o 'quantile'.")

        # Contar outliers para posible análisis
        outliers_count = ((data_clean[col] < lower_bound) | (data_clean[col] > upper_bound)).sum()
        # print(f"Columna {col}: {outliers_count} outliers identificados y tratados")

        # Aplicar límites
        data_clean[col] = np.clip(data_clean[col], lower_bound, upper_bound)

    return data_clean

# 1. Carga de datos
def load_data():
    """
    Carga los datasets de hoteles y reservas
    """
    hotels_df = pd.read_csv('data/hotels.csv')
    bookings_df = pd.read_csv('data/bookings_train.csv')
    return hotels_df, bookings_df

# 2. Unión de datasets
def merge_datasets(hotels_df, bookings_df):
    """
    Une los datasets de hoteles y reservas, eliminando hotel_id para prevenir data leakage
    """
    merged_df = pd.merge(bookings_df, hotels_df, on='hotel_id', how='left')
    merged_df = merged_df.drop('hotel_id', axis=1, errors='ignore')

    # Eliminar registros con estatus 'Booked' para evitar data leakage
    filtered_df = merged_df[~merged_df['reservation_status'].isin(['Booked', np.nan])].copy()

    return filtered_df

# Función para mapear países a continentes - simplificada
def map_countries_to_continents(data, country_col='country_x', continent_col='continent_customer', unknown='Desconocido'):
    """
    Mapea códigos de países a continentes
    """
    data_copy = data.copy()

    # Mapeo simplificado por regiones principales
    continent_mapping = {
        # Europa
        'SPA': 'Europa', 'FRA': 'Europa', 'POR': 'Europa', 'AUT': 'Europa',
        'NLD': 'Europa', 'ITA': 'Europa', 'GBR': 'Europa', 'DEU': 'Europa',
        'DNK': 'Europa', 'POL': 'Europa', 'BEL': 'Europa', 'FIN': 'Europa',
        'NOR': 'Europa', 'HUN': 'Europa', 'CHE': 'Europa', 'ROU': 'Europa',
        'SWE': 'Europa', 'UKR': 'Europa', 'GRC': 'Europa', 'IRL': 'Europa',

        # Asia
        'JPN': 'Asia', 'ISR': 'Asia', 'CHN': 'Asia', 'IND': 'Asia',
        'IRN': 'Asia', 'IRQ': 'Asia', 'PHL': 'Asia', 'MYS': 'Asia',
        'SGP': 'Asia', 'TWN': 'Asia', 'THA': 'Asia', 'KOR': 'Asia',

        # África
        'AGO': 'África', 'CMR': 'África', 'DZA': 'África', 'EGY': 'África',
        'MAR': 'África', 'ZAF': 'África', 'MOZ': 'África', 'TUN': 'África',

        # América del Norte
        'USA': 'América del Norte', 'MEX': 'América del Norte', 'CAN': 'América del Norte',
        'CUB': 'América del Norte', 'DOM': 'América del Norte', 'PRI': 'América del Norte',

        # América del Sur
        'BRA': 'América del Sur', 'ARG': 'América del Sur', 'ECU': 'América del Sur',
        'COL': 'América del Sur', 'PER': 'América del Sur', 'URY': 'América del Sur',
        'VEN': 'América del Sur', 'CHL': 'América del Sur', 'BOL': 'América del Sur',

        # Oceanía
        'AUS': 'Oceanía', 'NZL': 'Oceanía', 'PYF': 'Oceanía', 'NCL': 'Oceanía', 'FJI': 'Oceanía',

        # Otros
        'CN': 'Otros'
    }

    if country_col in data_copy.columns:
        data_copy[continent_col] = data_copy[country_col].map(continent_mapping).fillna(unknown)
    else:
        data_copy[continent_col] = unknown

    return data_copy

# 3. Ingeniería de características y preprocesamiento - simplificado y optimizado
def preprocess_data(df):
    """
    Realiza el preprocesamiento del dataset unificado con ingeniería de características enfocada en la predicción de cancelaciones
    """
    # Copia para evitar modificaciones indeseadas
    data = df.copy()

    # Rellenar con ceros las columnas de required_car_parking_spaces y special_requests
    zero_fill_columns = ['required_car_parking_spaces', 'special_requests']
    for col in zero_fill_columns:
        if col in data.columns:
            data[col] = data[col].fillna(0)

    # Conversión de fechas a objetos datetime
    date_columns = ['arrival_date', 'booking_date', 'reservation_status_date']
    for col in date_columns:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col])

    # Crear la variable objetivo: cancelación en los últimos 30 días
    data['cancellation_lead_time'] = (data['arrival_date'] - data['reservation_status_date']).dt.days

    # La variable target: el cliente canceló la reserva en los últimos 30 días (SÍ/NO)
    data['target'] = ((data['reservation_status'] == 'Canceled') &
                      (data['cancellation_lead_time'] <= 30) &
                      (data['cancellation_lead_time'] >= 0)).astype(int)

    # ---- MANEJO DE OUTLIERS ----
    # Tratar outliers en columnas numéricas clave
    numeric_cols_for_outliers = ['rate', 'total_guests', 'stay_nights']
    data = handle_outliers(data, columns=numeric_cols_for_outliers, method='iqr', factor=1.5)

    # ---- CARACTERÍSTICAS TEMPORALES CLAVE ----
    # Características de booking
    data['booking_month'] = data['booking_date'].dt.month
    data['booking_day_of_week'] = data['booking_date'].dt.dayofweek
    data['booking_weekend'] = data['booking_date'].dt.dayofweek.isin([5, 6]).astype(int)

    # Características de arrival
    data['arrival_month'] = data['arrival_date'].dt.month
    data['arrival_day_of_week'] = data['arrival_date'].dt.dayofweek
    data['is_weekend_arrival'] = data['arrival_day_of_week'].isin([4, 5]).astype(int)
    data['is_high_season'] = data['arrival_month'].isin([6, 7, 8, 12]).astype(int)

    # ---- CARACTERÍSTICAS DE LEAD TIME ----
    if 'arrival_date' in data.columns and 'booking_date' in data.columns:
        data['lead_time'] = (data['arrival_date'] - data['booking_date']).dt.days
        # Categorización de lead time (importante para cancelaciones)
        data['lead_time_category'] = pd.cut(
            data['lead_time'],
            bins=[-1, 7, 30, 90, 180, float('inf')],
            labels=['last_minute', 'short', 'medium', 'long', 'very_long']
        )

    # ---- CARACTERÍSTICAS DE PRECIO ----
    # Características de precio (clave para cancelaciones)
    data['price_per_night'] = data['rate'] / np.maximum(data['stay_nights'], 1)

    # Categorización de precios
    data['price_category'] = pd.qcut(
        data['price_per_night'],
        q=5,
        labels=['very_low', 'low', 'medium', 'high', 'very_high'],
        duplicates='drop'
    )

    # ---- CARACTERÍSTICAS DE HUÉSPEDES Y SOLICITUDES ----
    # Solicitudes especiales (indicador fuerte de compromiso del cliente)
    data['has_special_requests'] = (data['special_requests'] > 0).astype(int)
    data['requested_parking'] = (data['required_car_parking_spaces'] > 0).astype(int)

    # Duración de estancia (factor clave en decisiones de cancelación)
    data['stay_duration_category'] = pd.cut(
        data['stay_nights'],
        bins=[-1, 1, 3, 7, 14, float('inf')],
        labels=['1_night', '2-3_nights', '4-7_nights', '8-14_nights', '15+_nights']
    )

    # ---- CARACTERÍSTICAS DE TEMPORADA ----
    data['is_summer'] = data['arrival_month'].isin([6, 7, 8]).astype(int)
    data['is_winter'] = data['arrival_month'].isin([12, 1, 2]).astype(int)

    # ---- PAÍS/LOCALIZACIÓN ----
    # Característica de cliente extranjero (importante para predecir cancelaciones)
    country_col_x = 'country_x'
    country_col_y = 'country_y'
    if country_col_x in data.columns and country_col_y in data.columns:
        data['is_foreign'] = (data[country_col_x].astype(str) != data[country_col_y].astype(str)).astype(int)
        data.loc[data[country_col_x].isna() | data[country_col_y].isna(), 'is_foreign'] = 0

    # ---- ELIMINACIÓN DE COLUMNAS ----
    # Columnas que ya no son necesarias o podrían causar data leakage
    columns_to_drop = ['reservation_status', 'cancellation_lead_time',
                       'reservation_status_date', 'booking_date', 'arrival_date']
    data.drop(columns=columns_to_drop, inplace=True)

    # Manejo de valores infinitos que pueden surgir de divisiones
    for col in ['price_per_night']:
        if col in data.columns:
            data[col] = data[col].replace([np.inf, -np.inf], np.nan)
            data[col] = data[col].fillna(data[col].median())

    return data

# 4. División de características y definición de pipelines
def prepare_features(data):
    """
    Prepara las características para el modelo
    """
    # Primero aplicamos la función de mapeo de continentes
    data_with_continents = map_countries_to_continents(data)

    # También mapear el país del hotel
    if 'country_y' in data_with_continents.columns:
        data_with_continents = map_countries_to_continents(
            data_with_continents,
            country_col='country_y',
            continent_col='continent_hotel'
        )

    # Separamos features y target
    X = data_with_continents.drop(columns=['target'])
    y = data_with_continents['target']

    # Separamos los tipos de características
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Transformador numérico con mejor manejo de valores faltantes
    numerical_transformer = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=5, weights='uniform')),
        ('scaler', StandardScaler())
    ])

    # Transformador para características categóricas
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Columnas con tratamiento especial (zeros)
    special_cols = ['required_car_parking_spaces', 'special_requests']
    regular_num_cols = [col for col in numerical_features if col not in special_cols]

    # Preprocesador con manejo específico para columnas especiales
    transformers = [
        ('num', numerical_transformer, regular_num_cols),
        ('cat', categorical_transformer, categorical_features)
    ]

    # Si existen las columnas especiales, agregar un transformador específico para ellas
    special_present = [col for col in special_cols if col in numerical_features]
    if special_present:
        special_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
            ('scaler', StandardScaler())
        ])
        transformers.append(('special', special_transformer, special_present))

    # Creamos el procesador columnar
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'
    )

    return X, y, preprocessor

# 5. Crear y evaluar el modelo XGBoost
def create_and_evaluate_model(X_train, X_test, y_train, y_test, preprocessor):
    """
    Crea, entrena y evalúa el modelo XGBoost
    """
    # Aplicamos el preprocesamiento
    preprocessor.fit(X_train)
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Aplicamos SMOTETomek para balancear clases (solo al conjunto de entrenamiento)
    smote_tomek = SMOTETomek(random_state=42)
    X_train_resampled, y_train_resampled = smote_tomek.fit_resample(X_train_transformed, y_train)

    # Convertimos los datos a DMatrix
    dtrain = xgb.DMatrix(X_train_resampled, label=y_train_resampled)
    dtest = xgb.DMatrix(X_test_transformed, label=y_test)

    # Definimos los parámetros del modelo
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'hist',
        'grow_policy': 'lossguide',
        'scale_pos_weight': 1.5,
        'seed': 42
    }

    # Entrenamos el modelo con early stopping
    evals = [(dtrain, 'train'), (dtest, 'eval')]
    evals_result = {}
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=evals,
        early_stopping_rounds=50,
        evals_result=evals_result,
        verbose_eval=False
    )

    # Predicciones
    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Ajustar el umbral de clasificación para optimizar F1
    thresholds = np.arange(0.3, 0.7, 0.01)
    best_threshold = 0.5
    best_f1 = 0

    for threshold in thresholds:
        y_pred_threshold = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_test, y_pred_threshold)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    # Usar el mejor umbral
    y_pred_optimized = (y_pred_proba >= best_threshold).astype(int)

    # Métricas con umbral optimizado
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred_optimized),
        'F1 Score': f1_score(y_test, y_pred_optimized),
        'Precision': precision_score(y_test, y_pred_optimized),
        'Recall': recall_score(y_test, y_pred_optimized),
        'ROC AUC': roc_auc_score(y_test, y_pred_proba),
        'Best Threshold': best_threshold
    }

    # Mostramos las métricas
    print("\nMétricas del modelo en el conjunto de prueba:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    # Reporte de clasificación detallado
    print("\nReporte de clasificación (umbral optimizado):")
    print(classification_report(y_test, y_pred_optimized))

    # Obtener y guardar las características más importantes
    feature_names = []
    if hasattr(preprocessor, 'get_feature_names_out'):
        feature_names = preprocessor.get_feature_names_out()
    else:
        # Nombres genéricos si no se pueden obtener los nombres de características
        feature_names = [f'feature_{i}' for i in range(X_train_transformed.shape[1])]

    importance_dict = model.get_score(importance_type='weight')
    feature_importances = [importance_dict.get(f'f{i}', 0) for i in range(len(feature_names))]

    if len(feature_importances) == len(feature_names):
        feature_imp = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importances
        }).sort_values('Importance', ascending=False)

        print("\nTop 20 características más importantes:")
        print(feature_imp.head(20))

        # Guardar las características importantes en un CSV
        os.makedirs('data', exist_ok=True)
        feature_imp.to_csv('data/features_importance_baseline.csv', index=False)
        print("Características importantes guardadas en 'data/features_importance_baseline.csv'")

    # Creamos el pipeline completo para serialización
    # Incluimos el umbral optimizado como parte del modelo
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    # Guardamos el mejor umbral como atributo del pipeline
    full_pipeline.best_threshold = best_threshold

    return full_pipeline, metrics, feature_imp

# 6. Optimización de hiperparámetros
def optimize_hyperparameters(X_train, y_train, preprocessor):
    """
    Optimiza los hiperparámetros del modelo usando RandomizedSearchCV
    """
    # Aplicamos el preprocesamiento
    print("Aplicando preprocesamiento para optimización...")
    preprocessor.fit(X_train)
    X_train_transformed = preprocessor.transform(X_train)

    # Aplicamos SMOTETomek para balancear clases
    print("Aplicando SMOTETomek para balancear clases...")
    smote_tomek = SMOTETomek(random_state=42)
    X_train_resampled, y_train_resampled = smote_tomek.fit_resample(X_train_transformed, y_train)

    # Definimos los hiperparámetros a optimizar - reducidos para mayor eficiencia
    param_dist = {
        'n_estimators': [100, 300, 500],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'min_child_weight': [1, 3, 5],
        'scale_pos_weight': [1, 1.5, 2],
    }

    # Creamos el modelo base para la optimización
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        tree_method='hist',
        grow_policy='lossguide',
        random_state=42,
        use_label_encoder=False
    )

    # RandomizedSearchCV para buscar los mejores hiperparámetros
    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=20,  # Reducido para mayor eficiencia
        cv=3,
        scoring='f1',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )

    print("Iniciando búsqueda de hiperparámetros...")
    random_search.fit(X_train_resampled, y_train_resampled)

    print(f"Mejores parámetros: {random_search.best_params_}")
    print(f"Mejor F1-score CV: {random_search.best_score_:.4f}")

    return random_search.best_estimator_

# 7. Guardar el modelo para producción
def save_model(model, filename='model/model_boost.pkl'):
    """
    Guarda el modelo entrenado para su uso en producción
    """
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Modelo guardado como {filename}")

# Función para aplicar umbral óptimo durante la predicción
def predict_with_optimal_threshold(pipeline, X, threshold=None):
    """
    Realiza predicciones usando el umbral óptimo para maximizar F1
    """
    # Si no se especifica umbral, usar el almacenado en el pipeline
    if threshold is None:
        if hasattr(pipeline, 'best_threshold'):
            threshold = pipeline.best_threshold
        else:
            threshold = 0.5

    # Obtenemos probabilidades y aplicamos umbral
    y_proba = pipeline.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    return y_pred

# Evaluación del modelo optimizado
def evaluate_optimized_model(best_model, X_train, X_test, y_train, y_test, preprocessor):
    """
    Evalúa el modelo optimizado con los hiperparámetros encontrados
    """
    print("Evaluando modelo optimizado...")
    # Aplicamos el preprocesamiento
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Balanceamos clases solo en el conjunto de entrenamiento
    smote_tomek = SMOTETomek(random_state=42)
    X_train_resampled, y_train_resampled = smote_tomek.fit_resample(X_train_transformed, y_train)

    # Entrenamos el modelo optimizado
    eval_set = [(X_test_transformed, y_test)]
    best_model.fit(
        X_train_resampled,
        y_train_resampled,
        eval_set=eval_set,
        verbose=False
    )

    # Predecimos probabilidades
    y_pred_proba = best_model.predict_proba(X_test_transformed)[:, 1]

    # Buscamos umbral óptimo para F1
    thresholds = np.arange(0.3, 0.7, 0.01)
    best_threshold = 0.5
    best_f1 = 0

    for threshold in thresholds:
        y_pred_threshold = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_test, y_pred_threshold)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    # Predicciones con umbral optimizado
    y_pred_optimized = (y_pred_proba >= best_threshold).astype(int)

    # Métricas con umbral optimizado
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred_optimized),
        'F1 Score': f1_score(y_test, y_pred_optimized),
        'Precision': precision_score(y_test, y_pred_optimized),
        'Recall': recall_score(y_test, y_pred_optimized),
        'ROC AUC': roc_auc_score(y_test, y_pred_proba),
        'Best Threshold': best_threshold
    }

    # Mostramos las métricas
    print("\nMétricas del modelo optimizado:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    # Reporte de clasificación detallado
    print("\nReporte de clasificación (umbral optimizado):")
    print(classification_report(y_test, y_pred_optimized))

    return best_model, metrics, best_threshold

# Función principal para ejecutar todo el pipeline
def main():
    """
    Función principal que ejecuta el pipeline completo
    """
    print("Iniciando pipeline de modelado para predicción de cancelaciones de hotel...")

    # 1. Cargar datos
    print("Cargando datos...")
    hotels_df, bookings_df = load_data()

    # 2. Unir datasets
    print("Uniendo datasets...")
    merged_data = merge_datasets(hotels_df, bookings_df)

    # 3. Preprocesamiento y feature engineering
    print("Aplicando preprocesamiento y feature engineering...")
    processed_data = preprocess_data(merged_data)

    # 4. Preparar características
    print("Preparando características para el modelo...")
    X, y, preprocessor = prepare_features(processed_data)

    # 5. Dividir en conjuntos de entrenamiento y prueba
    print("Dividiendo datos en conjuntos de entrenamiento y prueba...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6. Crear y evaluar modelo base
    print("Creando y evaluando modelo base...")
    pipeline, metrics, feature_importances = create_and_evaluate_model(
        X_train, X_test, y_train, y_test, preprocessor
    )

    # 7. Optimizar hiperparámetros (opcional)
    print("¿Desea optimizar hiperparámetros? (s/n)")
    optimize = input().strip().lower()

    if optimize == 's':
        print("Optimizando hiperparámetros...")
        best_model = optimize_hyperparameters(X_train, y_train, preprocessor)

        # 8. Evaluar modelo optimizado
        print("Evaluando modelo optimizado...")
        optimized_model, optimized_metrics, best_threshold = evaluate_optimized_model(
            best_model, X_train, X_test, y_train, y_test, preprocessor
        )

        # 9. Guardar modelo optimizado
        print("Guardando modelo optimizado...")
        full_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', optimized_model)
        ])
        full_pipeline.best_threshold = best_threshold
        save_model(full_pipeline, 'model/optimized_model.pkl')
    else:
        # 9. Guardar modelo base
        print("Guardando modelo base...")
        save_model(pipeline)

    print("¡Proceso completado con éxito!")
    return

if __name__ == "__main__":
    main()