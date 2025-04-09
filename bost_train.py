import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import f1_score, classification_report, accuracy_score, precision_score, recall_score, roc_auc_score
import xgboost as xgb
from imblearn.combine import SMOTETomek
import pickle
import warnings
warnings.filterwarnings('ignore')

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

# Función para mapear países a continentes
def map_countries_to_continents(data, country_col='country_x', continent_col='continent_customer', unknown='Desconocido'):
    """
    Mapea códigos de países a continentes
    """
    data_copy = data.copy()

    continent_mapping = {
        'SPA': 'Europa', 'FRA': 'Europa', 'POR': 'Europa', 'AUT': 'Europa', 'NLD': 'Europa', 'ITA': 'Europa', 'GBR': 'Europa', 'DEU': 'Europa', 'DNK': 'Europa', 'POL': 'Europa', 'BEL': 'Europa',
        'FIN': 'Europa', 'NOR': 'Europa', 'HUN': 'Europa', 'CHE': 'Europa', 'ROU': 'Europa', 'SWE': 'Europa', 'UKR': 'Europa', 'GRC': 'Europa', 'LUX': 'Europa', 'MLT': 'Europa', 'CYP': 'Europa',
        'SVK': 'Europa', 'SRB': 'Europa', 'LTU': 'Europa', 'BIH': 'Europa', 'MKD': 'Europa', 'BGR': 'Europa', 'CZE': 'Europa', 'EST': 'Europa', 'LVA': 'Europa', 'ISL': 'Europa', 'SVN': 'Europa',
        'ALB': 'Europa', 'LIE': 'Europa', 'MNE': 'Europa', 'AND': 'Europa', 'IRL': 'Europa', 'HRV': 'Europa', 'IMN': 'Europa', 'FRO': 'Europa', 'GIB': 'Europa', 'SMR': 'Europa', 'GGY': 'Europa', 'JEY': 'Europa', 'GEO': 'Europa',
        'JPN': 'Asia', 'ISR': 'Asia', 'CHN': 'Asia', 'IND': 'Asia', 'IRN': 'Asia', 'IRQ': 'Asia', 'PHL': 'Asia', 'MYS': 'Asia', 'SGP': 'Asia', 'TWN': 'Asia', 'THA': 'Asia',
        'LKA': 'Asia', 'KWT': 'Asia', 'JOR': 'Asia', 'TUR': 'Asia', 'ARE': 'Asia', 'KOR': 'Asia', 'UZB': 'Asia', 'KAZ': 'Asia', 'MAC': 'Asia', 'HKG': 'Asia', 'KHM': 'Asia',
        'BGD': 'Asia', 'AZE': 'Asia', 'LBN': 'Asia', 'SYR': 'Asia', 'VNM': 'Asia', 'QAT': 'Asia', 'OMN': 'Asia', 'PAK': 'Asia', 'TMP': 'Asia', 'NPL': 'Asia', 'IDN': 'Asia', 'SAU': 'Asia', 'MMR': 'Asia', 'ARM': 'Asia',
        'AGO': 'África', 'CMR': 'África', 'DZA': 'África', 'EGY': 'África', 'MAR': 'África', 'ZAF': 'África', 'MOZ': 'África', 'TUN': 'África', 'GNB': 'África', 'NGA': 'África', 'CAF': 'África',
        'KEN': 'África', 'RWA': 'África', 'CIV': 'África', 'SYC': 'África', 'ETH': 'África', 'SEN': 'África', 'GHA': 'África', 'SDN': 'África', 'GAB': 'África', 'BEN': 'África', 'ZMB': 'África',
        'MWI': 'África', 'UGA': 'África', 'ZWE': 'África', 'MUS': 'África', 'TZA': 'África', 'CPV': 'África', 'NAM': 'África', 'MDG': 'África', 'MYT': 'África', 'REU': 'África', 'BWA': 'África',
        'USA': 'América del Norte', 'MEX': 'América del Norte', 'CAN': 'América del Norte', 'CUB': 'América del Norte', 'DOM': 'América del Norte', 'PRI': 'América del Norte', 'CYM': 'América del Norte', 'BHS': 'América del Norte', 'BRB': 'América del Norte',
        'VGB': 'América del Norte', 'JAM': 'América del Norte', 'LCA': 'América del Norte', 'PAN': 'América del Norte', 'CRI': 'América del Norte', 'GTM': 'América del Norte', 'NIC': 'América del Norte', 'HND': 'América del Norte',
        'BRA': 'América del Sur', 'ARG': 'América del Sur', 'ECU': 'América del Sur', 'COL': 'América del Sur', 'PER': 'América del Sur', 'URY': 'América del Sur', 'VEN': 'América del Sur', 'CHL': 'América del Sur', 'BOL': 'América del Sur',
        'PRY': 'América del Sur', 'SUR': 'América del Sur', 'GUF': 'América del Sur', 'GUY': 'América del Sur',
        'AUS': 'Oceanía', 'NZL': 'Oceanía', 'PYF': 'Oceanía', 'NCL': 'Oceanía', 'FJI': 'Oceanía',
        'ATA': 'Antártida',
        'CN': 'Otros'
    }

    if country_col in data_copy.columns:
        data_copy[continent_col] = data_copy[country_col].map(continent_mapping).fillna(unknown)
    else:
        data_copy[continent_col] = unknown

    return data_copy

# 3. Ingeniería de características y preprocesamiento
def preprocess_data(df):
    """
    Realiza el preprocesamiento del dataset unificado e integra la ingeniería de características
    """
    # Copia para evitar modificaciones indeseadas
    data = df.copy()

    # Conversión de fechas a objetos datetime
    date_columns = ['arrival_date', 'booking_date', 'reservation_status_date']
    for col in date_columns:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col])

    # Crear la variable objetivo: cancelación en los últimos 30 días
    # Solo para reservas que ya tienen un estado final (Canceled o CheckOut)
    data['cancellation_lead_time'] = (data['arrival_date'] - data['reservation_status_date']).dt.days

    # La variable target: el cliente canceló la reserva en los últimos 30 días (SÍ/NO)
    data['target'] = ((data['reservation_status'] == 'Canceled') &
                      (data['cancellation_lead_time'] <= 30) &
                      (data['cancellation_lead_time'] >= 0)).astype(int)

    # Características temporales mejoradas
    data['booking_month'] = data['booking_date'].dt.month
    data['booking_day_of_week'] = data['booking_date'].dt.dayofweek
    data['booking_day'] = data['booking_date'].dt.day
    data['booking_quarter'] = data['booking_date'].dt.quarter
    data['booking_weekend'] = data['booking_date'].dt.dayofweek.isin([5, 6]).astype(int)

    data['arrival_month'] = data['arrival_date'].dt.month
    data['arrival_day_of_week'] = data['arrival_date'].dt.dayofweek
    data['arrival_day'] = data['arrival_date'].dt.day
    data['arrival_quarter'] = data['arrival_date'].dt.quarter
    data['is_weekend_arrival'] = data['arrival_day_of_week'].isin([4, 5]).astype(int)

    # Característica de tiempos (lead time)
    if 'arrival_date' in data.columns and 'booking_date' in data.columns:
        data['lead_time'] = (data['arrival_date'] - data['booking_date']).dt.days
        # Categorización de lead time
        data['lead_time_category'] = pd.cut(
            data['lead_time'],
            bins=[-1, 7, 30, 90, 180, float('inf')],
            labels=['last_minute', 'short', 'medium', 'long', 'very_long']
        )

    # Características de precio
    data['price_per_night'] = data['rate'] / np.maximum(data['stay_nights'], 1)
    data['price_per_person'] = data['rate'] / np.maximum(data['total_guests'], 1)

    # Categorización de precios
    data['price_category'] = pd.qcut(
        data['price_per_night'],
        q=5,
        labels=['very_low', 'low', 'medium', 'high', 'very_high'],
        duplicates='drop'
    )

    # Características de ratio y proporciones
    data['special_requests_ratio'] = data['special_requests'] / np.maximum(data['total_guests'], 1)
    data['is_high_season'] = data['arrival_month'].isin([6, 7, 8, 12]).astype(int)
    data['has_special_requests'] = (data['special_requests'] > 0).astype(int)
    data['has_many_special_requests'] = (data['special_requests'] > 1).astype(int)

    # Ratio de parking solicitado
    data['parking_ratio'] = data['required_car_parking_spaces'] / np.maximum(data['total_guests'], 1)
    data['requested_parking'] = (data['required_car_parking_spaces'] > 0).astype(int)

    # Características de activos del hotel
    asset_cols = ['pool_and_spa', 'restaurant', 'parking']
    present_assets = [col for col in asset_cols if col in data.columns]
    if present_assets:
        data['num_assets'] = data[present_assets].fillna(0).astype(int).sum(axis=1)
    else:
        data['num_assets'] = 0

    # Característica de cliente extranjero y países
    country_col_x = 'country_x'
    country_col_y = 'country_y'
    if country_col_x in data.columns and country_col_y in data.columns:
        data['is_foreign'] = (data[country_col_x].astype(str) != data[country_col_y].astype(str)).astype(int)
        data.loc[data[country_col_x].isna() | data[country_col_y].isna(), 'is_foreign'] = 0
    else:
        data['is_foreign'] = 0

    # Duración de estancia
    data['stay_duration_category'] = pd.cut(
        data['stay_nights'],
        bins=[-1, 1, 3, 7, 14, float('inf')],
        labels=['1_night', '2-3_nights', '4-7_nights', '8-14_nights', '15+_nights']
    )

    # Interacciones importantes
    data['price_length_interaction'] = data['price_per_night'] * data['stay_nights']
    data['lead_price_interaction'] = data['lead_time'] * data['price_per_night']
    data['guests_price_interaction'] = data['total_guests'] * data['price_per_person']

    # Características de temporada
    data['is_summer'] = data['arrival_month'].isin([6, 7, 8]).astype(int)
    data['is_winter'] = data['arrival_month'].isin([12, 1, 2]).astype(int)
    data['is_spring'] = data['arrival_month'].isin([3, 4, 5]).astype(int)
    data['is_autumn'] = data['arrival_month'].isin([9, 10, 11]).astype(int)

    # Eliminamos columnas que ya no son necesarias o podrían causar data leakage
    columns_to_drop = ['reservation_status', 'cancellation_lead_time',
                       'reservation_status_date', 'booking_date', 'arrival_date']
    data.drop(columns=columns_to_drop, inplace=True)

    # Manejo de valores infinitos que pueden surgir de divisiones
    for col in ['price_per_night', 'price_per_person', 'special_requests_ratio', 'parking_ratio',]:
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

    # Definimos transformadores para cada tipo de características
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    numerical_transformer = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=5)),  # Mejor estrategia de imputación
        ('scaler', StandardScaler())
    ])

    # Creamos el procesador columnar
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
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
    # SMOTETomek combina over-sampling y under-sampling para mejor manejo del desbalance
    smote_tomek = SMOTETomek(random_state=42)
    X_train_resampled, y_train_resampled = smote_tomek.fit_resample(X_train_transformed, y_train)

    # Creamos y entrenamos el modelo con mejor configuración inicial
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',                # Cambiado de 'f1' a 'auc' que es soportado
        tree_method='hist',
        grow_policy='lossguide',
        scale_pos_weight=1.5,
        random_state=42
    )

    # Entrenar con early stopping para evitar overfitting
    eval_set = [(X_train_resampled, y_train_resampled), (X_test_transformed, y_test)]
    model.fit(
        X_train_resampled,
        y_train_resampled,
        eval_set=eval_set,
        early_stopping_rounds=50,  # XGBoost en versiones recientes acepta early_stopping_rounds
        verbose=False
    )

    # Predicciones
    y_pred = model.predict(X_test_transformed)
    y_pred_proba = model.predict_proba(X_test_transformed)[:, 1]

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

    # Mostrar las características más importantes (top 20)
    feature_names = []
    if hasattr(preprocessor, 'get_feature_names_out'):
        feature_names = preprocessor.get_feature_names_out()
    else:
        # Nombres genéricos si no se pueden obtener los nombres de características
        feature_names = [f'feature_{i}' for i in range(X_train_transformed.shape[1])]

    feature_importances = model.feature_importances_
    if len(feature_importances) == len(feature_names):
        feature_imp = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importances
        }).sort_values('Importance', ascending=False)

        print("\nTop 20 características más importantes:")
        print(feature_imp.head(20))

    # Creamos el pipeline completo para serialización
    # Incluimos el umbral optimizado como parte del modelo
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    # Guardamos el mejor umbral como atributo del pipeline
    full_pipeline.best_threshold = best_threshold

    return full_pipeline, metrics

# 6. Optimización de hiperparámetros
def optimize_hyperparameters(X_train, y_train, preprocessor):
    """
    Optimiza los hiperparámetros del modelo usando RandomizedSearchCV
    """
    # Aplicamos el preprocesamiento
    preprocessor.fit(X_train)
    X_train_transformed = preprocessor.transform(X_train)

    # Aplicamos SMOTETomek para balancear clases (mejor que SMOTE solo)
    smote_tomek = SMOTETomek(random_state=42)
    X_train_resampled, y_train_resampled = smote_tomek.fit_resample(X_train_transformed, y_train)

    # Definimos los hiperparámetros a optimizar con rangos más amplios
    param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.7, 0.8, 0.9],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
        'min_child_weight': [1, 3, 5, 7],
        'gamma': [0, 0.1, 0.2, 0.3],
        'scale_pos_weight': [1, 1.5, 2, 3],  # Importante para clases desbalanceadas
        'reg_alpha': [0, 0.1, 0.5, 1],       # Regularización L1
        'reg_lambda': [0, 0.1, 0.5, 1]       # Regularización L2
    }

    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        tree_method='hist',
        grow_policy='lossguide',
        random_state=42
    )

    # RandomizedSearchCV es más eficiente que GridSearchCV para grandes espacios de búsqueda
    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=30,          # Número de combinaciones a probar
        cv=3,
        scoring='f1',       # Optimizamos para F1
        n_jobs=-1,
        verbose=1,
        random_state=42
    )

    random_search.fit(X_train_resampled, y_train_resampled)

    print(f"Mejores parámetros: {random_search.best_params_}")
    print(f"Mejor F1-score CV: {random_search.best_score_:.4f}")

    return random_search.best_estimator_

# 7. Guardar el modelo para producción
def save_model(model, filename='model/model_boost.pkl'):
    """
    Guarda el modelo entrenado para su uso en producción
    """
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

# 8. Función principal
def main():
    # Carga de datos
    print("Cargando datos...")
    hotels_df, bookings_df = load_data()

    # Fusión de datasets y filtrado para evitar data leakage
    print("Uniendo datasets y filtrando registros 'Booked'...")
    merged_df = merge_datasets(hotels_df, bookings_df)

    # Preprocesamiento e ingeniería de características
    print("Preprocesando datos y aplicando ingeniería de características...")
    processed_data = preprocess_data(merged_df)

    # Visualización de la distribución de la variable objetivo
    print("Distribución de la variable objetivo:")
    target_counts = processed_data['target'].value_counts()
    print(target_counts)
    print(f"Ratio de cancelaciones: {target_counts[1] / len(processed_data):.2%}")

    # Preparación de características
    print("Preparando características...")
    X, y, preprocessor = prepare_features(processed_data)

    # División en conjuntos de entrenamiento y prueba
    print("Dividiendo en conjuntos de entrenamiento y prueba (estratificado)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Evaluación del modelo baseline mejorado
    print("Evaluando modelo baseline mejorado...")
    baseline_model, baseline_metrics = create_and_evaluate_model(
        X_train, X_test, y_train, y_test, preprocessor
    )

    try:
        # Optimización de hiperparámetros
        print("Optimizando hiperparámetros con RandomizedSearchCV...")
        best_model = optimize_hyperparameters(X_train, y_train, preprocessor)

        # Evaluación del modelo optimizado
        print("Evaluando modelo optimizado...")
        X_train_transformed = preprocessor.transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)

        # Uso SMOTETomek para el balanceo de clases
        smote_tomek = SMOTETomek(random_state=42)
        X_train_resampled, y_train_resampled = smote_tomek.fit_resample(X_train_transformed, y_train)

        # Entrenar modelo con early stopping
        eval_set = [(X_train_resampled, y_train_resampled), (X_test_transformed, y_test)]
        best_model.fit(
            X_train_resampled,
            y_train_resampled,
            eval_set=eval_set,
            eval_metric='logloss',
            verbose=False
        )

        # Buscar umbral óptimo para F1
        y_pred_proba = best_model.predict_proba(X_test_transformed)[:, 1]
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

        print("\nMétricas del mejor modelo (umbral optimizado):")
        print(f"F1 Score: {f1_score(y_test, y_pred_optimized):.4f}")
        print(f"Umbral óptimo: {best_threshold:.4f}")
        print("\nReporte de clasificación (umbral optimizado):")
        print(classification_report(y_test, y_pred_optimized))

        # Guardar el modelo optimizado
        print("Guardando el modelo optimizado...")
        full_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', best_model)
        ])
        full_pipeline.best_threshold = best_threshold
        save_model(full_pipeline)

    except Exception as e:
        print(f"Error durante la optimización de hiperparámetros o evaluación: {e}")

# Ejecutar la función principal
if __name__ == "__main__":
    main()
