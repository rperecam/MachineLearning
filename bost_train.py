import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import f1_score, classification_report, accuracy_score, precision_score, recall_score, roc_auc_score
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import VotingClassifier
import pickle
import warnings
import os

warnings.filterwarnings('ignore')

def load_data():
    """
    Carga los datasets de hoteles y reservas.
    """
    hotels_df = pd.read_csv('data/hotels.csv')
    bookings_df = pd.read_csv('data/bookings_train.csv')
    return hotels_df, bookings_df

def merge_datasets(hotels_df, bookings_df):
    """
    Une los datasets de hoteles y reservas, preservando hotel_id para GroupKFold.
    """
    merged_df = pd.merge(bookings_df, hotels_df, on='hotel_id', how='left')

    # Filtrar registros con estatus 'Booked' para evitar data leakage
    filtered_df = merged_df[~merged_df['reservation_status'].isin(['Booked', np.nan])].copy()

    # Guardamos hotel_id para GroupKFold DESPUÉS del filtrado
    hotel_ids = filtered_df['hotel_id'].copy()

    return filtered_df, hotel_ids

def map_countries_to_continents(data, country_col='country_x', continent_col='continent_customer', unknown='Desconocido'):
    """
    Mapea códigos de países a continentes.
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

def preprocess_data(df):
    """
    Realiza un preprocesamiento simplificado del dataset enfocado en predecir
    cancelaciones con 30 días de antelación.
    """
    data = df.copy()

    # Conversión de fechas a objetos datetime
    date_columns = ['arrival_date', 'booking_date', 'reservation_status_date']
    for col in date_columns:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col])

    # DEFINICIÓN DEL TARGET: Cancelaciones con al menos 30 días de antelación
    data['days_before_arrival'] = (data['arrival_date'] - data['reservation_status_date']).dt.days
    data['target'] = ((data['reservation_status'] == 'Canceled') &
                      (data['days_before_arrival'] >= 30)).astype(int)

    # Características temporales
    data['lead_time'] = (data['arrival_date'] - data['booking_date']).dt.days
    data['lead_time_category'] = pd.cut(
        data['lead_time'],
        bins=[-1, 7, 30, 90, 180, float('inf')],
        labels=['last_minute', 'short', 'medium', 'long', 'very_long']
    )
    data['is_high_season'] = data['arrival_date'].dt.month.isin([6, 7, 8, 12]).astype(int)
    data['is_weekend_arrival'] = data['arrival_date'].dt.dayofweek.isin([4, 5]).astype(int)

    # Características de precio
    data['price_per_night'] = data['rate'] / np.maximum(data['stay_nights'], 1)
    data['price_per_person'] = data['rate'] / np.maximum(data['total_guests'], 1)
    data['total_cost'] = data['rate']

    # Características de estancia
    data['stay_duration_category'] = pd.cut(
        data['stay_nights'],
        bins=[-1, 1, 3, 7, 14, float('inf')],
        labels=['1_night', '2-3_nights', '4-7_nights', '8-14_nights', '15+_nights']
    )

    # Características de solicitudes especiales
    data['has_special_requests'] = (data['special_requests'] > 0).astype(int)
    data['special_requests_ratio'] = data['special_requests'] / np.maximum(data['total_guests'], 1)

    # Características de localización y contexto
    if 'country_x' in data.columns and 'country_y' in data.columns:
        data['is_foreign'] = (data['country_x'] != data['country_y']).astype(int)
        data.loc[data['country_x'].isna() | data['country_y'].isna(), 'is_foreign'] = 0

    # Características interactivas
    data['price_length_interaction'] = data['price_per_night'] * data['stay_nights']
    data['lead_price_interaction'] = data['lead_time'] * data['price_per_night']

    # Características de transporte y logística
    data['requested_parking'] = (data['required_car_parking_spaces'] > 0).astype(int)

    # Eliminamos columnas que ya no son necesarias o podrían causar data leakage
    columns_to_drop = ['reservation_status', 'reservation_status_date', 'booking_date',
                       'arrival_date', 'days_before_arrival']
    data.drop(columns=columns_to_drop, inplace=True)

    # Manejo de valores infinitos que pueden surgir de divisiones
    for col in ['price_per_night', 'price_per_person', 'special_requests_ratio']:
        if col in data.columns:
            data[col] = data[col].replace([np.inf, -np.inf], np.nan)
            data[col] = data[col].fillna(data[col].median())

    return data

def prepare_features(data):
    """
    Prepara las características para el modelo.
    """
    data_with_continents = map_countries_to_continents(data)

    if 'country_y' in data_with_continents.columns:
        data_with_continents = map_countries_to_continents(
            data_with_continents,
            country_col='country_y',
            continent_col='continent_hotel'
        )

    X = data_with_continents.drop(columns=['target'])
    y = data_with_continents['target']

    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    numerical_transformer = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=7)),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

    return X, y, preprocessor

def find_optimal_threshold(y_true, y_pred_proba):
    """Encuentra el umbral óptimo para maximizar F1 de forma más detallada"""
    thresholds = np.linspace(0.2, 0.8, 100)
    best_threshold, best_f1 = 0.5, 0

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1, best_threshold = f1, threshold

    return best_threshold, best_f1

def create_and_evaluate_model(X_train, X_test, y_train, y_test, preprocessor, hotel_ids_train=None):
    """
    Crea, entrena y evalúa el modelo XGBoost con mejor manejo de overfitting.
    """
    preprocessor.fit(X_train)
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    rus = RandomUnderSampler(sampling_strategy=0.3, random_state=42)
    X_train_under, y_train_under = rus.fit_resample(X_train_transformed, y_train)

    smote = SMOTE(sampling_strategy=0.9, random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_under, y_train_under)

    if hotel_ids_train is not None:
        cv = GroupKFold(n_splits=5)
        groups = hotel_ids_train

        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            max_depth=4,
            min_child_weight=5,
            gamma=0.2,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=3.0,
            learning_rate=0.05,
            n_estimators=300,
            random_state=42
        )

        cv_scores = []
        print("\nEvaluando modelo con GroupKFold por hotel_id:")
        for train_idx, val_idx in cv.split(X_train, y_train, groups=groups):
            X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            X_cv_train_processed = preprocessor.transform(X_cv_train)
            X_cv_val_processed = preprocessor.transform(X_cv_val)

            rus_cv = RandomUnderSampler(sampling_strategy=0.3, random_state=42)
            X_cv_under, y_cv_under = rus_cv.fit_resample(X_cv_train_processed, y_cv_train)

            smote_cv = SMOTE(sampling_strategy=0.9, random_state=42)
            X_cv_balanced, y_cv_balanced = smote_cv.fit_resample(X_cv_under, y_cv_under)

            xgb_model.fit(X_cv_balanced, y_cv_balanced)

            y_cv_proba = xgb_model.predict_proba(X_cv_val_processed)[:, 1]
            threshold, _ = find_optimal_threshold(y_cv_val, y_cv_proba)
            y_cv_pred = (y_cv_proba >= threshold).astype(int)

            cv_f1 = f1_score(y_cv_val, y_cv_pred)
            cv_scores.append(cv_f1)

        print(f"F1 scores en CV: {cv_scores}")
        print(f"Media F1 CV: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

    model = xgb.XGBClassifier(
        objective='binary:logistic',
        max_depth=4,
        min_child_weight=5,
        gamma=0.2,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=3.0,
        learning_rate=0.05,
        n_estimators=300,
        random_state=42
    )

    # With:
    model.fit(
        X_train_resampled,
        y_train_resampled,
        eval_set=[(X_test_transformed, y_test)],
        verbose=False,
    )

    y_pred_proba = model.predict_proba(X_test_transformed)[:, 1]
    best_threshold, best_f1 = find_optimal_threshold(y_test, y_pred_proba)
    y_pred_optimized = (y_pred_proba >= best_threshold).astype(int)

    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred_optimized),
        'F1 Score': f1_score(y_test, y_pred_optimized),
        'Precision': precision_score(y_test, y_pred_optimized),
        'Recall': recall_score(y_test, y_pred_optimized),
        'ROC AUC': roc_auc_score(y_test, y_pred_proba),
        'Best Threshold': best_threshold
    }

    print("\nMétricas del modelo en el conjunto de prueba:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    print("\nReporte de clasificación (umbral optimizado):")
    print(classification_report(y_test, y_pred_optimized))

    feature_names = []
    if hasattr(preprocessor, 'get_feature_names_out'):
        feature_names = preprocessor.get_feature_names_out()
    else:
        feature_names = [f'feature_{i}' for i in range(X_train_transformed.shape[1])]

    importance_scores = model.feature_importances_

    if len(importance_scores) == len(feature_names):
        feature_imp = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance_scores
        }).sort_values('Importance', ascending=False)

        print("\nTop 20 características más importantes:")
        print(feature_imp.head(20))

        os.makedirs('data', exist_ok=True)
        feature_imp.to_csv('data/features_importance_improved.csv', index=False)
        print("Características importantes guardadas en 'data/features_importance_improved.csv'")

    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    full_pipeline.best_threshold = best_threshold

    return full_pipeline, metrics


def optimize_hyperparameters(X_train, y_train, preprocessor):
    """
    Optimiza los hiperparámetros del modelo usando RandomizedSearchCV.
    """
    preprocessor.fit(X_train)
    X_train_transformed = preprocessor.transform(X_train)

    rus = RandomUnderSampler(sampling_strategy=0.3, random_state=42)
    X_train_under, y_train_under = rus.fit_resample(X_train_transformed, y_train)

    smote = SMOTE(sampling_strategy=0.9, random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_under, y_train_under)

    param_dist = {
        'n_estimators': [200, 300, 400, 500],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'subsample': [0.6, 0.7, 0.8],
        'colsample_bytree': [0.6, 0.7, 0.8],
        'min_child_weight': [3, 5, 7],
        'gamma': [0.1, 0.2, 0.3],
        'scale_pos_weight': [1.5, 2.0, 3.0],
        'reg_alpha': [0.05, 0.1, 0.5],
        'reg_lambda': [0.5, 1.0, 1.5]
    }

    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        tree_method='hist',
        grow_policy='lossguide',
        random_state=42,
        use_label_encoder=False
    )

    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=30,
        cv=3,
        scoring='f1',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )

    random_search.fit(X_train_resampled, y_train_resampled)

    print(f"Mejores parámetros: {random_search.best_params_}")
    print(f"Mejor F1-score CV: {random_search.best_score_:.4f}")

    return random_search.best_estimator_

def save_model(model, filename='model/improved_model.pkl'):
    """
    Guarda el modelo entrenado para su uso en producción.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Modelo guardado como {filename}")

def create_ensemble_model(X_train, X_test, y_train, y_test, preprocessor):
    """
    Crea un ensamble de modelos XGBoost para mayor robustez.
    """
    preprocessor.fit(X_train)
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    rus = RandomUnderSampler(sampling_strategy=0.3, random_state=42)
    X_train_under, y_train_under = rus.fit_resample(X_train_transformed, y_train)

    smote = SMOTE(sampling_strategy=0.9, random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_under, y_train_under)

    xgb1 = xgb.XGBClassifier(
        max_depth=4,
        learning_rate=0.05,
        n_estimators=300,
        subsample=0.7,
        colsample_bytree=0.7,
        scale_pos_weight=4.0,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.05,
        reg_lambda=1.0,
        random_state=42,
        eval_metric='logloss'  # Move eval_metric here
    )

    xgb2 = xgb.XGBClassifier(
        max_depth=3,
        learning_rate=0.03,
        n_estimators=400,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=2.0,
        min_child_weight=5,
        gamma=0.2,
        reg_alpha=0.1,
        reg_lambda=1.5,
        random_state=42,
        eval_metric='logloss'  # Move eval_metric here
    )

    xgb3 = xgb.XGBClassifier(
        max_depth=5,
        learning_rate=0.04,
        n_estimators=350,
        subsample=0.75,
        colsample_bytree=0.75,
        scale_pos_weight=3.0,
        min_child_weight=4,
        gamma=0.15,
        reg_alpha=0.08,
        reg_lambda=1.2,
        random_state=42,
        eval_metric='logloss'  # Move eval_metric here
    )

    # Updated fit method calls
    xgb1.fit(X_train_resampled, y_train_resampled,
             eval_set=[(X_test_transformed, y_test)],
             early_stopping_rounds=50,
             verbose=False)

    xgb2.fit(X_train_resampled, y_train_resampled,
             eval_set=[(X_test_transformed, y_test)],
             early_stopping_rounds=50,
             verbose=False)

    xgb3.fit(X_train_resampled, y_train_resampled,
             eval_set=[(X_test_transformed, y_test)],
             early_stopping_rounds=50,
             verbose=False)

    ensemble = VotingClassifier(
        estimators=[
            ('xgb_recall', xgb1),
            ('xgb_precision', xgb2),
            ('xgb_balanced', xgb3)
        ],
        voting='soft',
        weights=[1, 1, 2]
    )

    ensemble.fit(X_train_resampled, y_train_resampled)

    y_pred_proba = ensemble.predict_proba(X_test_transformed)[:, 1]
    best_threshold, best_f1 = find_optimal_threshold(y_test, y_pred_proba)
    y_pred_optimized = (y_pred_proba >= best_threshold).astype(int)

    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred_optimized),
        'F1 Score': f1_score(y_test, y_pred_optimized),
        'Precision': precision_score(y_test, y_pred_optimized),
        'Recall': recall_score(y_test, y_pred_optimized),
        'ROC AUC': roc_auc_score(y_test, y_pred_proba),
        'Best Threshold': best_threshold
    }

    print("\nMétricas del modelo ensamble:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    print("\nReporte de clasificación (ensamble):")
    print(classification_report(y_test, y_pred_optimized))

    ensemble_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', ensemble)
    ])

    ensemble_pipeline.best_threshold = best_threshold

    return ensemble_pipeline, metrics

def predict_cancellation(model, data):
    """
    Realiza predicciones con el modelo entrenado.

    Parámetros:
        model: Pipeline con preprocesador y modelo entrenado
        data: DataFrame con los datos de reservas a predecir

    Retorna:
        DataFrame con las predicciones
    """
    cancellation_prob = model.predict_proba(data)[:, 1]
    cancellation_pred = (cancellation_prob >= model.best_threshold).astype(int)

    results = pd.DataFrame({
        'cancellation_probability': cancellation_prob,
        'cancellation_prediction': cancellation_pred
    })

    return results

def main():
    """
    Función principal que ejecuta todo el proceso de entrenamiento y evaluación.
    """
    print("Iniciando proceso de modelado para predicción de cancelaciones")

    hotels_df, bookings_df = load_data()
    print(f"Datos cargados. Hoteles: {hotels_df.shape}, Reservas: {bookings_df.shape}")

    merged_data, hotel_ids = merge_datasets(hotels_df, bookings_df)
    print(f"Datos unidos. Shape final: {merged_data.shape}")

    preprocessed_data = preprocess_data(merged_data)
    print(f"Datos preprocesados. Shape: {preprocessed_data.shape}")

    X, y, preprocessor = prepare_features(preprocessed_data)
    print(f"Características preparadas. X shape: {X.shape}, y shape: {y.shape}")

    class_counts = y.value_counts()
    print(f"\nDistribución de clases:\n{class_counts}")
    print(f"Proporción de clase positiva: {class_counts[1]/len(y):.2f}")

    X_train, X_test, y_train, y_test, hotel_ids_train, hotel_ids_test = train_test_split(
        X, y, hotel_ids, test_size=0.2, random_state=42, stratify=y
    )
    print(f"División completada. Train: {X_train.shape}, Test: {X_test.shape}")

    model_pipeline, metrics = create_and_evaluate_model(
        X_train, X_test, y_train, y_test, preprocessor, hotel_ids_train
    )

    do_hyperparameter_optimization = False
    if do_hyperparameter_optimization:
        print("\nOptimizando hiperparámetros...")
        optimized_model = optimize_hyperparameters(X_train, y_train, preprocessor)
        print("Optimización completada.")

    ensemble_pipeline, ensemble_metrics = create_ensemble_model(
        X_train, X_test, y_train, y_test, preprocessor
    )

    save_model(model_pipeline, 'model/hotel_cancellation_model.pkl')
    save_model(ensemble_pipeline, 'model/hotel_cancellation_ensemble.pkl')

    print("\nComparando modelos:")
    print(f"Modelo base - F1 Score: {metrics['F1 Score']:.4f}")
    print(f"Ensamble    - F1 Score: {ensemble_metrics['F1 Score']:.4f}")

    print("\nProceso completado con éxito!")

if __name__ == "__main__":
    main()
