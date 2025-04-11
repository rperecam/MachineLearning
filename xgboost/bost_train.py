import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import VotingClassifier
import pickle
import warnings

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

# Define a custom pipeline class for filtering features
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

# Define a custom ensemble pipeline class for filtering features
class FilteredEnsemblePipeline:
    def __init__(self, preprocessor, ensemble, important_indices, best_threshold):
        self.preprocessor = preprocessor
        self.ensemble = ensemble
        self.important_indices = important_indices
        self.best_threshold = best_threshold

    def predict_proba(self, X):
        X_transformed = self.preprocessor.transform(X)
        X_filtered = X_transformed[:, self.important_indices]
        return self.ensemble.predict_proba(X_filtered)

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.best_threshold).astype(int)

# Load hotel and booking data
def load_data():
    hotels = pd.read_csv('C:/Users/Administrador/DataspellProjects/Aprendizaje_automatico2/data/hotels.csv')
    bookings = pd.read_csv('C:/Users/Administrador/DataspellProjects/Aprendizaje_automatico2/data/bookings_train.csv')
    return hotels, bookings

# Merge hotel and booking data
def merge_data(hotels, bookings):
    merged = pd.merge(bookings, hotels, on='hotel_id', how='left')
    filtered = merged[~merged['reservation_status'].isin(['Booked', np.nan])].copy()
    hotel_ids = filtered['hotel_id'].copy()
    return filtered, hotel_ids

# Preprocess data for cancellation prediction
def preprocess_data(data):
    data = data.copy()

    # Convert date columns to datetime
    date_columns = ['arrival_date', 'booking_date', 'reservation_status_date']
    for col in date_columns:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col])

    # Define target: Cancellations with at least 30 days in advance
    data['days_before_arrival'] = (data['arrival_date'] - data['reservation_status_date']).dt.days
    data['target'] = ((data['reservation_status'] == 'Canceled') & (data['days_before_arrival'] >= 30)).astype(int)

    # Extract temporal features
    data['lead_time'] = (data['arrival_date'] - data['booking_date']).dt.days
    data['lead_time_category'] = pd.cut(data['lead_time'], bins=[-1, 7, 30, 90, 180, float('inf')], labels=['last_minute', 'short', 'medium', 'long', 'very_long'])
    data['is_high_season'] = data['arrival_date'].dt.month.isin([6, 7, 8, 12]).astype(int)
    data['is_weekend_arrival'] = data['arrival_date'].dt.dayofweek.isin([4, 5]).astype(int)

    # Adding more features for seasonality
    data['arrival_month'] = data['arrival_date'].dt.month
    data['arrival_dayofweek'] = data['arrival_date'].dt.dayofweek
    data['booking_month'] = data['booking_date'].dt.month

    # Extract price features
    data['price_per_night'] = data['rate'] / np.maximum(data['stay_nights'], 1)
    data['price_per_person'] = data['rate'] / np.maximum(data['total_guests'], 1)
    data['total_cost'] = data['rate']

    # Extract stay duration features
    data['stay_duration_category'] = pd.cut(data['stay_nights'], bins=[-1, 1, 3, 7, 14, float('inf')], labels=['1_night', '2-3_nights', '4-7_nights', '8-14_nights', '15+_nights'])

    # Extract special requests features
    data['has_special_requests'] = (data['special_requests'] > 0).astype(int)
    data['special_requests_ratio'] = data['special_requests'] / np.maximum(data['total_guests'], 1)

    # Extract location features
    if 'country_x' in data.columns and 'country_y' in data.columns:
        data['is_foreign'] = (data['country_x'] != data['country_y']).astype(int)
        data.loc[data['country_x'].isna() | data['country_y'].isna(), 'is_foreign'] = 0

    # Extract interactive features
    data['price_length_interaction'] = data['price_per_night'] * data['stay_nights']
    data['lead_price_interaction'] = data['lead_time'] * data['price_per_night']

    # New feature: Price deviation from average by hotel
    hotel_avg_price = data.groupby('hotel_id')['price_per_night'].transform('mean')
    data['price_deviation'] = (data['price_per_night'] - hotel_avg_price) / hotel_avg_price

    # Extract transport features
    data['requested_parking'] = (data['required_car_parking_spaces'] > 0).astype(int)

    # Drop columns that may cause data leakage
    columns_to_drop = ['reservation_status', 'reservation_status_date', 'booking_date', 'arrival_date', 'days_before_arrival']
    data.drop(columns=columns_to_drop, inplace=True)

    # Handle infinite and null values in numerical features
    for col in ['price_per_night', 'price_per_person', 'special_requests_ratio', 'price_deviation']:
        if col in data.columns:
            data[col] = data[col].replace([np.inf, -np.inf], np.nan)
            data[col] = data[col].fillna(data[col].median())

    return data

# Prepare features for the model
def prepare_features(data):
    X = data.drop(columns=['target'])
    y = data['target']

    # Identify categorical and numerical features
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Pipeline for categorical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Pipeline for numerical features
    numerical_transformer = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', StandardScaler())
    ])

    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

    return X, y, preprocessor

# Find the optimal threshold for F1 score
def find_optimal_threshold(y_true, y_pred_proba):
    # Más granularidad en el rango de umbrales
    thresholds = np.linspace(0.1, 0.9, 200)
    best_threshold, best_f1 = 0.5, 0

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1, best_threshold = f1, threshold

    return best_threshold, best_f1

# Filter out zero-importance features with un umbral adaptativo
def filter_zero_importance_features(model, feature_names, X_train_transformed, X_test_transformed):
    importance_scores = model.feature_importances_

    # Usando percentil para seleccionar características
    importance_threshold = np.percentile(importance_scores, 15)  # Mantiene el 85% superior
    important_feature_indices = np.where(importance_scores > importance_threshold)[0]

    # Asegurar que mantenemos al menos un número mínimo de características
    min_features = max(10, int(X_train_transformed.shape[1] * 0.5))
    if len(important_feature_indices) < min_features:
        important_feature_indices = np.argsort(importance_scores)[-min_features:]

    X_train_array = np.array(X_train_transformed)
    X_test_array = np.array(X_test_transformed)

    X_train_filtered = X_train_array[:, important_feature_indices]
    X_test_filtered = X_test_array[:, important_feature_indices]

    return X_train_filtered, X_test_filtered, important_feature_indices

# Create and evaluate the XGBoost model
def create_and_evaluate_model(X_train, X_test, y_train, y_test, preprocessor, hotel_ids_train=None):
    preprocessor.fit(X_train)
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Balance the data - adjusted sampling strategies
    rus = RandomUnderSampler(sampling_strategy=0.30, random_state=42)  # Ajustado de 0.25
    X_train_under, y_train_under = rus.fit_resample(X_train_transformed, y_train)

    smote = SMOTE(sampling_strategy=0.65, random_state=42, k_neighbors=5)  # Ajustado de 0.7
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_under, y_train_under)

    # Evaluate with GroupKFold if hotel_ids are provided
    if hotel_ids_train is not None:
        cv = GroupKFold(n_splits=5)
        groups = hotel_ids_train

        # Parámetros optimizados para generalización
        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            max_depth=4,                # Incrementado para capturar relaciones complejas
            min_child_weight=6,         # Ligeramente reducido para balancear regularización
            gamma=0.25,                 # Reducido para permitir más splits
            subsample=0.6,              # Reducido para combatir sobreajuste
            colsample_bytree=0.6,       # Reducido para mayor diversidad
            reg_alpha=0.6,              # Incrementado regularización L1
            reg_lambda=2.5,             # Incrementado regularización L2
            scale_pos_weight=1.8,       # Ajustado para balancear clases
            learning_rate=0.01,         # Tasa de aprendizaje reducida
            n_estimators=450,           # Más árboles para mejor convergencia
            random_state=42
        )

        cv_scores = []
        for train_idx, val_idx in cv.split(X_train, y_train, groups=groups):
            X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            X_cv_train_processed = preprocessor.transform(X_cv_train)
            X_cv_val_processed = preprocessor.transform(X_cv_val)

            rus_cv = RandomUnderSampler(sampling_strategy=0.30, random_state=42)
            X_cv_under, y_cv_under = rus_cv.fit_resample(X_cv_train_processed, y_cv_train)

            smote_cv = SMOTE(sampling_strategy=0.65, random_state=42, k_neighbors=5)
            X_cv_balanced, y_cv_balanced = smote_cv.fit_resample(X_cv_under, y_cv_under)

            xgb_model.fit(
                X_cv_balanced, y_cv_balanced,
                eval_set=[(X_cv_val_processed, y_cv_val)],
                verbose=False
            )

            y_cv_proba = xgb_model.predict_proba(X_cv_val_processed)[:, 1]
            threshold, _ = find_optimal_threshold(y_cv_val, y_cv_proba)
            y_cv_pred = (y_cv_proba >= threshold).astype(int)

            cv_f1 = f1_score(y_cv_val, y_cv_pred)
            cv_scores.append(cv_f1)

    # Train the initial model for feature importance
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        max_depth=4,
        min_child_weight=6,
        gamma=0.25,
        subsample=0.6,
        colsample_bytree=0.6,
        reg_alpha=0.6,
        reg_lambda=2.5,
        scale_pos_weight=1.8,
        learning_rate=0.01,
        n_estimators=450,
        random_state=42
    )

    model.fit(
        X_train_resampled,
        y_train_resampled,
        eval_set=[(X_test_transformed, y_test)],
        verbose=False,
    )

    feature_names = preprocessor.get_feature_names_out() if hasattr(preprocessor, 'get_feature_names_out') else [f'feature_{i}' for i in range(X_train_transformed.shape[1])]

    # Filter zero-importance features from both train_transformed and test_transformed
    X_train_transformed_filtered, X_test_filtered, important_indices = filter_zero_importance_features(
        model, feature_names, X_train_transformed, X_test_transformed
    )

    # Apply the same filter to the resampled data
    X_train_resampled_array = np.array(X_train_resampled)
    X_train_resampled_filtered = X_train_resampled_array[:, important_indices]

    # Create a model specifically for filtered features
    filtered_model = xgb.XGBClassifier(
        objective='binary:logistic',
        max_depth=4,
        min_child_weight=6,
        gamma=0.25,
        subsample=0.6,
        colsample_bytree=0.6,
        reg_alpha=0.6,
        reg_lambda=2.5,
        scale_pos_weight=1.8,
        learning_rate=0.01,
        n_estimators=450,
        random_state=42
    )

    filtered_model.fit(
        X_train_resampled_filtered,
        y_train_resampled,
        eval_set=[(X_test_filtered, y_test)],
        verbose=False,
    )

    y_pred_proba = filtered_model.predict_proba(X_test_filtered)[:, 1]
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

    filtered_pipeline = FilteredPipeline(
        preprocessor=preprocessor,
        model=filtered_model,
        important_indices=important_indices,
        best_threshold=best_threshold
    )

    return filtered_pipeline, metrics

# Create an ensemble model for robustness
def create_ensemble_model(X_train, X_test, y_train, y_test, preprocessor):
    preprocessor.fit(X_train)
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Balance the data - ligeramente ajustados para mejorar generalización
    rus = RandomUnderSampler(sampling_strategy=0.30, random_state=42)
    X_train_under, y_train_under = rus.fit_resample(X_train_transformed, y_train)

    smote = SMOTE(sampling_strategy=0.65, random_state=42, k_neighbors=5)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_under, y_train_under)

    # Modelos más diversos para mejorar la generalización
    xgb1 = xgb.XGBClassifier(
        max_depth=5,                # Modelo complejo para maximizar recall
        learning_rate=0.01,
        n_estimators=400,
        subsample=0.55,             # Submuestreo más agresivo contra sobreajuste
        colsample_bytree=0.55,      # Submuestreo de características más agresivo
        scale_pos_weight=2.5,       # Mayor peso a positivos para recall
        min_child_weight=5,
        gamma=0.2,                  # Bajo gamma para permitir árboles complejos
        reg_alpha=0.7,              # Alta regularización L1 para esparcidad
        reg_lambda=1.5,
        random_state=42
    )

    xgb2 = xgb.XGBClassifier(
        max_depth=3,                # Modelo simple para maximizar precisión
        learning_rate=0.008,        # Tasa de aprendizaje muy baja para generalización
        n_estimators=600,           # Muchos árboles simples
        subsample=0.8,              # Menor varianza
        colsample_bytree=0.8,
        scale_pos_weight=1.2,       # Más equilibrado para precisión
        min_child_weight=9,         # Alta regularización
        gamma=0.5,                  # Alto gamma para simplicidad
        reg_alpha=0.8,
        reg_lambda=3.0,
        random_state=42
    )

    xgb3 = xgb.XGBClassifier(
        max_depth=4,                # Modelo equilibrado
        learning_rate=0.01,
        n_estimators=500,
        subsample=0.65,
        colsample_bytree=0.65,
        scale_pos_weight=1.8,       # Equilibrio entre recall y precisión
        min_child_weight=6,
        gamma=0.35,
        reg_alpha=0.6,
        reg_lambda=2.2,
        random_state=42
    )

    xgb1.fit(X_train_resampled, y_train_resampled, verbose=False)
    xgb2.fit(X_train_resampled, y_train_resampled, verbose=False)
    xgb3.fit(X_train_resampled, y_train_resampled, verbose=False)

    feature_names = preprocessor.get_feature_names_out() if hasattr(preprocessor, 'get_feature_names_out') else [f'feature_{i}' for i in range(X_train_transformed.shape[1])]

    importance1 = xgb1.feature_importances_
    importance2 = xgb2.feature_importances_
    importance3 = xgb3.feature_importances_

    # Ponderación que favorece ligeramente el modelo equilibrado
    combined_importance = (0.3 * importance1 + 0.3 * importance2 + 0.4 * importance3)

    # Usando percentil para umbral de importancia
    importance_threshold = np.percentile(combined_importance, 15)
    important_feature_indices = np.where(combined_importance > importance_threshold)[0]

    # Garantizar un número mínimo de características
    min_features = max(10, int(X_train_transformed.shape[1] * 0.5))
    if len(important_feature_indices) < min_features:
        important_feature_indices = np.argsort(combined_importance)[-min_features:]

    X_train_array = np.array(X_train_resampled)
    X_test_array = np.array(X_test_transformed)

    X_train_filtered = X_train_array[:, important_feature_indices]
    X_test_filtered = X_test_array[:, important_feature_indices]

    # Reentrenar modelos con características filtradas
    xgb1_filtered = xgb.XGBClassifier(
        max_depth=5, learning_rate=0.01, n_estimators=400,
        subsample=0.55, colsample_bytree=0.55, scale_pos_weight=2.5,
        min_child_weight=5, gamma=0.2, reg_alpha=0.7, reg_lambda=1.5,
        random_state=42
    )

    xgb2_filtered = xgb.XGBClassifier(
        max_depth=3, learning_rate=0.008, n_estimators=600,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=1.2,
        min_child_weight=9, gamma=0.5, reg_alpha=0.8, reg_lambda=3.0,
        random_state=42
    )

    xgb3_filtered = xgb.XGBClassifier(
        max_depth=4, learning_rate=0.01, n_estimators=500,
        subsample=0.65, colsample_bytree=0.65, scale_pos_weight=1.8,
        min_child_weight=6, gamma=0.35, reg_alpha=0.6, reg_lambda=2.2,
        random_state=42
    )

    xgb1_filtered.fit(X_train_filtered, y_train_resampled, verbose=False)
    xgb2_filtered.fit(X_train_filtered, y_train_resampled, verbose=False)
    xgb3_filtered.fit(X_train_filtered, y_train_resampled, verbose=False)

    # Pesos ajustados para favorecer el modelo balanceado
    ensemble_filtered = VotingClassifier(
        estimators=[
            ('xgb_recall', xgb1_filtered),
            ('xgb_precision', xgb2_filtered),
            ('xgb_balanced', xgb3_filtered)
        ],
        voting='soft',
        weights=[1.0, 1.0, 1.5]  # Mayor peso al modelo equilibrado
    )

    ensemble_filtered.fit(X_train_filtered, y_train_resampled)

    y_pred_proba = ensemble_filtered.predict_proba(X_test_filtered)[:, 1]
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

    filtered_ensemble_pipeline = FilteredEnsemblePipeline(
        preprocessor=preprocessor,
        ensemble=ensemble_filtered,
        important_indices=important_feature_indices,
        best_threshold=best_threshold
    )

    return filtered_ensemble_pipeline, metrics

# Save the trained model to a file
def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved in '{filename}'")

# Load a saved model from a file
def load_model(filename):
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    print(f"Model loaded from '{filename}'")
    return model

# Main function to run the complete model workflow
def main():
    print("Loading data...")
    hotels, bookings = load_data()

    print(f"Hotels: {hotels.shape}, Bookings: {bookings.shape}")

    print("Merging datasets and preprocessing...")
    merged, hotel_ids = merge_data(hotels, bookings)

    print(f"Preprocessed data: {merged.shape}")

    print("Preprocessing features...")
    processed_data = preprocess_data(merged)

    print(f"Data with features: {processed_data.shape}")

    print("Preparing features for the model...")
    X, y, preprocessor = prepare_features(processed_data)

    print(f"X: {X.shape}, y: {y.shape}")
    print(f"Class distribution: {y.value_counts(normalize=True)}")

    print("Splitting data into train and test sets...")
    # Ajustado a 0.3 como solicitado
    X_train, X_test, y_train, y_test, hotel_ids_train, hotel_ids_test = train_test_split(
        X, y, hotel_ids, test_size=0.3, random_state=42, stratify=y
    )

    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

    print("Creating and evaluating XGBoost model...")
    xgb_pipeline, xgb_metrics = create_and_evaluate_model(
        X_train, X_test, y_train, y_test, preprocessor, hotel_ids_train
    )

    print("Creating and evaluating ensemble model...")
    ensemble_pipeline, ensemble_metrics = create_ensemble_model(
        X_train, X_test, y_train, y_test, preprocessor
    )

    print("Model comparison:")
    models_comparison = pd.DataFrame({
        'XGBoost': list(xgb_metrics.values()),
        'Ensemble': list(ensemble_metrics.values())
    }, index=list(xgb_metrics.keys()))

    print(models_comparison)

    best_model = xgb_pipeline if xgb_metrics['F1 Score'] > ensemble_metrics['F1 Score'] else ensemble_pipeline
    best_model_name = "XGBoost" if xgb_metrics['F1 Score'] > ensemble_metrics['F1 Score'] else "Ensemble"

    print(f"Best model: {best_model_name}")

    save_model(best_model, f'models/model_{best_model_name.lower()}.pkl')
    print("Model training and evaluation complete.")

if __name__ == "__main__":
    main()