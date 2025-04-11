import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GroupKFold
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

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

# Load hotel and booking data
def load_data():
    hotels = pd.read_csv('data/hotels.csv')
    bookings = pd.read_csv('data/bookings_train.csv')
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

    # Extract transport features
    data['requested_parking'] = (data['required_car_parking_spaces'] > 0).astype(int)

    # Drop columns that may cause data leakage
    columns_to_drop = ['reservation_status', 'reservation_status_date', 'booking_date', 'arrival_date', 'days_before_arrival']
    data.drop(columns=columns_to_drop, inplace=True)

    # Handle infinite and null values in numerical features
    for col in ['price_per_night', 'price_per_person', 'special_requests_ratio']:
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
    thresholds = np.linspace(0.2, 0.8, 100)
    best_threshold, best_f1 = 0.5, 0

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1, best_threshold = f1, threshold

    return best_threshold, best_f1

# Filter out zero-importance features
def filter_zero_importance_features(model, feature_names, X_train_transformed, X_test_transformed):
    importance_scores = model.feature_importances_
    important_feature_indices = np.where(importance_scores > 0)[0]
    important_feature_names = [feature_names[i] for i in important_feature_indices]

    X_train_array = np.array(X_train_transformed)
    X_test_array = np.array(X_test_transformed)

    X_train_filtered = X_train_array[:, important_feature_indices]
    X_test_filtered = X_test_array[:, important_feature_indices]

    print(f"Original features: {len(feature_names)}")
    print(f"Removed features: {len(feature_names) - len(important_feature_indices)} (importance = 0)")
    print(f"Remaining features: {len(important_feature_indices)}")

    return X_train_filtered, X_test_filtered, important_feature_indices, important_feature_names

# Create and evaluate the XGBoost model
def create_and_evaluate_model(X_train, X_test, y_train, y_test, preprocessor, hotel_ids_train=None):
    preprocessor.fit(X_train)
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Balance the data
    rus = RandomUnderSampler(sampling_strategy=0.3, random_state=42)
    X_train_under, y_train_under = rus.fit_resample(X_train_transformed, y_train)

    smote = SMOTE(sampling_strategy=0.9, random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_under, y_train_under)

    # Evaluate with GroupKFold
    if hotel_ids_train is not None:
        cv = GroupKFold(n_splits=5)
        groups = hotel_ids_train

        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            max_depth=3,
            min_child_weight=6,
            gamma=0.3,
            subsample=0.6,
            colsample_bytree=0.6,
            reg_alpha=0.2,
            reg_lambda=1.5,
            scale_pos_weight=2.5,
            learning_rate=0.03,
            n_estimators=250,
            random_state=42
        )

        cv_scores = []
        print("\nEvaluating model with GroupKFold by hotel_id:")
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

        print(f"F1 scores in CV: {cv_scores}")
        print(f"Mean F1 CV: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

    # Train the initial model for feature importance
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        max_depth=3,
        min_child_weight=6,
        gamma=0.3,
        subsample=0.6,
        colsample_bytree=0.6,
        reg_alpha=0.2,
        reg_lambda=1.5,
        scale_pos_weight=2.5,
        learning_rate=0.03,
        n_estimators=250,
        random_state=42
    )

    # Debugging: Print shapes before fitting
    print(f"Shape of X_train_resampled: {X_train_resampled.shape}")
    print(f"Shape of y_train_resampled: {y_train_resampled.shape}")

    model.fit(
        X_train_resampled,
        y_train_resampled,
        eval_set=[(X_test_transformed, y_test)],
        verbose=False,
    )

    feature_names = preprocessor.get_feature_names_out() if hasattr(preprocessor, 'get_feature_names_out') else [f'feature_{i}' for i in range(X_train_transformed.shape[1])]

    # Filter zero-importance features from both train_transformed and test_transformed
    X_train_transformed_filtered, X_test_filtered, important_indices, important_features = filter_zero_importance_features(
        model, feature_names, X_train_transformed, X_test_transformed
    )

    # Apply the same filter to the resampled data
    X_train_resampled_array = np.array(X_train_resampled)
    X_train_resampled_filtered = X_train_resampled_array[:, important_indices]

    filtered_model = xgb.XGBClassifier(
        objective='binary:logistic',
        max_depth=3,
        min_child_weight=6,
        gamma=0.3,
        subsample=0.6,
        colsample_bytree=0.6,
        reg_alpha=0.2,
        reg_lambda=1.5,
        scale_pos_weight=2.5,
        learning_rate=0.03,
        n_estimators=250,
        random_state=42
    )

    # Debugging: Print shapes before fitting the filtered model
    print(f"Shape of X_train_resampled_filtered: {X_train_resampled_filtered.shape}")
    print(f"Shape of y_train_resampled: {y_train_resampled.shape}")

    filtered_model.fit(
        X_train_resampled_filtered,  # Use filtered AND resampled data
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

    print("\nMetrics for the filtered model (without zero-importance features):")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    print("\nClassification report (optimized threshold):")
    print(classification_report(y_test, y_pred_optimized))

    importance_scores = filtered_model.feature_importances_

    feature_imp = pd.DataFrame({
        'Feature': important_features,
        'Importance': importance_scores
    }).sort_values('Importance', ascending=False)

    print("\nTop 20 most important features:")
    print(feature_imp.head(20))

    os.makedirs('data', exist_ok=True)
    feature_imp.to_csv('data/features_importance_improved.csv', index=False)
    print("Important features saved in 'data/features_importance_improved.csv'")

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

    rus = RandomUnderSampler(sampling_strategy=0.3, random_state=42)
    X_train_under, y_train_under = rus.fit_resample(X_train_transformed, y_train)

    smote = SMOTE(sampling_strategy=0.9, random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_under, y_train_under)

    xgb1 = xgb.XGBClassifier(
        max_depth=3,
        learning_rate=0.03,
        n_estimators=250,
        subsample=0.6,
        colsample_bytree=0.6,
        scale_pos_weight=4.0,
        min_child_weight=6,
        gamma=0.2,
        reg_alpha=0.2,
        reg_lambda=1.5,
        random_state=42,
        eval_metric='logloss'
    )

    xgb2 = xgb.XGBClassifier(
        max_depth=2,
        learning_rate=0.02,
        n_estimators=300,
        subsample=0.7,
        colsample_bytree=0.7,
        scale_pos_weight=1.5,
        min_child_weight=8,
        gamma=0.3,
        reg_alpha=0.3,
        reg_lambda=2.0,
        random_state=42,
        eval_metric='logloss'
    )

    xgb3 = xgb.XGBClassifier(
        max_depth=3,
        learning_rate=0.03,
        n_estimators=275,
        subsample=0.65,
        colsample_bytree=0.65,
        scale_pos_weight=2.5,
        min_child_weight=7,
        gamma=0.25,
        reg_alpha=0.25,
        reg_lambda=1.8,
        random_state=42,
        eval_metric='logloss'
    )

    xgb1.fit(X_train_resampled, y_train_resampled,
             eval_set=[(X_test_transformed, y_test)],
             verbose=False)

    xgb2.fit(X_train_resampled, y_train_resampled,
             eval_set=[(X_test_transformed, y_test)],
             verbose=False)

    xgb3.fit(X_train_resampled, y_train_resampled,
             eval_set=[(X_test_transformed, y_test)],
             verbose=False)

    feature_names = preprocessor.get_feature_names_out() if hasattr(preprocessor, 'get_feature_names_out') else [f'feature_{i}' for i in range(X_train_transformed.shape[1])]

    importance1 = xgb1.feature_importances_
    importance2 = xgb2.feature_importances_
    importance3 = xgb3.feature_importances_

    combined_importance = np.maximum.reduce([importance1, importance2, importance3])
    important_feature_indices = np.where(combined_importance > 0)[0]
    important_feature_names = [feature_names[i] for i in important_feature_indices]

    print(f"\nOriginal features in ensemble: {len(feature_names)}")
    print(f"Removed features: {len(feature_names) - len(important_feature_indices)} (importance = 0 in all models)")
    print(f"Remaining features: {len(important_feature_indices)}")

    X_train_array = np.array(X_train_resampled)
    X_test_array = np.array(X_test_transformed)

    X_train_filtered = X_train_array[:, important_feature_indices]
    X_test_filtered = X_test_array[:, important_feature_indices]

    xgb1_filtered = xgb.XGBClassifier(
        max_depth=3, learning_rate=0.03, n_estimators=250,
        subsample=0.6, colsample_bytree=0.6, scale_pos_weight=4.0,
        min_child_weight=6, gamma=0.2, reg_alpha=0.2, reg_lambda=1.5,
        random_state=42, eval_metric='logloss'
    )

    xgb2_filtered = xgb.XGBClassifier(
        max_depth=2, learning_rate=0.02, n_estimators=300,
        subsample=0.7, colsample_bytree=0.7, scale_pos_weight=1.5,
        min_child_weight=8, gamma=0.3, reg_alpha=0.3, reg_lambda=2.0,
        random_state=42, eval_metric='logloss'
    )

    xgb3_filtered = xgb.XGBClassifier(
        max_depth=3, learning_rate=0.03, n_estimators=275,
        subsample=0.65, colsample_bytree=0.65, scale_pos_weight=2.5,
        min_child_weight=7, gamma=0.25, reg_alpha=0.25, reg_lambda=1.8,
        random_state=42, eval_metric='logloss'
    )

    xgb1_filtered.fit(X_train_filtered, y_train_resampled,
                      eval_set=[(X_test_filtered, y_test)],
                      verbose=False)

    xgb2_filtered.fit(X_train_filtered, y_train_resampled,
                      eval_set=[(X_test_filtered, y_test)],
                      verbose=False)

    xgb3_filtered.fit(X_train_filtered, y_train_resampled,
                      eval_set=[(X_test_filtered, y_test)],
                      verbose=False)

    ensemble_filtered = VotingClassifier(
        estimators=[
            ('xgb_recall', xgb1_filtered),
            ('xgb_precision', xgb2_filtered),
            ('xgb_balanced', xgb3_filtered)
        ],
        voting='soft',
        weights=[1, 1, 2]
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

    print("\nMetrics for the filtered ensemble:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    print("\nClassification report for the ensemble (optimized threshold):")
    print(classification_report(y_test, y_pred_optimized))

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

    filtered_ensemble_pipeline = FilteredEnsemblePipeline(
        preprocessor=preprocessor,
        ensemble=ensemble_filtered,
        important_indices=important_feature_indices,
        best_threshold=best_threshold
    )

    all_importances = []
    for name, model in [('xgb_recall', xgb1_filtered), ('xgb_precision', xgb2_filtered), ('xgb_balanced', xgb3_filtered)]:
        importances = model.feature_importances_
        for i, (feature, importance) in enumerate(zip(important_feature_names, importances)):
            all_importances.append({
                'Model': name,
                'Feature': feature,
                'Importance': importance
            })

    feature_imp_df = pd.DataFrame(all_importances)
    feature_imp_df.to_csv('data/ensemble_features_importance.csv', index=False)
    print("Ensemble feature importance saved in 'data/ensemble_features_importance.csv'")

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

    print("\nMerging datasets and preprocessing...")
    merged, hotel_ids = merge_data(hotels, bookings)

    print(f"Preprocessed data: {merged.shape}")

    print("\nPreprocessing features...")
    processed_data = preprocess_data(merged)

    print(f"Data with features: {processed_data.shape}")

    print("\nPreparing features for the model...")
    X, y, preprocessor = prepare_features(processed_data)

    print(f"X: {X.shape}, y: {y.shape}")
    print(f"Class distribution: {y.value_counts(normalize=True)}")

    print("\nSplitting data into train and test sets...")
    X_train, X_test, y_train, y_test, hotel_ids_train, hotel_ids_test = train_test_split(
        X, y, hotel_ids, test_size=0.2, random_state=42, stratify=y
    )

    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

    print("\nCreating and evaluating XGBoost model...")
    xgb_pipeline, xgb_metrics = create_and_evaluate_model(
        X_train, X_test, y_train, y_test, preprocessor, hotel_ids_train
    )

    print("\nCreating and evaluating ensemble model...")
    ensemble_pipeline, ensemble_metrics = create_ensemble_model(
        X_train, X_test, y_train, y_test, preprocessor
    )

    print("\nModel comparison:")
    models_comparison = pd.DataFrame({
        'XGBoost': list(xgb_metrics.values()),
        'Ensemble': list(ensemble_metrics.values())
    }, index=list(xgb_metrics.keys()))

    print(models_comparison)

    best_model = xgb_pipeline if xgb_metrics['F1 Score'] > ensemble_metrics['F1 Score'] else ensemble_pipeline
    best_model_name = "XGBoost" if xgb_metrics['F1 Score'] > ensemble_metrics['F1 Score'] else "Ensemble"

    print(f"\nBest model: {best_model_name}")

    save_model(best_model, f'model/best_cancellation_model_{best_model_name.lower()}.pkl')

    print("\nProcess completed successfully!")

    return best_model

if __name__ == "__main__":
    os.makedirs('model', exist_ok=True)
    model = main()
