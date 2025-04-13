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

# Load hotel and booking data
def load_data():
    # Use paths relative to the project root
    hotels = pd.read_csv('/app/data/hotels.csv')
    bookings = pd.read_csv('/app/data/bookings_train.csv')
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
    thresholds = np.linspace(0.1, 0.9, 200)
    best_threshold, best_f1 = 0.5, 0

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1, best_threshold = f1, threshold

    return best_threshold, best_f1

# Filter out zero-importance features with an adaptive threshold
def filter_zero_importance_features(model, feature_names, X_train_transformed, X_test_transformed):
    importance_scores = model.feature_importances_

    # Using percentile to select features
    importance_threshold = np.percentile(importance_scores, 15)  # Keep top 85%
    important_feature_indices = np.where(importance_scores > importance_threshold)[0]

    # Ensure we keep at least a minimum number of features
    min_features = max(10, int(X_train_transformed.shape[1] * 0.5))
    if len(important_feature_indices) < min_features:
        important_feature_indices = np.argsort(importance_scores)[-min_features:]

    X_train_array = np.array(X_train_transformed)
    X_test_array = np.array(X_test_transformed)

    X_train_filtered = X_train_array[:, important_feature_indices]
    X_test_filtered = X_test_array[:, important_feature_indices]

    return X_train_filtered, X_test_filtered, important_feature_indices

# Function to train a model with specified parameters on specific GPU
def train_model_on_gpu(param_set, X_train, y_train, X_eval, y_eval, gpu_id):
    # Set specific GPU device
    param_set = param_set.copy()
    param_set.update({
        'tree_method': 'gpu_hist',
        'gpu_id': gpu_id,
        'objective': 'binary:logistic',
        'random_state': 42
    })

    # Train model
    model = xgb.XGBClassifier(**param_set)
    model.fit(
        X_train, y_train,
        eval_set=[(X_eval, y_eval)],
        verbose=False
    )

    # Predict and calculate F1 score
    y_pred_proba = model.predict_proba(X_eval)[:, 1]
    best_threshold, best_f1 = find_optimal_threshold(y_eval, y_pred_proba)

    return {
        'params': param_set,
        'model': model,
        'f1_score': best_f1,
        'threshold': best_threshold
    }

# Perform custom parallel search using both CPU and GPU
def custom_parallel_search(X_train, X_test, y_train, y_test, preprocessor, num_iterations=30):
    print("Starting parallelized GPU/CPU search to find the best XGBoost parameters...")

    # Preprocess the data
    preprocessor.fit(X_train)
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Balance the data
    rus = RandomUnderSampler(sampling_strategy=0.30, random_state=42)
    X_train_under, y_train_under = rus.fit_resample(X_train_transformed, y_train)

    smote = SMOTE(sampling_strategy=0.65, random_state=42, k_neighbors=5)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_under, y_train_under)

    # Train initial model for feature importance
    init_model = xgb.XGBClassifier(
        objective='binary:logistic',
        tree_method='gpu_hist',  # Enable GPU acceleration
        gpu_id=0,                # Use first GPU
        random_state=42
    )

    init_model.fit(X_train_resampled, y_train_resampled)

    # Filter features based on importance
    feature_names = preprocessor.get_feature_names_out() if hasattr(preprocessor, 'get_feature_names_out') else [f'feature_{i}' for i in range(X_train_transformed.shape[1])]
    X_train_filtered, X_test_filtered, important_indices = filter_zero_importance_features(
        init_model, feature_names, X_train_resampled, X_test_transformed
    )

    # Define parameters focused on combating overfitting
    param_dist = {
        'max_depth': randint(2, 6),  # Lower depths to prevent overfitting
        'min_child_weight': randint(3, 10),  # Higher values prevent overfitting
        'gamma': uniform(0.1, 0.9),  # Higher values make algorithm more conservative
        'subsample': uniform(0.5, 0.4),  # Subsample ratio of training data
        'colsample_bytree': uniform(0.5, 0.4),  # Subsample ratio of columns
        'reg_alpha': uniform(0.3, 1.7),  # L1 regularization
        'reg_lambda': uniform(1.0, 4.0),  # L2 regularization
        'learning_rate': uniform(0.005, 0.095),  # Lower learning rates
        'n_estimators': randint(300, 700),  # Number of boosting rounds
        'scale_pos_weight': uniform(1.0, 3.0)  # Balance positive and negative weights
    }

    # Generate random parameter sets
    param_sets = []
    for _ in range(num_iterations):
        params = {
            'max_depth': int(randint.rvs(2, 6)),
            'min_child_weight': int(randint.rvs(3, 10)),
            'gamma': float(uniform.rvs(0.1, 0.9)),
            'subsample': float(uniform.rvs(0.5, 0.4)),
            'colsample_bytree': float(uniform.rvs(0.5, 0.4)),
            'reg_alpha': float(uniform.rvs(0.3, 1.7)),
            'reg_lambda': float(uniform.rvs(1.0, 4.0)),
            'learning_rate': float(uniform.rvs(0.005, 0.095)),
            'n_estimators': int(randint.rvs(300, 700)),
            'scale_pos_weight': float(uniform.rvs(1.0, 3.0))
        }
        param_sets.append(params)

    # Check available GPUs and CPU cores
    try:
        num_gpus = len(os.popen('nvidia-smi -L').read().strip().split('\n'))
    except:
        num_gpus = 1  # Default to 1 if cannot detect

    num_cpu_cores = multiprocessing.cpu_count()
    print(f"Detected {num_gpus} GPUs and {num_cpu_cores} CPU cores")

    # Distribute workload between GPU and CPU
    gpu_tasks = []
    cpu_tasks = []

    # Assign more tasks to GPUs but ensure CPU is also utilized
    gpu_task_ratio = 0.7  # 70% tasks on GPU, 30% on CPU
    gpu_tasks_count = int(len(param_sets) * gpu_task_ratio)

    for i, params in enumerate(param_sets):
        if i < gpu_tasks_count:
            # Distribute across available GPUs
            gpu_id = i % num_gpus
            gpu_tasks.append((params, X_train_filtered, y_train_resampled, X_test_filtered, y_test, gpu_id))
        else:
            # CPU tasks
            cpu_tasks.append((params, X_train_filtered, y_train_resampled, X_test_filtered, y_test))

    results = []

    # Process GPU tasks
    print(f"Processing {len(gpu_tasks)} tasks on GPU(s)...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_gpus) as executor:
        future_to_task = {
            executor.submit(train_model_on_gpu, *task): task for task in gpu_tasks
        }
        for future in concurrent.futures.as_completed(future_to_task):
            try:
                result = future.result()
                results.append(result)
                print(f"GPU task completed with F1: {result['f1_score']:.4f}")
            except Exception as e:
                print(f"GPU task failed: {e}")

    # Process CPU tasks with CPU-optimized XGBoost
    print(f"Processing {len(cpu_tasks)} tasks on CPU cores...")

    def train_model_on_cpu(param_set, X_train, y_train, X_eval, y_eval):
        param_set = param_set.copy()
        param_set.update({
            'tree_method': 'hist',  # Use histogram method for CPU
            'objective': 'binary:logistic',
            'random_state': 42,
            'n_jobs': max(1, num_cpu_cores // 4)  # Use multiple cores but don't saturate CPU
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

    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, num_cpu_cores // 4)) as executor:
        future_to_task = {
            executor.submit(train_model_on_cpu, *task[:-1]): task for task in cpu_tasks
        }
        for future in concurrent.futures.as_completed(future_to_task):
            try:
                result = future.result()
                results.append(result)
                print(f"CPU task completed with F1: {result['f1_score']:.4f}")
            except Exception as e:
                print(f"CPU task failed: {e}")

    # Find best result
    best_result = max(results, key=lambda x: x['f1_score'])

    print("Best Parameters Found: ", best_result['params'])
    print(f"Best F1 Score: {best_result['f1_score']:.4f}")

    # Create the final best model with the found parameters
    final_params = best_result['params'].copy()

    # Create model with best parameters but ensure it uses GPU for final training
    if 'tree_method' in final_params:
        final_params['tree_method'] = 'gpu_hist'
    if 'gpu_id' not in final_params:
        final_params['gpu_id'] = 0

    best_model = xgb.XGBClassifier(**final_params)

    # Train final model
    best_model.fit(
        X_train_filtered,
        y_train_resampled,
        eval_set=[(X_test_filtered, y_test)],
        verbose=False
    )

    # Evaluate final model
    y_pred_proba = best_model.predict_proba(X_test_filtered)[:, 1]
    best_threshold, best_f1 = find_optimal_threshold(y_test, y_pred_proba)
    y_pred_optimized = (y_pred_proba >= best_threshold).astype(int)

    # Calculate metrics
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred_optimized),
        'F1 Score': f1_score(y_test, y_pred_optimized),
        'Precision': precision_score(y_test, y_pred_optimized),
        'Recall': recall_score(y_test, y_pred_optimized),
        'ROC AUC': roc_auc_score(y_test, y_pred_proba),
        'Best Threshold': best_threshold
    }

    # Create filtered pipeline with the best model
    filtered_pipeline = FilteredPipeline(
        preprocessor=preprocessor,
        model=best_model,
        important_indices=important_indices,
        best_threshold=best_threshold
    )

    return filtered_pipeline, metrics, final_params

# Save the trained model to a file
def save_model(model, filename):
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
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
    X_train, X_test, y_train, y_test, hotel_ids_train, hotel_ids_test = train_test_split(
        X, y, hotel_ids, test_size=0.3, random_state=42, stratify=y
    )

    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

    print("Finding best XGBoost model with hybrid GPU-CPU optimization...")
    best_model, best_metrics, best_params = custom_parallel_search(
        X_train, X_test, y_train, y_test, preprocessor, num_iterations=40
    )

    print("Model metrics:")
    for metric, value in best_metrics.items():
        print(f"{metric}: {value}")

    print("\nBest parameters:")
    for param, value in best_params.items():
        print(f"{param}: {value}")

    models_path = '/app/models/xgboost_hybrid_optimized_best.pkl'
    save_model(best_model, models_path)
    print("Model training and evaluation complete.")

if __name__ == "__main__":
    main()