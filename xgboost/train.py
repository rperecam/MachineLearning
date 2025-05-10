import os
import warnings
import numpy as np
import pandas as pd
import cloudpickle
from sklearn import set_config
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedGroupKFold
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Configuración global
set_config(transform_output="pandas")
warnings.filterwarnings('ignore')

def get_X_y():
    """Carga datos, construye la variable objetivo y prepara los features."""
    print("Cargando datos de entrenamiento...")

    hotels_path = os.getenv("HOTELS_DATA_PATH", "data/hotels.csv")
    bookings_train_path = os.getenv("TRAIN_DATA_PATH", "data/bookings_train.csv")

    hotels = pd.read_csv(hotels_path)
    bookings = pd.read_csv(bookings_train_path)

    data = bookings.merge(hotels, on='hotel_id', how='left')
    data = data[data['reservation_status'] != 'Booked'].copy()
    data['reservation_status'].replace('No-Show', 'Check-Out', inplace=True)

    date_cols = ['arrival_date', 'booking_date', 'reservation_status_date']
    for col in date_cols:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], errors='coerce')

    data['required_car_parking_spaces'] = data['required_car_parking_spaces'].fillna(0)
    data['lead_time'] = (data['arrival_date'] - data['booking_date']).dt.days

    data['days_before_arrival'] = (data['arrival_date'] - data['reservation_status_date']).dt.days
    data['target'] = ((data['reservation_status'] == 'Canceled') &
                      (data['days_before_arrival'] <= 30)).astype(int)

    drop_cols = ['reservation_status', 'reservation_status_date', 'days_before_arrival',
                 'arrival_date', 'booking_date']

    X = data.drop(columns=['target', 'hotel_id'] + [c for c in drop_cols if c in data.columns])
    y = data['target']
    hotel_ids = data['hotel_id']

    print(f"Datos cargados y preprocesados: {X.shape[0]} registros, {X.shape[1]} características.")
    return X, y, hotel_ids

def get_pipeline(X_sample):
    """Crea el pipeline de preprocesamiento y el modelo."""
    numeric_features = X_sample.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X_sample.select_dtypes(include=['object', 'category']).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ], remainder='passthrough')

    model = XGBClassifier(
        objective='binary:logistic',
        tree_method='hist',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42,
        min_child_weight=4,
        learning_rate=0.05,
        n_estimators=300,
        gamma=0.1,
    )

    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42, k_neighbors=5, sampling_strategy=0.8)),
        ('classifier', model)
    ])

    return pipeline

def find_threshold(y_true, y_proba, beta=0.5):
    """Encuentra el mejor umbral según F-beta score."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    fbeta = np.zeros_like(precision)
    valid_indices = (precision + recall > 0)
    beta_sq = beta ** 2
    fbeta[valid_indices] = (1 + beta_sq) * precision[valid_indices] * recall[valid_indices] / \
                           (beta_sq * precision[valid_indices] + recall[valid_indices])

    if len(thresholds) == 0:
        return 0.5, 0.0

    idx = np.argmax(fbeta[:-1])
    return thresholds[idx], fbeta[idx]

def cross_validate_threshold(X, y, groups, pipeline_template, cv_splitter, beta=1.0):
    """Calcula el umbral óptimo global para F-beta score usando validación cruzada."""
    print(f"Buscando umbral óptimo con CV (F{beta}-score)...")
    all_true_labels, all_predicted_probas = [], []
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(cv_splitter.split(X, y, groups)):
        print(f"  Fold {fold + 1}/{cv_splitter.get_n_splits()}...")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        pipeline_template.fit(X_train, y_train)
        y_val_proba = pipeline_template.predict_proba(X_val)[:, 1]

        all_true_labels.extend(y_val)
        all_predicted_probas.extend(y_val_proba)

    optimal_threshold, global_fbeta = find_threshold(np.array(all_true_labels),
                                                             np.array(all_predicted_probas), beta=beta)

    print(f"\nResultados de la búsqueda de umbral por fold:")
    for fm in fold_metrics:
        print(f"  Fold {fm['fold']}: Umbral={fm['threshold']:.4f}, F{beta}-score: {fm[f'f{beta}_score']:.4f}")

    print(f"\nUmbral óptimo global (basado en todas las preds de CV) (F{beta}-score): {optimal_threshold:.4f}")
    print(f"F{beta}-score global (basado en todas las preds de CV): {global_fbeta:.4f}")
    return optimal_threshold

def save_pipeline(pipeline, threshold, path=None):
    """Guarda el pipeline entrenado y el umbral."""
    model_path = path or os.getenv("MODEL_PATH", "models/pipeline.cloudpkl")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    package_to_save = {'pipeline': pipeline, 'threshold': threshold}

    with open(model_path, "wb") as f:
        cloudpickle.dump(package_to_save, f)
    print(f"Paquete de modelo (pipeline y umbral) guardado en {model_path}")

def main():
    print("Iniciando proceso de entrenamiento...")
    X, y, hotel_ids_for_grouping = get_X_y()

    if X.empty:
        print("No hay datos para entrenar después del preprocesamiento. Abortando.")
        return

    pipeline_template = get_pipeline(X.head(1))
    cv_splitter = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=42)

    optimal_threshold = cross_validate_threshold(X, y, hotel_ids_for_grouping, pipeline_template, cv_splitter, beta=1.0)

    print("\nEntrenando modelo final con todos los datos de entrenamiento...")
    final_pipeline = get_pipeline(X)
    final_pipeline.fit(X, y)

    save_pipeline(final_pipeline, optimal_threshold)

    print(f"\nUmbral óptimo final calculado y guardado: {optimal_threshold:.4f}")
    print("Proceso de entrenamiento completado.")

if __name__ == "__main__":
    main()