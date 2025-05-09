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

    hotels = pd.read_csv(os.getenv("HOTELS_DATA_PATH", "data/hotels.csv"))
    bookings = pd.read_csv(os.getenv("TRAIN_DATA_PATH", "data/bookings_train.csv"))

    data = bookings.merge(hotels, on='hotel_id', how='left')
    data = data[data['reservation_status'] != 'Booked'].copy()

    data['reservation_status'].replace('No-Show', 'Check-Out', inplace=True)
    date_cols = ['arrival_date', 'booking_date', 'reservation_status_date']
    data[date_cols] = data[date_cols].apply(pd.to_datetime)

    data['days_before_arrival'] = (data['arrival_date'] - data['reservation_status_date']).dt.days
    data['lead_time'] = (data['arrival_date'] - data['booking_date']).dt.days
    data['target'] = ((data['reservation_status'] == 'Canceled') &
                      (data['days_before_arrival'] <= 30)).astype(int)

    drop_cols = ['reservation_status', 'reservation_status_date', 'days_before_arrival',
                 'arrival_date', 'booking_date']
    X = data.drop(columns=['target'] + [c for c in drop_cols if c in data])
    y = data['target']
    hotel_ids = data['hotel_id']
    return X, y, hotel_ids

def create_pipeline(X):
    numeric_features = X.select_dtypes(include=['int64', 'float64', 'bool']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    model = XGBClassifier(
        n_estimators=250,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss',
        tree_method='hist',
        random_state=42
    )

    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', model)
    ])

    return pipeline

def find_threshold(y_true, y_proba, beta=1.0):
    """Encuentra el mejor umbral según F-beta score."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    beta_sq = beta ** 2
    fbeta = (1 + beta_sq) * precision * recall / (beta_sq * precision + recall + 1e-7)
    idx = np.argmax(fbeta[:-1])
    return thresholds[idx], fbeta[idx]

def cv_threshold(X, y, groups, pipeline, cv, beta=1.0):
    """Calcula el umbral óptimo global para F-beta score con CV."""
    print("Buscando umbral óptimo con CV...")
    all_true, all_proba = [], []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        pipeline.fit(X_train, y_train)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        all_true.extend(y_test)
        all_proba.extend(y_proba)

    threshold, fbeta = find_threshold(np.array(all_true), np.array(all_proba), beta=beta)
    print(f"Umbral óptimo global (F{beta}-score): {threshold:.4f}")
    print(f"F{beta}-score global: {fbeta:.4f}")
    return threshold


def save_model(pipeline, threshold, path=None):
    """Guarda el pipeline entrenado y el umbral."""
    model_path = path or os.getenv("MODEL_PATH", "models/pipeline.cloudpkl")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        cloudpickle.dump({'pipeline': pipeline, 'threshold': threshold}, f)
    print(f"Modelo guardado en {model_path}")

def main():
    print("Iniciando entrenamiento...")
    X, y, groups = get_X_y()
    pipeline = create_pipeline(X)
    cv = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=42)
    threshold = cv_threshold(X, y, groups, pipeline, cv)

    print("\nEntrenando modelo final...")
    pipeline.fit(X, y)
    save_model(pipeline, threshold)

    print(f"\nUmbral óptimo final: {threshold:.4f}")
    print("Entrenamiento completado. ¡Listo para inferencia!")

if __name__ == "__main__":
    main()