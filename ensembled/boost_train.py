import os
import time
import warnings
import numpy as np
import pandas as pd
import cloudpickle
from sklearn import set_config
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Configuración
set_config(transform_output="pandas")
warnings.filterwarnings('ignore')

def get_X_y():
    """Carga y preprocesa los datos para el entrenamiento."""
    print("Cargando datos...")

    # Cargar datos
    hotels = pd.read_csv(os.environ.get("HOTELS_DATA_PATH"))
    bookings = pd.read_csv(os.environ.get("TRAIN_DATA_PATH"))

    # Combinar datos
    data = pd.merge(bookings, hotels, on='hotel_id', how='left')

    # Filtrar datos relevantes - excluir reservas que aún están en estado 'Booked'
    data = data[data['reservation_status'] != 'Booked'].copy()

    # Interpreto los NoShow como Check-Out, ya que realmente han pagado
    data['reservation_status'] = data['reservation_status'].replace('No-Show', 'Check-Out')

    # Convertir fechas
    date_columns = ['arrival_date', 'booking_date', 'reservation_status_date']
    for col in date_columns:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col])

    # Crear características clave
    data['days_before_arrival'] = (data['arrival_date'] - data['reservation_status_date']).dt.days
    data['target'] = ((data['reservation_status'] == 'Canceled') &
                      (data['days_before_arrival'] >= 30)).astype(int)

    data['lead_time'] = (data['arrival_date'] - data['booking_date']).dt.days
    data['is_high_season'] = data['arrival_date'].dt.month.isin([6, 7, 8, 12]).astype(int)
    data['is_weekend_arrival'] = data['arrival_date'].dt.dayofweek.isin([4, 5]).astype(int)
    data['price_per_night'] = data['rate'] / np.maximum(data['stay_nights'], 1)
    data['has_special_requests'] = (data['special_requests'] > 0).astype(int)

    # Característica de cliente extranjero si existen las columnas
    if 'country_x' in data.columns and 'country_y' in data.columns:
        data['is_foreign'] = (data['country_x'].astype(str) != data['country_y'].astype(str)).astype(int)
        data.loc[data['country_x'].isna() | data['country_y'].isna(), 'is_foreign'] = 0

    # Eliminar columnas que podrían causar data leakage
    columns_to_drop = [
        'reservation_status', 'reservation_status_date', 'days_before_arrival',
        'arrival_date', 'booking_date', 'special_requests', 'stay_nights',
        'country_y', 'country_x'
    ]
    columns_to_drop = [col for col in columns_to_drop if col in data.columns]

    X = data.drop(columns=['target'] + columns_to_drop)
    y = data['target']

    print(f"Datos cargados: {X.shape[0]} registros, {X.shape[1]} características")
    return X, y

def create_preprocessor(X: pd.DataFrame):
    """Crea un preprocesador que incluye la imputación de valores nulos."""
    num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    bool_features = X.select_dtypes(include=["bool"]).columns.tolist()
    cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    cat_features = [col for col in cat_features if col not in bool_features]

    # Pipelines para cada tipo de característica
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # Si hay características booleanas, las mantenemos como están
    bool_transformer = 'passthrough'

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_features),
            ("cat", categorical_transformer, cat_features),
            ("bool", bool_transformer, bool_features),
        ]
    )

    return preprocessor

def create_base_model(model_type='xgboost', pos_weight=1.0, random_state=42):
    """Crea un modelo base con hiperparámetros originales."""
    model_params = {
        'xgboost': {
            'objective': 'binary:logistic',
            'tree_method': 'hist',
            'eval_metric': 'logloss',
            'use_label_encoder': False,
            'scale_pos_weight': pos_weight,
            'max_depth': 6,
            'n_estimators': 150,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_lambda': 1.0,
            'reg_alpha': 0.1,
            'verbosity': 0,
            'random_state': random_state
        },
        'lightgbm': {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'scale_pos_weight': pos_weight,
            'n_estimators': 150,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'verbose': -1,
            'random_state': random_state
        },
        'rf': {
            'n_estimators': 150,
            'max_depth': 10,
            'min_samples_split': 10,
            'min_samples_leaf': 4,
            'class_weight': 'balanced',
            'random_state': random_state,
            'n_jobs': -1
        },
        'gbm': {
            'n_estimators': 150,
            'learning_rate': 0.05,
            'max_depth': 6,
            'min_samples_split': 10,
            'min_samples_leaf': 4,
            'subsample': 0.8,
            'random_state': random_state
        }
    }

    if model_type not in model_params:
        raise ValueError(f"Tipo de modelo no soportado: {model_type}")

    if model_type == 'xgboost':
        return XGBClassifier(**model_params[model_type])
    elif model_type == 'lightgbm':
        return LGBMClassifier(**model_params[model_type])
    elif model_type == 'rf':
        return RandomForestClassifier(**model_params[model_type])
    elif model_type == 'gbm':
        return GradientBoostingClassifier(**model_params[model_type])

class StackingEnsemble(BaseEstimator, ClassifierMixin):
    """Ensemble simplificado de modelos mediante stacking."""

    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.fold_models = []
        self.meta_model = None
        self.model_types = ['xgboost', 'lightgbm', 'rf', 'gbm']  # Modelos a usar

    def fit(self, X, y):
        """Entrena múltiples modelos usando validación cruzada."""
        n_splits = 7 # Muy importante
        groups = X['hotel_id'].copy() if 'hotel_id' in X.columns else None
        X = X.drop(columns=['hotel_id']) if groups is not None else X

        group_cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_splits = list(group_cv.split(X, y, groups)) if groups is not None else list(group_cv.split(X, y))

        n_samples = X.shape[0]
        n_models_per_fold = len(self.model_types)
        oof_preds = np.zeros((n_samples, n_splits * n_models_per_fold))

        start_time = time.time()
        self.fold_models = []

        for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
            print(f"Entrenando fold {fold_idx + 1}/{n_splits}...")
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            fold_models = []

            for i, model_type in enumerate(self.model_types):
                preprocessor = create_preprocessor(X_train)
                pos_weight = len(y_train[y_train == 0]) / max(len(y_train[y_train == 1]), 1)
                classifier = create_base_model(model_type, pos_weight, random_state=42 + fold_idx * 10 + i)

                pipeline = ImbPipeline(steps=[
                    ("preprocessor", preprocessor),
                    ("sampling", SMOTE(random_state=42 + fold_idx, k_neighbors=3)),
                    ("classifier", classifier)
                ])

                pipeline.fit(X_train, y_train)
                fold_models.append(pipeline)

                val_preds_proba = pipeline.predict_proba(X_val)[:, 1]
                col_idx = fold_idx * n_models_per_fold + i
                oof_preds[val_idx, col_idx] = val_preds_proba

            self.fold_models.append(fold_models)

        self.meta_model = LogisticRegression(
            C=0.6,
            class_weight='balanced',
            solver='liblinear',
            max_iter=1000,
            random_state=42
        )
        self.meta_model.fit(oof_preds, y)

        total_time = time.time() - start_time
        print(f"Entrenamiento completado en {total_time:.2f} segundos")

        return self

    def predict_proba(self, X):
        """Genera probabilidades combinando las predicciones de todos los modelos base."""
        if not self.fold_models or self.meta_model is None:
            raise ValueError("El modelo no ha sido entrenado. Llama a fit() primero.")

        groups = X['hotel_id'].copy() if 'hotel_id' in X.columns else None
        X = X.drop(columns=['hotel_id']) if groups is not None else X

        n_models_per_fold = len(self.model_types)
        n_folds = len(self.fold_models)
        all_preds = np.zeros((X.shape[0], n_folds * n_models_per_fold))

        for fold_idx, fold_models in enumerate(self.fold_models):
            for model_idx, model in enumerate(fold_models):
                col_idx = fold_idx * n_models_per_fold + model_idx
                all_preds[:, col_idx] = model.predict_proba(X)[:, 1]

        meta_probs = self.meta_model.predict_proba(all_preds)
        return np.column_stack((1 - meta_probs[:, 1], meta_probs[:, 1]))

    def predict(self, X):
        """Predice la clase aplicando el umbral a las probabilidades."""
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)

def find_optimal_threshold(y_true, y_pred_proba):
    """Encuentra el umbral óptimo que maximiza el F1-score."""
    thresholds = np.linspace(0.05, 0.99, 50)
    best_threshold, best_f1 = 0.5, 0

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold


def get_pipeline():
    """Retorna el pipeline de modelo stacking."""
    return StackingEnsemble()

def save_pipeline(pipe):
    """Guarda el modelo entrenado."""
    model_path = os.environ.get("MODEL_PATH")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, mode="wb") as f:
        cloudpickle.dump(pipe, f)
    print(f"Modelo guardado en {model_path}")

if __name__ == "__main__":
    print("=" * 60)
    print("PREDICCIÓN DE CANCELACIONES ANTICIPADAS DE RESERVAS DE HOTEL")
    print("=" * 60)

    X, y = get_X_y()

    # Dividir en entrenamiento y validación para evaluar el rendimiento final
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Obtener y entrenar el pipeline
    pipe = get_pipeline()
    pipe.fit(X_train, y_train)

    # Encontrar el umbral óptimo en validación
    val_proba = pipe.predict_proba(X_val)[:, 1]
    optimal_threshold = find_optimal_threshold(y_val, val_proba)
    pipe.threshold = optimal_threshold
    print(f"Umbral óptimo encontrado: {optimal_threshold:.4f}")

    # Evaluar en validación
    val_pred = pipe.predict(X_val)
    val_f1 = f1_score(y_val, val_pred)
    val_precision = precision_score(y_val, val_pred)
    val_recall = recall_score(y_val, val_pred)
    val_accuracy = accuracy_score(y_val, val_pred)
    val_auc = roc_auc_score(y_val, val_proba)

    print("\nRendimiento en validación:")
    print(f"Precisión: {val_precision:.4f}")
    print(f"Recall: {val_recall:.4f}")
    print(f"F1-Score: {val_f1:.4f}")
    print(f"Accuracy: {val_accuracy:.4f}")
    print(f"AUC-ROC: {val_auc:.4f}")

    # Guardar el modelo ya entrenado con los datos de entrenamiento
    save_pipeline(pipe)

    print("\n¡Modelo entrenado y guardado con éxito!")