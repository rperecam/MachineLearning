import os
import pandas as pd
import numpy as np
import cloudpickle
from sklearn import set_config
from sklearn.model_selection import StratifiedGroupKFold, RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, make_scorer
import warnings
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
import multiprocessing.resource_tracker
from copy import deepcopy

# Parche para silenciar "ChildProcessError: No child processes"
def silence_resource_tracker():
    def noop(*args, **kwargs):
        pass
    multiprocessing.resource_tracker._cleanup = noop
    multiprocessing.resource_tracker._unregister = noop
    warnings.filterwarnings("ignore", category=UserWarning)

silence_resource_tracker()

# Configuración
set_config(transform_output="pandas")
warnings.filterwarnings('ignore')


def get_X_y():
    """Carga y preprocesa los datos para el entrenamiento."""
    print("Cargando datos...")

    # Cargar datos
    hotels = pd.read_csv(os.environ.get("HOTELS_DATA_PATH", 'data/hotels.csv'))
    bookings = pd.read_csv(os.environ.get("TRAIN_DATA_PATH", 'data/bookings_train.csv'))

    # Combinar datos
    data = pd.merge(bookings, hotels, on='hotel_id', how='left')

    # Filtrar datos relevantes
    data = data[data['reservation_status'] != 'Booked'].copy()

    # Convertir fechas
    for col in ['arrival_date', 'booking_date', 'reservation_status_date']:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col])

    # Target: cancelación con al menos 30 días de anticipación
    data['days_before_arrival'] = (data['arrival_date'] - data['reservation_status_date']).dt.days
    data['target'] = ((data['reservation_status'] == 'Canceled') &
                      (data['days_before_arrival'] >= 30)).astype(int)

    hotel_cancellation_rate = data.groupby('hotel_id')['target'].mean()
    data['hotel_cancel_rate'] = data['hotel_id'].map(hotel_cancellation_rate)

    # Crear características clave
    data['lead_time'] = (data['arrival_date'] - data['booking_date']).dt.days
    data['is_high_season'] = data['arrival_date'].dt.month.isin([6, 7, 8, 12]).astype(int)
    data['is_weekend_arrival'] = data['arrival_date'].dt.dayofweek.isin([4, 5]).astype(int)
    data['price_per_night'] = data['rate'] / np.maximum(data['stay_nights'], 1)
    data['stay_duration_category'] = pd.cut(data['stay_nights'],
                                            bins=[-1, 1, 3, 7, 14, float('inf')],
                                            labels=['1_night', '2-3_nights', '4-7_nights', '8-14_nights', '15+_nights'])
    data['has_special_requests'] = (data['special_requests'] > 0).astype(int)

    # Característica de cliente extranjero (importante para predecir cancelaciones)
    country_col_x = 'country_x'
    country_col_y = 'country_y'
    if country_col_x in data.columns and country_col_y in data.columns:
        data['is_foreign'] = (data[country_col_x].astype(str) != data[country_col_y].astype(str)).astype(int)
        data.loc[data[country_col_x].isna() | data[country_col_y].isna(), 'is_foreign'] = 0

    # Eliminar columnas que podrían causar data leakage
    columns_to_drop = [
        'reservation_status', 'reservation_status_date', 'days_before_arrival',
        'arrival_date', 'booking_date''special_requests', 'stay_nights', 'country_y'
    ]
    columns_to_drop = [col for col in columns_to_drop if col in data.columns]

    # Manejar valores nulos en todo el dataset
    for col in data.select_dtypes(include=['int64', 'float64']).columns:
        if data[col].isna().any():
            data[col] = data[col].fillna(data[col].median())

    for col in data.select_dtypes(include=['object', 'category']).columns:
        if col not in columns_to_drop and data[col].isna().any():
            data[col] = data[col].fillna(data[col].mode()[0])

    X = data.drop(columns=['target'] + columns_to_drop)
    y = data['target']
    groups = data['hotel_id'].copy()

    print(f"Datos cargados: {X.shape[0]} registros, {X.shape[1]} características")
    return X, y, groups


def get_pipeline(X: pd.DataFrame):
    num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    bool_features = X.select_dtypes(include=["bool"]).columns.tolist()
    cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    cat_features = [col for col in cat_features if col not in bool_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
            ("bool", "passthrough", bool_features),
        ]
    )

    classifier = XGBClassifier(
        objective='binary:logistic',
        tree_method='hist',
        eval_metric='logloss',
        use_label_encoder=False,
        verbosity=0,
        scale_pos_weight=len(y_dev[y_dev == 0]) / len(y_dev[y_dev == 1])  # Añadir esta línea
    )

    pipeline = ImbPipeline(steps=[
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42, k_neighbors=3)),
        ("classifier", classifier)
    ])

    return pipeline


def find_optimal_threshold(y_true, y_pred_proba):
    thresholds = np.linspace(0.1, 0.95, 60)
    best_threshold, best_f1 = 0.5, 0

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold


def save_pipeline(pipe):
    model_path = os.environ.get("MODEL_PATH", "models/xgboost_model.pkl")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, mode="wb") as f:
        cloudpickle.dump(pipe, f)
    print(f"Modelo guardado en {model_path}")


class ThresholdClassifier:
    def __init__(self, classifier, threshold=0.5):
        self.classifier = classifier
        self.threshold = threshold

    def predict(self, X):
        return (self.classifier.predict_proba(X)[:, 1] >= self.threshold).astype(int)

    def predict_proba(self, X):
        return self.classifier.predict_proba(X)

if __name__ == "__main__":
    print("=" * 60)
    print("PREDICCIÓN DE CANCELACIONES ANTICIPADAS DE RESERVAS DE HOTEL")
    print("=" * 60)

    X, y, groups = get_X_y()
    print(f"Distribución de clases: Clase 0: {sum(y == 0)}, Clase 1: {sum(y == 1)}")

    X_dev, X_test, y_dev, y_test, groups_dev, groups_test = train_test_split(
        X, y, groups, test_size=0.3, random_state=42, stratify=y
    )

    print(f"\nConjunto de desarrollo: {X_dev.shape[0]} muestras")
    print(f"Conjunto de test (separado): {X_test.shape[0]} muestras")

    if 'hotel_id' in X_dev.columns:
        X_dev = X_dev.drop(columns=['hotel_id'])
        X_test = X_test.drop(columns=['hotel_id'])

    pipe = get_pipeline(X_dev)

    param_dist = {
        'classifier__max_depth': [3, 4, 5],
        'classifier__min_child_weight': [3, 5],
        'classifier__subsample': [0.8, 0.9],
        'classifier__colsample_bytree': [0.7, 0.9],
        'classifier__learning_rate': [0.05],
        'classifier__n_estimators': [50, 100],
        'classifier__lambda': [1, 2, 5],
        'classifier__alpha': [0, 0.1, 0.5]
    }

    group_cv = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=42)
    print("\nDistribución de folds con GroupKFold:")
    for fold_idx, (train_idx, val_idx) in enumerate(group_cv.split(X_dev, y_dev, groups_dev)):
        print(f"Fold {fold_idx + 1}:")
        print(f"  Training: {len(train_idx)} muestras")
        print(f"  Validación: {len(val_idx)} muestras")

    scoring = {'f1': make_scorer(f1_score)}

    print("\nEntrenando modelo con validación cruzada GroupKFold...")
    random_search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=14,
        cv=group_cv.split(X_dev, y_dev, groups_dev),
        scoring=scoring,
        refit='f1',
        random_state=42,
        n_jobs=-1,
        verbose=10
    )

    best_model = random_search.fit(X_dev, y_dev).best_estimator_
    print(f"Mejores parámetros: {random_search.best_params_}")

    final_model = best_model.fit(X_dev, y_dev)

    fold_thresholds = []

    print("\nCalculando umbral óptimo por fold...")
    for fold_idx, (train_idx, val_idx) in enumerate(group_cv.split(X_dev, y_dev, groups_dev)):
        X_train, X_val = X_dev.iloc[train_idx], X_dev.iloc[val_idx]
        y_train, y_val = y_dev.iloc[train_idx], y_dev.iloc[val_idx]

        model_fold = deepcopy(best_model)
        model_fold.fit(X_train, y_train)

        y_val_proba = model_fold.predict_proba(X_val)[:, 1]
        threshold_fold = find_optimal_threshold(y_val, y_val_proba)
        fold_thresholds.append(threshold_fold)

        print(f"Fold {fold_idx + 1} — Threshold óptimo: {threshold_fold:.4f}")

    # Umbral final (puedes elegir el que prefieras)
    threshold = np.mean(fold_thresholds)  # Alternativa más robusta
    print(f"\nUmbral final promedio usado: {threshold:.4f}")


    print("\nEvaluando en conjunto de test no visto...")
    y_test_proba = final_model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba >= threshold).astype(int)

    print("\nMétricas en conjunto de test:")
    print(f"Exactitud:    {accuracy_score(y_test, y_test_pred):.4f}")
    print(f"Precisión:    {precision_score(y_test, y_test_pred):.4f}")
    print(f"Sensibilidad: {recall_score(y_test, y_test_pred):.4f}")
    print(f"F1-Score:     {f1_score(y_test, y_test_pred):.4f}")
    print(f"ROC AUC:      {roc_auc_score(y_test, y_test_proba):.4f}")

    threshold_classifier = ThresholdClassifier(final_model, threshold)
    save_pipeline(threshold_classifier)

    print("\n¡Proceso completado con éxito!")

# Mostrar la pipeline final
#for name, step in pipe.named_steps.items():
# print(f"\nPaso: {name}\n{step}")
