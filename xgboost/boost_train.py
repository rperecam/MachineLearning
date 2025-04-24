import os
import pandas as pd
import numpy as np
import cloudpickle
from sklearn import set_config
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, make_scorer
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import warnings

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
    data['reservation_status'] = data['reservation_status'].replace('No Show', np.nan)
    data = data[data['reservation_status'].notna()].copy()
    data = data[data['reservation_status'] != 'Booked'].copy()

    # Convertir fechas
    for col in ['arrival_date', 'booking_date', 'reservation_status_date']:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col])

    # Target: cancelación con al menos 30 días de anticipación
    data['days_before_arrival'] = (data['arrival_date'] - data['reservation_status_date']).dt.days
    data['target'] = ((data['reservation_status'] == 'Canceled') &
                      (data['days_before_arrival'] >= 30)).astype(int)

    # Crear características clave
    data['lead_time'] = (data['arrival_date'] - data['booking_date']).dt.days
    data['is_high_season'] = data['arrival_date'].dt.month.isin([6, 7, 8, 12]).astype(int)
    data['is_weekend_arrival'] = data['arrival_date'].dt.dayofweek.isin([4, 5]).astype(int)
    data['price_per_night'] = data['rate'] / np.maximum(data['stay_nights'], 1)
    data['stay_duration_category'] = pd.cut(data['stay_nights'],
                                            bins=[-1, 1, 3, 7, 14, float('inf')],
                                            labels=['1_night', '2-3_nights', '4-7_nights', '8-14_nights', '15+_nights'])
    data['has_special_requests'] = (data['special_requests'] > 0).astype(int)

    # Eliminar columnas que podrían causar data leakage
    columns_to_drop = [
        'reservation_status', 'reservation_status_date', 'days_before_arrival',
        'arrival_date', 'booking_date', 'special_requests', 'stay_nights',
    ]
    columns_to_drop = [col for col in columns_to_drop if col in data.columns]

    # Manejar valores nulos en todo el dataset
    # Imputar valores nulos en columnas numéricas
    for col in data.select_dtypes(include=['int64', 'float64']).columns:
        if data[col].isna().any():
            data[col] = data[col].fillna(data[col].median())

    # Imputar valores nulos en columnas categóricas
    for col in data.select_dtypes(include=['object', 'category']).columns:
        if col not in columns_to_drop and data[col].isna().any():
            data[col] = data[col].fillna(data[col].mode()[0])

    # Preparar X e y
    X = data.drop(columns=['target'] + columns_to_drop)
    y = data['target']

    # Preservar hotel_id para GroupKFold
    groups = data['hotel_id'].copy()

    print(f"Datos cargados: {X.shape[0]} registros, {X.shape[1]} características")
    return X, y, groups


from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier

def get_pipeline(X: pd.DataFrame):
    # Identificar tipos de columnas
    num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_features = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # Preprocesamiento
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
        ]
    )

    # Clasificador
    classifier = XGBClassifier(
        objective='binary:logistic',
        tree_method='gpu_hist',
        eval_metric='logloss',
        use_label_encoder=False,
        verbosity=0
    )

    # Pipeline completa con SMOTE
    pipeline = ImbPipeline(steps=[
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("classifier", classifier)
    ])

    return pipeline


def find_optimal_threshold(y_true, y_pred_proba):
    """Encuentra el umbral óptimo para maximizar F1."""
    thresholds = np.linspace(0.3, 0.7, 40)
    best_threshold, best_f1 = 0.5, 0

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold


def balance_dataset(X_train, y_train):
    """Balancea el conjunto de datos con SMOTE."""
    print(f"Distribución original - Clase 0: {sum(y_train == 0)}, Clase 1: {sum(y_train == 1)}")

    smote = SMOTE(sampling_strategy=0.7, random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X_train, y_train)

    print(f"Distribución balanceada - Clase 0: {sum(y_balanced == 0)}, Clase 1: {sum(y_balanced == 1)}")
    return X_balanced, y_balanced


def save_pipeline(pipe):
    """Serializa el modelo entrenado."""
    model_path = os.environ.get("MODEL_PATH", "models/xgboost_model.pkl")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    with open(model_path, mode="wb") as f:
        cloudpickle.dump(pipe, f)
    print(f"Modelo guardado en {model_path}")


class ThresholdClassifier:
    """Wrapper para aplicar un umbral personalizado al clasificador."""

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

    # Cargar datos
    X, y, groups = get_X_y()
    print(f"Distribución de clases: Clase 0: {sum(y == 0)}, Clase 1: {sum(y == 1)}")

    # Crear conjunto de test separado antes de cualquier procesamiento
    X_dev, X_test, y_dev, y_test, groups_dev, groups_test = train_test_split(
        X, y, groups, test_size=0.3, random_state=42, stratify=y
    )

    print(f"\nConjunto de desarrollo: {X_dev.shape[0]} muestras")
    print(f"Conjunto de test (separado): {X_test.shape[0]} muestras")

    # Guardar hotel_id para GroupKFold pero sacarlo de las características
    if 'hotel_id' in X_dev.columns:
        hotel_id_dev = X_dev['hotel_id'].copy()
        hotel_id_test = X_test['hotel_id'].copy()
        X_dev = X_dev.drop(columns=['hotel_id'])
        X_test = X_test.drop(columns=['hotel_id'])

    # Obtener la pipeline
    pipe = get_pipeline(X_dev)

    # Búsqueda de hiperparámetros conservadora
    param_dist = {
        'classifier__max_depth': [3, 4, 5],
        'classifier__min_child_weight': [3, 5],
        'classifier__subsample': [0.8, 0.9],
        'classifier__colsample_bytree': [0.7, 0.9],
        'classifier__learning_rate': [0.05],
        'classifier__n_estimators': [50, 100]
    }

    # Validación cruzada con GroupKFold
    group_cv = GroupKFold(n_splits=5)
    print("\nDistribución de folds con GroupKFold:")
    for fold_idx, (train_idx, val_idx) in enumerate(group_cv.split(X_dev, y_dev, groups_dev)):
        print(f"Fold {fold_idx + 1}:")
        unique_hotels_train = np.unique(groups_dev.iloc[train_idx]).size
        unique_hotels_val = np.unique(groups_dev.iloc[val_idx]).size
        print(f"  Training: {len(train_idx)} muestras, {unique_hotels_train} hoteles únicos")
        print(f"  Validación: {len(val_idx)} muestras, {unique_hotels_val} hoteles únicos")

    # Definir métrica para RandomSearch
    scoring = {'f1': make_scorer(f1_score)}

    # Realizar búsqueda aleatoria con validación cruzada
    print("\nEntrenando modelo con validación cruzada GroupKFold...")
    random_search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=10,
        cv=group_cv.split(X_dev, y_dev, groups_dev),
        scoring=scoring,
        refit='f1',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    best_model = random_search.fit(X_dev, y_dev).best_estimator_
    print(f"Mejores parámetros: {random_search.best_params_}")

    #
    final_model = best_model.fit(X_dev, y_dev)

    # Predicciones en conjunto de desarrollo (para obtener el mejor umbral)
    y_dev_proba = final_model.predict_proba(X_dev)[:, 1]
    threshold = find_optimal_threshold(y_dev, y_dev_proba)
    print(f"\nUmbral óptimo: {threshold:.4f}")

    # Evaluación en conjunto de test
    print("\nEvaluando en conjunto de test no visto...")
    y_test_proba = final_model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba >= threshold).astype(int)

    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_roc_auc = roc_auc_score(y_test, y_test_proba)

    print("\nMétricas en conjunto de test:")
    print(f"Exactitud:    {test_accuracy:.4f}")
    print(f"Precisión:    {test_precision:.4f}")
    print(f"Sensibilidad: {test_recall:.4f}")
    print(f"F1-Score:     {test_f1:.4f}")
    print(f"ROC AUC:      {test_roc_auc:.4f}")

    # Guardar clasificador con umbral personalizado
    threshold_classifier = ThresholdClassifier(final_model, threshold)
    save_pipeline(threshold_classifier)

    print("\n¡Proceso completado con éxito!")