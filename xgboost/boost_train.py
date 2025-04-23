import os
import pandas as pd
import numpy as np
import cloudpickle
from sklearn import set_config
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, make_scorer
import xgboost as xgb
from imblearn.over_sampling import ADASYN
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
    data = data[data['reservation_status'] != 'Booked'].copy()

    # Convertir fechas
    for col in ['arrival_date', 'booking_date', 'reservation_status_date']:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col])

    # Definir target: cancelación con al menos 30 días de anticipación
    data['days_before_arrival'] = (data['arrival_date'] - data['reservation_status_date']).dt.days
    data['target'] = ((data['reservation_status'] == 'Canceled') &
                      (data['days_before_arrival'] >= 30)).astype(int)

    # Crear características clave
    # Tiempo de anticipación
    data['lead_time'] = (data['arrival_date'] - data['booking_date']).dt.days

    # Temporada alta
    data['is_high_season'] = data['arrival_date'].dt.month.isin([6, 7, 8, 12]).astype(int)

    # Fin de semana
    data['is_weekend_arrival'] = data['arrival_date'].dt.dayofweek.isin([4, 5]).astype(int)

    # Precio por noche/persona
    data['price_per_night'] = data['rate'] / np.maximum(data['stay_nights'], 1)

    # Duración de estancia categorizada
    data['stay_duration_category'] = pd.cut(data['stay_nights'],
                                            bins=[-1, 1, 3, 7, 14, float('inf')],
                                            labels=['1_night', '2-3_nights', '4-7_nights', '8-14_nights', '15+_nights'])

    # Solicitudes especiales
    data['has_special_requests'] = (data['special_requests'] > 0).astype(int)

    # Eliminar columnas que podrían causar data leakage
    columns_to_drop = [
        'reservation_status', 'reservation_status_date', 'days_before_arrival',
        'arrival_date', 'booking_date','special_requests','stay_nights',
    ]
    columns_to_drop = [col for col in columns_to_drop if col in data.columns]

    # Preparar X e y
    X = data.drop(columns=['target'] + columns_to_drop)
    y = data['target']

    # Preservar hotel_id para GroupKFold
    groups = data['hotel_id'].copy()

    print(f"Datos cargados: {X.shape[0]} registros, {X.shape[1]} características")
    return X, y, groups


def get_pipeline(X):
    """Define y retorna la pipeline para entrenar el modelo."""
    # Identificar tipos de características
    # Seleccionar características categóricas y numéricas por tipo de dato
    categorical_features = list(X.select_dtypes(include=['object', 'category', 'bool']).columns)
    numerical_features = list(X.select_dtypes(include=['int64', 'float64']).columns)

    # Excluir hotel_id de las características para el modelo
    if 'hotel_id' in categorical_features:
        categorical_features.remove('hotel_id')

    # Pipeline para características categóricas
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Pipeline para características numéricas
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Preprocesador
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

    # Modelo XGBoost
    xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')

    # Pipeline final
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', xgb_model)
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


def balance_dataset(X, y):
    """Balancea el conjunto de datos con ADASYN."""
    print(f"Distribución original - Clase 0: {sum(y == 0)}, Clase 1: {sum(y == 1)}")

    # Verificar si hay suficientes ejemplos para aplicar ADASYN
    if sum(y == 1) >= 5:  # ADASYN necesita al menos 5 ejemplos
        adasyn = ADASYN(sampling_strategy=0.7, random_state=42)
        X_balanced, y_balanced = adasyn.fit_resample(X, y)
        print(f"Distribución balanceada - Clase 0: {sum(y_balanced == 0)}, Clase 1: {sum(y_balanced == 1)}")
        return X_balanced, y_balanced
    else:
        print("No se pudo aplicar ADASYN, usando datos originales")
        return X, y


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

    # Eliminar hotel_id de las características si existe (ya lo usamos como grupo)
    if 'hotel_id' in X.columns:
        X = X.drop(columns=['hotel_id'])

    # Obtener la pipeline básica
    pipe = get_pipeline(X)

    # Definir hiperparámetros para RandomSearch
    param_dist = {
        'classifier__max_depth': [6, 8, 10],
        'classifier__min_child_weight': [1, 2, 3],
        'classifier__gamma': [0, 0.1, 0.2],
        'classifier__subsample': [0.7, 0.8, 0.9],
        'classifier__colsample_bytree': [0.5, 0.6, 0.7],
        'classifier__reg_alpha': [0, 0.1, 0.5],
        'classifier__reg_lambda': [1, 2, 3],
        'classifier__learning_rate': [0.01, 0.05, 0.1],
        'classifier__n_estimators': [100, 200, 300]
    }

    # Definir validación cruzada con GroupKFold
    group_cv = GroupKFold(n_splits=5)

    # Crear splits para visualización y análisis
    print("\nDistribución de folds con GroupKFold:")
    for fold_idx, (train_idx, test_idx) in enumerate(group_cv.split(X, y, groups)):
        print(f"Fold {fold_idx + 1}:")
        unique_hotels_train = groups.iloc[train_idx].nunique()
        unique_hotels_test = groups.iloc[test_idx].nunique()
        print(f"  Training: {len(train_idx)} muestras, {unique_hotels_train} hoteles únicos")
        print(f"  Testing:  {len(test_idx)} muestras, {unique_hotels_test} hoteles únicos")

    # Definir métrica para RandomSearch
    scoring = {'f1': make_scorer(f1_score)}

    # Realizar RandomSearch con GroupKFold
    random_search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=30,
        cv=group_cv.split(X, y, groups),
        scoring=scoring,
        refit='f1',
        random_state=42,
        n_jobs=-1
    )

    # Entrenar el modelo
    print("\nEntrenando modelo con validación cruzada GroupKFold...")
    best_model = random_search.fit(X, y, groups=groups).best_estimator_
    print(f"Mejores parámetros: {random_search.best_params_}")

    # Crear conjunto de validación final para evaluar el rendimiento
    # Usamos GroupKFold para crear un split final de validación
    final_cv = list(GroupKFold(n_splits=4).split(X, y, groups))[-1]  # Tomamos el último fold
    train_idx, val_idx = final_cv
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Ajustar el modelo final en este conjunto de entrenamiento
    final_model = best_model.fit(X_train, y_train)

    # Realizamos predicciones sobre el conjunto de validación
    y_val_proba = final_model.predict_proba(X_val)[:, 1]

    # Encontrar umbral óptimo
    threshold = find_optimal_threshold(y_val, y_val_proba)
    print(f"\nUmbral óptimo: {threshold:.4f}")

    # Realizamos predicciones con el umbral óptimo
    y_val_pred = (y_val_proba >= threshold).astype(int)

    # Calcular métricas
    accuracy = accuracy_score(y_val, y_val_pred)
    precision = precision_score(y_val, y_val_pred)
    recall = recall_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred)
    roc_auc = roc_auc_score(y_val, y_val_proba)

    # Imprimir métricas
    print("\nMétricas en conjunto de validación:")
    print(f"Exactitud:    {accuracy:.4f}")
    print(f"Precisión:    {precision:.4f}")
    print(f"Sensibilidad: {recall:.4f}")
    print(f"F1-Score:     {f1:.4f}")
    print(f"ROC AUC:      {roc_auc:.4f}")

    # Crear y entrenar el modelo final con todos los datos
    print("\nEntrenando modelo final con todos los datos...")
    final_model = best_model.fit(X, y)

    # Creamos un clasificador con umbral personalizado
    threshold_classifier = ThresholdClassifier(final_model, threshold)

    # Guardar el modelo con umbral personalizado
    save_pipeline(threshold_classifier)

    print("\n¡Proceso completado con éxito!")