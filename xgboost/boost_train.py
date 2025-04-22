import os
import pandas as pd
import numpy as np
import cloudpickle
from sklearn import set_config
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import warnings

# Configuración
set_config(transform_output="pandas")
warnings.filterwarnings('ignore')


class CustomPipeline:
    """Pipeline personalizada con selección de características y umbral personalizado."""

    def __init__(self, preprocessor, model, best_threshold=0.5):
        self.preprocessor = preprocessor
        self.model = model
        self.best_threshold = best_threshold
        self.is_fitted = hasattr(model, 'classes_')
        self.feature_names = None

    def fit(self, X, y):
        """Entrena la pipeline con los datos proporcionados."""
        X_transformed = self.preprocessor.fit_transform(X)
        self.feature_names = X_transformed.columns
        self.model.fit(X_transformed, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X):
        """Predice probabilidades para los datos de entrada."""
        X_transformed = self.preprocessor.transform(X)
        return self.model.predict_proba(X_transformed)

    def predict(self, X):
        """Predice clases usando el umbral óptimo."""
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.best_threshold).astype(int)


def get_X_y():
    """Carga y preprocesa los datos para el entrenamiento."""
    print("Cargando datos...")

    # Cargar datos
    hotels = pd.read_csv(os.environ.get("HOTELS_DATA_PATH", 'data/hotels.csv'))
    bookings = pd.read_csv(os.environ.get("TRAIN_DATA_PATH", 'data/bookings_train.csv'))

    # Combinar datos
    data = pd.merge(bookings, hotels, on='hotel_id', how='left')
    # Cambio los No Show a Nan y quito los valores de Booked
    data['reservation_status'] = data['reservation_status'].replace('No Show', np.nan)
    data = data[data['reservation_status'] != 'Booked']

    # Convertir fechas
    date_columns = ['arrival_date', 'booking_date', 'reservation_status_date']
    for col in date_columns:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col])

    # Definir target: cancelaciones con al menos 30 días de anticipación
    data['days_before_arrival'] = (data['arrival_date'] - data['reservation_status_date']).dt.days
    data['target'] = ((data['reservation_status'] == 'Canceled') &
                      (data['days_before_arrival'] >= 30)).astype(int)

    # Crear características temporales
    data['lead_time'] = (data['arrival_date'] - data['booking_date']).dt.days
    data['lead_time_category'] = pd.cut(data['lead_time'],
                                        bins=[-1, 7, 30, 90, 180, float('inf')],
                                        labels=['last_minute', 'short', 'medium', 'long', 'very_long'])
    data['is_high_season'] = data['arrival_date'].dt.month.isin([6, 7, 8, 12]).astype(int)
    data['is_weekend_arrival'] = data['arrival_date'].dt.dayofweek.isin([4, 5]).astype(int)
    data['arrival_month'] = data['arrival_date'].dt.month
    data['arrival_dayofweek'] = data['arrival_date'].dt.dayofweek
    data['booking_month'] = data['booking_date'].dt.month

    # Características de precio
    data['price_per_night'] = data['rate'] / np.maximum(data['stay_nights'], 1)
    data['price_per_person'] = data['rate'] / np.maximum(data['total_guests'], 1)

    # Limitar valores extremos
    for col in ['price_per_night', 'price_per_person']:
        cap = np.percentile(data[col].dropna(), 90)
        data[col] = data[col].clip(upper=cap)

    # Características de estancia
    data['stay_duration_category'] = pd.cut(data['stay_nights'],
                                            bins=[-1, 1, 3, 7, 14, float('inf')],
                                            labels=['1_night', '2-3_nights', '4-7_nights', '8-14_nights', '15+_nights'])

    # Características de solicitudes
    data['has_special_requests'] = (data['special_requests'] > 0).astype(int)
    data['special_requests_ratio'] = data['special_requests'] / np.maximum(data['total_guests'], 1)

    # Características de ubicación
    if 'country_x' in data.columns and 'country_y' in data.columns:
        data['is_foreign'] = (data['country_x'] != data['country_y']).astype(int)
        data.loc[data['country_x'].isna() | data['country_y'].isna(), 'is_foreign'] = 0

    # Características interactivas
    data['guest_nights'] = data['total_guests'] * data['stay_nights']
    data['booking_lead_ratio'] = data['lead_time'] / np.maximum(data['stay_nights'], 1)
    data['requested_parking'] = (data['required_car_parking_spaces'] > 0).astype(int)

    # Eliminar columnas no necesarias
    columns_to_drop = [
        'reservation_status', 'reservation_status_date', 'booking_date',
        'arrival_date', 'days_before_arrival', 'country_x', 'country_y', 'special_request'
    ]
    columns_to_drop = [col for col in columns_to_drop if col in data.columns]
    data.drop(columns=columns_to_drop, inplace=True)

    # Manejar valores infinitos y nulos
    for col in ['price_per_night', 'price_per_person', 'special_requests_ratio', 'booking_lead_ratio']:
        if col in data.columns:
            data[col] = data[col].replace([np.inf, -np.inf], np.nan)

    # Preparar X e y
    X = data.drop(columns=['target'])
    y = data['target']

    print(f"Datos cargados: {X.shape[0]} registros, {X.shape[1]} características")
    return X, y


def get_preprocessor(X):
    """Define el preprocesador de datos."""
    # Identificar características categóricas y numéricas
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

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

    # Combinar transformadores
    return ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )


def balance_dataset(X, y, method='combined'):
    """Balancea el conjunto de datos."""
    class_counts = y.value_counts()
    print(f"Distribución original - Clase 0: {class_counts[0]}, Clase 1: {class_counts.get(1, 0)}")

    if method == 'under':
        # Solo submuestreo
        rus = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
        X_balanced, y_balanced = rus.fit_resample(X, y)
    elif method == 'over':
        # Solo sobremuestreo
        smote = SMOTE(sampling_strategy=0.7, random_state=42, k_neighbors=5)
        X_balanced, y_balanced = smote.fit_resample(X, y)
    else:
        # Combinación de técnicas
        rus = RandomUnderSampler(sampling_strategy=0.3, random_state=42)
        X_under, y_under = rus.fit_resample(X, y)
        smote = SMOTE(sampling_strategy=0.65, random_state=42, k_neighbors=5)
        X_balanced, y_balanced = smote.fit_resample(X_under, y_under)

    class_counts = pd.Series(y_balanced).value_counts()
    print(f"Distribución balanceada - Clase 0: {class_counts[0]}, Clase 1: {class_counts.get(1, 0)}")
    return X_balanced, y_balanced


def find_optimal_threshold(y_true, y_pred_proba):
    """Encuentra el umbral óptimo para maximizar F1."""
    thresholds = np.linspace(0.1, 0.9, 100)
    best_threshold, best_f1 = 0.5, 0

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1, best_threshold = f1, threshold

    return best_threshold, best_f1


def evaluate_model(model, X, y, threshold=0.5):
    """Evalúa el modelo usando varias métricas."""
    try:
        y_pred_proba = model.predict_proba(X)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)

        metrics = {
            'Exactitud': accuracy_score(y, y_pred),
            'Puntuación F1': f1_score(y, y_pred),
            'Precisión': precision_score(y, y_pred),
            'Recuperación': recall_score(y, y_pred),
            'AUC ROC': roc_auc_score(y, y_pred_proba)
        }
    except Exception as e:
        print(f"Error al evaluar el modelo: {e}")
        metrics = {metric: 0.0 for metric in ['Exactitud', 'Puntuación F1', 'Precisión', 'Recuperación', 'AUC ROC']}

    return metrics


def train_with_cv(X, y, params, cv_splits=5):
    """Entrena un modelo utilizando validación cruzada."""
    print(f"Entrenando con validación cruzada ({cv_splits} pliegues)...")

    # Asegurar que existe hotel_id para la agrupación
    if 'hotel_id' not in X.columns:
        print("Advertencia: hotel_id no encontrado. Utilizando CV estándar.")
        X = X.copy()
        X['hotel_id'] = np.arange(len(X))

    # Preparar validación cruzada
    cv = StratifiedGroupKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    groups = X['hotel_id'].values

    # Listas para almacenar resultados
    fold_metrics = []
    fold_models = []
    fold_thresholds = []

    # Realizar validación cruzada
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
        print(f"Fold {fold + 1}/{cv_splits}", end="")

        # Dividir datos para este fold
        X_train_fold = X.iloc[train_idx].copy()
        y_train_fold = y.iloc[train_idx].copy()
        X_val_fold = X.iloc[val_idx].copy()
        y_val_fold = y.iloc[val_idx].copy()

        # Eliminar hotel_id para el entrenamiento
        if 'hotel_id' in X_train_fold.columns:
            X_train_fold_no_hotel = X_train_fold.drop(columns=['hotel_id'])
            X_val_fold_no_hotel = X_val_fold.drop(columns=['hotel_id'])
        else:
            X_train_fold_no_hotel = X_train_fold
            X_val_fold_no_hotel = X_val_fold

        # Preprocesar y balancear datos
        preprocessor = get_preprocessor(X_train_fold_no_hotel)
        X_train_processed = preprocessor.fit_transform(X_train_fold_no_hotel)
        X_train_balanced, y_train_balanced = balance_dataset(X_train_processed, y_train_fold)

        # Entrenar modelo
        model = xgb.XGBClassifier(**params)
        model.fit(X_train_balanced, y_train_balanced)

        # Procesar datos de validación
        X_val_processed = preprocessor.transform(X_val_fold_no_hotel)

        # Encontrar umbral óptimo
        y_val_proba = model.predict_proba(X_val_processed)[:, 1]
        best_threshold, _ = find_optimal_threshold(y_val_fold, y_val_proba)

        # Evaluar en datos de validación
        temp_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        y_val_pred = (temp_pipeline.predict_proba(X_val_fold_no_hotel)[:, 1] >= best_threshold).astype(int)
        metrics = {
            'Exactitud': accuracy_score(y_val_fold, y_val_pred),
            'Puntuación F1': f1_score(y_val_fold, y_val_pred),
            'Precisión': precision_score(y_val_fold, y_val_pred),
            'Recuperación': recall_score(y_val_fold, y_val_pred),
            'AUC ROC': roc_auc_score(y_val_fold, temp_pipeline.predict_proba(X_val_fold_no_hotel)[:, 1])
        }

        # Guardar resultados
        fold_metrics.append(metrics)
        fold_models.append((preprocessor, model))
        fold_thresholds.append(best_threshold)

        print(f" - F1: {metrics['Puntuación F1']:.4f}, Umbral: {best_threshold:.3f}")

    # Calcular métricas promedio
    print("\nMétricas promedio de validación cruzada:")
    avg_metrics = {}
    for metric in fold_metrics[0].keys():
        avg_metrics[metric] = np.mean([fold[metric] for fold in fold_metrics])
        print(f"{metric}: {avg_metrics[metric]:.4f}")

    # Seleccionar mejor modelo según F1-Score
    best_fold_idx = np.argmax([fold['Puntuación F1'] for fold in fold_metrics])
    best_preprocessor, best_model = fold_models[best_fold_idx]
    best_threshold = fold_thresholds[best_fold_idx]

    print(f"Mejor modelo: fold {best_fold_idx + 1}, F1: {fold_metrics[best_fold_idx]['Puntuación F1']:.4f}")
    return best_preprocessor, best_model, best_threshold


def get_pipeline():
    """Define y entrena el modelo final utilizando validación cruzada."""
    # Parámetros optimizados para XGBoost
    xgb_params = {
        'max_depth': 8,
        'min_child_weight': 2,
        'gamma': 0.4,
        'subsample': 0.8,
        'colsample_bytree': 0.5,
        'reg_alpha': 0.7,
        'reg_lambda': 3.6,
        'learning_rate': 0.05,
        'n_estimators': 500,
        'scale_pos_weight': 1.0,
        'objective': 'binary:logistic',
        'tree_method': 'hist',
        'random_state': 42,
        'n_jobs': -1
    }

    # Cargar datos
    X, y = get_X_y()

    # Entrenar con validación cruzada para obtener el mejor modelo
    print("Entrenando modelo con validación cruzada...")
    best_preprocessor, best_model, best_threshold = train_with_cv(X, y, xgb_params, cv_splits=5)

    # Crear pipeline personalizada con el mejor modelo
    pipe = CustomPipeline(best_preprocessor, best_model, best_threshold)

    # Dividir para evaluación final (solo para métrica de test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Evaluar en conjunto de prueba separado
    print("\nEvaluando en conjunto de prueba separado...")
    # Eliminar hotel_id si existe
    X_test_no_hotel = X_test.drop(columns=['hotel_id']) if 'hotel_id' in X_test.columns else X_test

    y_test_proba = pipe.predict_proba(X_test_no_hotel)[:, 1]
    y_test_pred = (y_test_proba >= best_threshold).astype(int)

    test_metrics = {
        'Exactitud': accuracy_score(y_test, y_test_pred),
        'Puntuación F1': f1_score(y_test, y_test_pred),
        'Precisión': precision_score(y_test, y_test_pred),
        'Recuperación': recall_score(y_test, y_test_pred),
        'AUC ROC': roc_auc_score(y_test, y_test_proba)
    }

    print("\nMétricas en conjunto de prueba final:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")

    # Reentrenar con todos los datos para el modelo final
    print("\nEntrenando modelo final con todos los datos...")
    X_no_hotel = X.drop(columns=['hotel_id']) if 'hotel_id' in X.columns else X

    # Preprocesar y balancear todos los datos para el modelo final
    X_processed = best_preprocessor.fit_transform(X_no_hotel)
    X_balanced, y_balanced = balance_dataset(X_processed, y)

    # Entrenar modelo final con todos los datos
    final_model = xgb.XGBClassifier(**xgb_params)
    final_model.fit(X_balanced, y_balanced)

    # Actualizar el modelo en la pipeline
    pipe = CustomPipeline(best_preprocessor, final_model, best_threshold)

    return pipe


def save_pipeline(pipe, filename="models/xgboost_model.pkl"):
    """Serializa el modelo entrenado."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    model_path = os.environ.get("MODEL_PATH", filename)

    with open(model_path, mode="wb") as f:
        cloudpickle.dump(pipe, f)
    print(f"Modelo guardado en {model_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("PREDICCIÓN DE CANCELACIONES ANTICIPADAS DE RESERVAS DE HOTEL")
    print("=" * 60)

    # Cargar datos
    X, y = get_X_y()
    print(f"Distribución de clases: {y.value_counts().to_dict()}")

    # Crear y entrenar pipeline con validación cruzada
    print("Entrenando modelo final...")
    pipe = get_pipeline()

    # Guardar pipeline entrenada
    save_pipeline(pipe)

    print("\n¡Proceso completado con éxito!")