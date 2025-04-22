import os
import pandas as pd
import numpy as np
import cloudpickle
from sklearn import set_config
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import warnings

# Configuración para que sklearn devuelva DataFrames
set_config(transform_output="pandas")

# Ignorar advertencias para una salida más limpia
warnings.filterwarnings('ignore')

class CustomPipeline:
    """
    Pipeline personalizada que permite la selección de características y
    umbral de clasificación personalizado.
    """
    def __init__(self, preprocessor, model, best_threshold=0.5):
        self.preprocessor = preprocessor
        self.model = model
        self.best_threshold = best_threshold
        self.is_fitted = False
        self.feature_names = None

    def fit(self, X, y):
        print("Preprocesando datos...")
        X_transformed = self.preprocessor.fit_transform(X)
        self.feature_names = X_transformed.columns

        print("Entrenando modelo...")
        self.model.fit(X_transformed, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("El modelo no ha sido entrenado")
        X_transformed = self.preprocessor.transform(X)
        return self.model.predict_proba(X_transformed)

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.best_threshold).astype(int)

def check_feature_correlation(X, threshold=0.8):
    """
    Identifica características altamente correlacionadas.

    Args:
        X: DataFrame con características
        threshold: Umbral de correlación para considerar dos características como correlacionadas

    Returns:
        lista de tuplas de características correlacionadas
    """
    print("Identificando características altamente correlacionadas...")
    # Seleccionar solo características numéricas
    numeric_X = X.select_dtypes(include=['int64', 'float64'])

    # Calcular matriz de correlación
    corr_matrix = numeric_X.corr(method='spearman').abs()

    # Encontrar pares de características altamente correlacionadas
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    highly_correlated = [(corr_matrix.index[i], corr_matrix.columns[j], upper_tri.iloc[i, j])
                         for i, j in zip(*np.where(upper_tri > threshold))]

    return highly_correlated

def check_target_correlation(X, y, threshold=0.2):
    """
    Analiza la correlación entre cada característica y el target.

    Args:
        X: DataFrame con características
        y: Serie con variable objetivo
        threshold: Umbral para considerar correlación significativa

    Returns:
        Serie con correlaciones ordenadas
    """
    print("Analizando correlación entre características y target...")
    # Preparar datos
    data = X.copy()
    data['target'] = y

    # Seleccionar solo características numéricas
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
    numeric_cols = [col for col in numeric_cols if col != 'target']

    # Calcular correlación con target
    correlations = {}
    for col in numeric_cols:
        from scipy.stats import spearmanr
        corr, _ = spearmanr(data[col], data['target'])
        correlations[col] = abs(corr)

    # Ordenar correlaciones
    corr_series = pd.Series(correlations).sort_values(ascending=False)

    return corr_series

def get_X_y():
    """
    Carga y preprocesa los datos para el entrenamiento, evitando data leakage.

    Returns:
        X: DataFrame con características
        y: Serie con variable objetivo
    """
    print("Cargando datos...")
    # Cargar datos
    hotels = pd.read_csv(os.environ.get("HOTELS_DATA_PATH", 'data/hotels.csv'))
    bookings = pd.read_csv(os.environ.get("TRAIN_DATA_PATH", 'data/bookings_train.csv'))

    # Combinar datos
    merged = pd.merge(bookings, hotels, on='hotel_id', how='left')

    # CORRECCIÓN: No filtrar por reservation_status para evitar pérdida de datos
    data = merged.copy()

    # Convertir columnas de fecha a datetime
    date_columns = ['arrival_date', 'booking_date', 'reservation_status_date']
    for col in date_columns:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col])

    # CORRECCIÓN: Definir target correctamente
    # Cancelaciones con al menos 30 días de anticipación
    data['days_before_arrival'] = (data['arrival_date'] - data['reservation_status_date']).dt.days
    data['target'] = ((data['reservation_status'] == 'Canceled') &
                      (data['days_before_arrival'] >= 30)).astype(int)

    # Extraer características temporales (evitando data leakage)
    data['lead_time'] = (data['arrival_date'] - data['booking_date']).dt.days
    data['lead_time_category'] = pd.cut(data['lead_time'],
                                        bins=[-1, 7, 30, 90, 180, float('inf')],
                                        labels=['last_minute', 'short', 'medium', 'long', 'very_long'])
    data['is_high_season'] = data['arrival_date'].dt.month.isin([6, 7, 8, 12]).astype(int)
    data['is_weekend_arrival'] = data['arrival_date'].dt.dayofweek.isin([4, 5]).astype(int)
    data['arrival_month'] = data['arrival_date'].dt.month
    data['arrival_dayofweek'] = data['arrival_date'].dt.dayofweek
    data['booking_month'] = data['booking_date'].dt.month

    # CORRECCIÓN: Características derivadas de precio con comprobación de valores razonables
    data['price_per_night'] = data['rate'] / np.maximum(data['stay_nights'], 1)
    data['price_per_person'] = data['rate'] / np.maximum(data['total_guests'], 1)

    # CORRECCIÓN: Limitar valores extremos en ratios
    price_cap = np.percentile(data['price_per_night'].dropna(), 99)
    data['price_per_night'] = data['price_per_night'].clip(upper=price_cap)
    price_person_cap = np.percentile(data['price_per_person'].dropna(), 99)
    data['price_per_person'] = data['price_per_person'].clip(upper=price_person_cap)

    # Extraer características de duración de estancia
    data['stay_duration_category'] = pd.cut(data['stay_nights'],
                                            bins=[-1, 1, 3, 7, 14, float('inf')],
                                            labels=['1_night', '2-3_nights', '4-7_nights', '8-14_nights', '15+_nights'])

    # Extraer características de solicitudes especiales
    data['has_special_requests'] = (data['special_requests'] > 0).astype(int)
    data['special_requests_ratio'] = data['special_requests'] / np.maximum(data['total_guests'], 1)

    # Extraer características de ubicación
    if 'country_x' in data.columns and 'country_y' in data.columns:
        data['is_foreign'] = (data['country_x'] != data['country_y']).astype(int)
        data.loc[data['country_x'].isna() | data['country_y'].isna(), 'is_foreign'] = 0

    # CORRECCIÓN: Reemplazar características interactivas que podrían causar fugas
    # Evitamos mezclar características de precio con características de estancia
    # y usamos solo características disponibles en el momento de la reserva
    data['guest_nights'] = data['total_guests'] * data['stay_nights']
    data['booking_lead_ratio'] = data['lead_time'] / np.maximum(data['stay_nights'], 1)

    # Extraer características de transporte
    data['requested_parking'] = (data['required_car_parking_spaces'] > 0).astype(int)

    # CORRECCIÓN: Eliminar columnas que pueden causar filtración de datos
    columns_to_drop = [
        'reservation_status', 'reservation_status_date', 'booking_date',
        'arrival_date', 'days_before_arrival', 'country_x', 'country_y', 'special_request'
    ]
    columns_to_drop = [col for col in columns_to_drop if col in data.columns]
    data.drop(columns=columns_to_drop, inplace=True)

    # Manejar valores infinitos y nulos en características numéricas
    for col in ['price_per_night', 'price_per_person', 'special_requests_ratio', 'booking_lead_ratio']:
        if col in data.columns:
            data[col] = data[col].replace([np.inf, -np.inf], np.nan)

    # Preparar X e y
    X = data.drop(columns=['target'])
    y = data['target']

    print("Datos cargados y preprocesados.")
    return X, y

def get_preprocessor(X):
    """
    Define el preprocesador de datos.

    Args:
        X: DataFrame de ejemplo para identificar columnas

    Returns:
        Objeto ColumnTransformer para preprocesamiento
    """
    print("Definiendo preprocesador de datos...")
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
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

    print("Preprocesador definido.")
    return preprocessor

def balance_dataset(X, y, method='combined'):
    """
    Balancea el conjunto de datos aplicando técnicas de submuestreo y sobremuestreo.

    Args:
        X: DataFrame con características
        y: Serie con variable objetivo
        method: Método de balanceo ('under', 'over', 'combined')

    Returns:
        X_balanced: DataFrame balanceado
        y_balanced: Serie balanceada
    """
    print("Balanceando el conjunto de datos...")
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
        # Primero submuestra para reducir la clase mayoritaria
        rus = RandomUnderSampler(sampling_strategy=0.3, random_state=42)
        X_under, y_under = rus.fit_resample(X, y)
        # Luego sobremuestrea para aumentar la clase minoritaria
        smote = SMOTE(sampling_strategy=0.65, random_state=42, k_neighbors=5)
        X_balanced, y_balanced = smote.fit_resample(X_under, y_under)

    print("Conjunto de datos balanceado.")
    return X_balanced, y_balanced

def find_optimal_threshold(y_true, y_pred_proba):
    """
    Encuentra el umbral óptimo para maximizar F1.

    Args:
        y_true: Valores reales
        y_pred_proba: Probabilidades predichas

    Returns:
        best_threshold: Umbral óptimo
        best_f1: Mejor valor F1
    """
    print("Encontrando umbral óptimo...")
    thresholds = np.linspace(0.1, 0.9, 100)
    best_threshold, best_f1 = 0.5, 0

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1, best_threshold = f1, threshold

    print(f"Umbral óptimo encontrado: {best_threshold} con F1: {best_f1}")
    return best_threshold, best_f1

def evaluate_model(model, X, y, threshold=0.5):
    """
    Evalúa el modelo usando varias métricas.

    Args:
        model: Modelo entrenado
        X: Características
        y: Target
        threshold: Umbral de clasificación

    Returns:
        dict: Métricas de evaluación
    """
    print("Evaluando el modelo...")
    # Predicciones
    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Calcular métricas
    metrics = {
        'Exactitud': accuracy_score(y, y_pred),
        'Puntuación F1': f1_score(y, y_pred),
        'Precisión': precision_score(y, y_pred),
        'Recuperación': recall_score(y, y_pred),
        'AUC ROC': roc_auc_score(y, y_pred_proba)
    }

    print("Métricas de evaluación calculadas.")
    return metrics

def save_model(pipe, filename="models/xgboost_model.pkl"):
    """
    Serializa el modelo entrenado.

    Args:
        pipe: Pipeline entrenada
        filename: Nombre del archivo para guardar
    """
    print(f"Guardando modelo en {filename}...")
    with open(os.environ.get("MODEL_PATH", filename), mode="wb") as f:
        cloudpickle.dump(pipe, f)
    print("Modelo guardado.")

def train_final_model(X, y):
    """
    Entrena el modelo final utilizando todo el conjunto de datos y las mejores prácticas.

    Args:
        X: Características
        y: Target

    Returns:
        pipe: Pipeline entrenada final
    """
    print("Entrenando modelo final...")
    # Dividir datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Definir preprocesador
    preprocessor = get_preprocessor(X_train)

    # Aplicar preprocesamiento
    X_train_processed = preprocessor.fit_transform(X_train)

    # Balancear datos de entrenamiento
    X_train_balanced, y_train_balanced = balance_dataset(X_train_processed, y_train)

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

    # Crear y entrenar modelo
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X_train_balanced, y_train_balanced)

    # Aplicar preprocesamiento a datos de prueba
    X_test_processed = preprocessor.transform(X_test)

    # Evaluar en conjunto de prueba
    y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
    best_threshold, _ = find_optimal_threshold(y_test, y_pred_proba)

    # Crear pipeline final
    final_pipe = CustomPipeline(preprocessor, model, best_threshold)

    # Importante: Ajustar el pipeline completo para establecer is_fitted=True
    final_pipe.fit(X_train, y_train)

    print("Modelo final entrenado.")
    return final_pipe

if __name__ == "__main__":
    # Cargar y preparar datos
    X, y = get_X_y()

    # Entrenar modelo final con todo el conjunto de datos
    final_pipe = train_final_model(X, y)

    # Guardar modelo final
    save_model(final_pipe)
