"""
Modelo de predicción de cancelaciones anticipadas de reservas de hotel

Este script entrena un modelo de machine learning para predecir qué reservas de hotel
tienen alta probabilidad de ser canceladas con al menos 30 días de anticipación.

Autor: [Tu nombre]
Fecha: 22/04/2025
"""

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
from scipy.stats import spearmanr

# Configuración para que sklearn devuelva DataFrames
set_config(transform_output="pandas")

# Ignorar advertencias para una salida más limpia
warnings.filterwarnings('ignore')


#######################
# CLASES PRINCIPALES  #
#######################

class CustomPipeline:
    """
    Pipeline personalizada que permite la selección de características y
    umbral de clasificación personalizado.

    Attributes:
        preprocessor: Transformador para preprocesar los datos
        model: Modelo de clasificación
        best_threshold: Umbral óptimo para clasificación binaria
        is_fitted: Indica si el modelo ha sido entrenado
        feature_names: Nombres de las características después del preprocesamiento
    """

    def __init__(self, preprocessor, model, best_threshold=0.5):
        self.preprocessor = preprocessor
        self.model = model
        self.best_threshold = best_threshold
        # Inicializar is_fitted dependiendo del estado del modelo
        self.is_fitted = hasattr(model, 'classes_')
        self.feature_names = None

    def fit(self, X, y):
        """
        Entrenar la pipeline con los datos proporcionados.

        Args:
            X: DataFrame con características
            y: Serie con variable objetivo

        Returns:
            self: Pipeline entrenada
        """
        X_transformed = self.preprocessor.fit_transform(X)
        self.feature_names = X_transformed.columns
        self.model.fit(X_transformed, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X):
        """
        Predecir probabilidades para los datos de entrada.

        Args:
            X: DataFrame con características

        Returns:
            array: Probabilidades predichas
        """
        X_transformed = self.preprocessor.transform(X)
        return self.model.predict_proba(X_transformed)

    def predict(self, X):
        """
        Predecir clases usando el umbral óptimo.

        Args:
            X: DataFrame con características

        Returns:
            array: Predicciones binarias
        """
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.best_threshold).astype(int)


#######################
# FUNCIONES DE DATOS  #
#######################

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
    #Quitos los valores de Booked en reservation_status
    merged['reservation_status'] = merged['reservation_status'].replace({'Booked': 'No Show'})
    data = merged.copy()

    # Convertir columnas de fecha a datetime
    date_columns = ['arrival_date', 'booking_date', 'reservation_status_date']
    for col in date_columns:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col])

    # Definir target: cancelaciones con al menos 30 días de anticipación
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

    # Características derivadas de precio con comprobación de valores razonables
    data['price_per_night'] = data['rate'] / np.maximum(data['stay_nights'], 1)
    data['price_per_person'] = data['rate'] / np.maximum(data['total_guests'], 1)

    # Limitar valores extremos en ratios
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

    # Características interactivas que evitan data leakage
    data['guest_nights'] = data['total_guests'] * data['stay_nights']
    data['booking_lead_ratio'] = data['lead_time'] / np.maximum(data['stay_nights'], 1)

    # Extraer características de transporte
    data['requested_parking'] = (data['required_car_parking_spaces'] > 0).astype(int)

    # Eliminar columnas no necesarias
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

    print(f"Datos cargados: {X.shape[0]} registros, {X.shape[1]} características")
    return X, y


##########################
# FUNCIONES DE ANÁLISIS  #
##########################

def check_feature_correlation(X, threshold=0.8):
    """
    Identifica características altamente correlacionadas.

    Args:
        X: DataFrame con características
        threshold: Umbral de correlación para considerar dos características como correlacionadas

    Returns:
        lista de tuplas de características correlacionadas
    """
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
    # Preparar datos
    data = X.copy()
    data['target'] = y

    # Seleccionar solo características numéricas
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
    numeric_cols = [col for col in numeric_cols if col != 'target']

    # Calcular correlación con target
    correlations = {}
    for col in numeric_cols:
        corr, _ = spearmanr(data[col], data['target'])
        correlations[col] = abs(corr)

    # Ordenar correlaciones
    corr_series = pd.Series(correlations).sort_values(ascending=False)

    return corr_series


################################
# FUNCIONES DE PREPROCESAMIENTO #
################################

def get_preprocessor(X):
    """
    Define el preprocesador de datos.

    Args:
        X: DataFrame de ejemplo para identificar columnas

    Returns:
        Objeto ColumnTransformer para preprocesamiento
    """
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
        # Primero submuestra para reducir la clase mayoritaria
        rus = RandomUnderSampler(sampling_strategy=0.3, random_state=42)
        X_under, y_under = rus.fit_resample(X, y)
        # Luego sobremuestrea para aumentar la clase minoritaria
        smote = SMOTE(sampling_strategy=0.65, random_state=42, k_neighbors=5)
        X_balanced, y_balanced = smote.fit_resample(X_under, y_under)

    class_counts = pd.Series(y_balanced).value_counts()
    print(f"Distribución balanceada - Clase 0: {class_counts[0]}, Clase 1: {class_counts.get(1, 0)}")
    return X_balanced, y_balanced


###########################
# FUNCIONES DE EVALUACIÓN #
###########################

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
    thresholds = np.linspace(0.1, 0.9, 100)
    best_threshold, best_f1 = 0.5, 0

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1, best_threshold = f1, threshold

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
    # Predicciones
    try:
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
    except Exception as e:
        print(f"Error al evaluar el modelo: {e}")
        # Devolver métricas vacías en caso de error
        metrics = {
            'Exactitud': 0.0,
            'Puntuación F1': 0.0,
            'Precisión': 0.0,
            'Recuperación': 0.0,
            'AUC ROC': 0.0
        }

    return metrics


def save_model(pipe, filename="models/xgboost_model.pkl"):
    """
    Serializa el modelo entrenado.

    Args:
        pipe: Pipeline entrenada
        filename: Nombre del archivo para guardar
    """
    # Asegurarse de que el directorio existe
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    model_path = os.environ.get("MODEL_PATH", filename)
    with open(model_path, mode="wb") as f:
        cloudpickle.dump(pipe, f)
    print(f"Modelo guardado en {model_path}")


##############################
# FUNCIONES DE ENTRENAMIENTO #
##############################

def train_with_cv(X, y, params, cv_splits=5):
    """
    Entrena un modelo utilizando validación cruzada StratifiedGroupKFold.

    Args:
        X: Características
        y: Target
        params: Parámetros para el modelo XGBoost
        cv_splits: Número de pliegues para validación cruzada

    Returns:
        tuple: Mejor preprocesador, mejor modelo, mejor umbral
    """
    print(f"Entrenando con validación cruzada ({cv_splits} pliegues)...")

    # Verificar si X contiene hotel_id para la agrupación
    if 'hotel_id' not in X.columns:
        print("Advertencia: hotel_id no encontrado. Utilizando CV estándar.")
        # Crear una columna ficticia de hotel_id
        X = X.copy()
        X['hotel_id'] = np.arange(len(X))

    # Preparar validación cruzada
    cv = StratifiedGroupKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    groups = X['hotel_id'].values  # Usar hotel_id como grupo

    # Listas para almacenar resultados de cada fold
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

        # Definir preprocesador y aplicar
        preprocessor = get_preprocessor(X_train_fold_no_hotel)
        X_train_processed = preprocessor.fit_transform(X_train_fold_no_hotel)

        # Balancear datos de entrenamiento
        X_train_balanced, y_train_balanced = balance_dataset(X_train_processed, y_train_fold)

        # Entrenar modelo
        model = xgb.XGBClassifier(**params)
        model.fit(X_train_balanced, y_train_balanced)

        # Procesar datos de validación
        X_val_processed = preprocessor.transform(X_val_fold_no_hotel)

        # Encontrar umbral óptimo
        y_val_proba = model.predict_proba(X_val_processed)[:, 1]
        best_threshold, best_f1 = find_optimal_threshold(y_val_fold, y_val_proba)

        # Evaluar en datos de validación
        pipeline_temp = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        # Crear un diccionario para almacenar las métricas
        y_val_pred = (pipeline_temp.predict_proba(X_val_fold_no_hotel)[:, 1] >= best_threshold).astype(int)
        metrics = {
            'Exactitud': accuracy_score(y_val_fold, y_val_pred),
            'Puntuación F1': f1_score(y_val_fold, y_val_pred),
            'Precisión': precision_score(y_val_fold, y_val_pred),
            'Recuperación': recall_score(y_val_fold, y_val_pred),
            'AUC ROC': roc_auc_score(y_val_fold, pipeline_temp.predict_proba(X_val_fold_no_hotel)[:, 1])
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


def train_final_model(X_train, y_train, X_test=None, y_test=None):
    """
    Entrena el modelo final con validación cruzada y luego evalúa en el conjunto de prueba.

    Args:
        X_train: Características de entrenamiento
        y_train: Target de entrenamiento
        X_test: Características de prueba (opcional)
        y_test: Target de prueba (opcional)

    Returns:
        CustomPipeline: Pipeline entrenada final
    """
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

    print("Entrenando modelo final...")
    # Obtener mejor preprocesador, modelo y umbral mediante validación cruzada
    best_preprocessor, best_model, best_threshold = train_with_cv(X_train, y_train, xgb_params)

    # Eliminar hotel_id para el entrenamiento final si existe
    if 'hotel_id' in X_train.columns:
        X_train_no_hotel = X_train.drop(columns=['hotel_id'])
    else:
        X_train_no_hotel = X_train

    # Aplicar el mejor preprocesador encontrado
    X_processed = best_preprocessor.fit_transform(X_train_no_hotel)

    # Balancear datos
    X_balanced, y_balanced = balance_dataset(X_processed, y_train)

    # Crear y entrenar modelo final
    final_model = xgb.XGBClassifier(**xgb_params)
    final_model.fit(X_balanced, y_balanced)

    # Crear pipeline final
    final_pipe = CustomPipeline(best_preprocessor, final_model, best_threshold)

    print(f"Modelo final entrenado con umbral: {best_threshold:.4f}")

    # Evaluar en conjunto de prueba si está disponible
    if X_test is not None and y_test is not None:
        print("\nEvaluando en conjunto de prueba...")
        if 'hotel_id' in X_test.columns:
            X_test_no_hotel = X_test.drop(columns=['hotel_id'])
        else:
            X_test_no_hotel = X_test

        # Usar pipeline de scikit-learn para evaluación
        pipeline_temp = Pipeline([
            ('preprocessor', best_preprocessor),
            ('classifier', final_model)
        ])

        # Calcular métricas manualmente
        y_test_proba = pipeline_temp.predict_proba(X_test_no_hotel)[:, 1]
        y_test_pred = (y_test_proba >= best_threshold).astype(int)

        test_metrics = {
            'Exactitud': accuracy_score(y_test, y_test_pred),
            'Puntuación F1': f1_score(y_test, y_test_pred),
            'Precisión': precision_score(y_test, y_test_pred),
            'Recuperación': recall_score(y_test, y_test_pred),
            'AUC ROC': roc_auc_score(y_test, y_test_proba)
        }

        print("\nMétricas en conjunto de prueba:")
        for metric, value in test_metrics.items():
            print(f"{metric}: {value:.4f}")

    return final_pipe


################
# FUNCIÓN MAIN #
################

def main():
    """
    Función principal que ejecuta el flujo completo de entrenamiento y evaluación.
    """
    print("=" * 60)
    print("PREDICCIÓN DE CANCELACIONES ANTICIPADAS DE RESERVAS DE HOTEL")
    print("=" * 60)

    # Cargar y preparar datos
    X, y = get_X_y()
    print(f"Distribución de clases: {y.value_counts().to_dict()}")

    # Dividir en conjuntos de entrenamiento y prueba (70-30)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"Conjunto de entrenamiento: {X_train.shape[0]} muestras")
    print(f"Conjunto de prueba: {X_test.shape[0]} muestras")

    # Analizar correlación entre características
    correlated_features = check_feature_correlation(X_train)
    if correlated_features:
        print("\nCaracterísticas altamente correlacionadas:")
        for feat1, feat2, corr in correlated_features[:5]:  # Mostrar solo las 5 primeras
            print(f"  - {feat1} y {feat2}: {corr:.4f}")

        if len(correlated_features) > 5:
            print(f"  ... y {len(correlated_features) - 5} más")

    # Analizar correlación con target
    target_corrs = check_target_correlation(X_train, y_train)
    print("\nTop 10 características más correlacionadas con la variable objetivo:")
    for feat, corr in target_corrs.head(10).items():
        print(f"  - {feat}: {corr:.4f}")

    # Entrenar modelo final con conjunto de entrenamiento y evaluar en conjunto de prueba
    final_pipe = train_final_model(X_train, y_train, X_test, y_test)

    # Guardar modelo final
    save_model(final_pipe)

    print("\n¡Proceso completado con éxito!")


if __name__ == "__main__":
    main()