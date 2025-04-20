import os
import pandas as pd
import numpy as np
import cloudpickle
from sklearn import set_config
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
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
        # Preprocesar datos
        X_transformed = self.preprocessor.fit_transform(X)
        self.feature_names = X_transformed.columns

        # Entrenar modelo
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
    Identifica y visualiza características altamente correlacionadas.

    Args:
        X: DataFrame con características
        threshold: Umbral de correlación para considerar dos características como correlacionadas

    Returns:
        lista de tuplas de características correlacionadas
    """
    print("Analizando correlaciones entre características...")

    # Seleccionar solo características numéricas
    numeric_X = X.select_dtypes(include=['int64', 'float64'])

    # Calcular matriz de correlación
    corr_matrix = numeric_X.corr(method='spearman').abs()

    # Encontrar pares de características altamente correlacionadas
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    highly_correlated = [(corr_matrix.index[i], corr_matrix.columns[j], upper_tri.iloc[i, j])
                         for i, j in zip(*np.where(upper_tri > threshold))]

    if highly_correlated:
        print(f"Características altamente correlacionadas (>{threshold}):")
        for feat1, feat2, corr in highly_correlated:
            print(f"  - {feat1} y {feat2}: {corr:.4f}")

        # Visualizar las correlaciones más altas
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', vmin=0, vmax=1)
        plt.title('Matriz de Correlación de Características')
        plt.tight_layout()
        plt.savefig('correlation_matrix.png')
        print("Matriz de correlación guardada como 'correlation_matrix.png'")
    else:
        print("No se encontraron características altamente correlacionadas.")

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
    print("Analizando correlación con variable objetivo...")

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

    print("Correlaciones más significativas con target:")
    for feature, corr in corr_series.head(10).items():
        print(f"  - {feature}: {corr:.4f}")

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
    hotels = pd.read_csv('data/hotels.csv')
    bookings = pd.read_csv('data/bookings_train.csv')

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
        'arrival_date', 'days_before_arrival', 'country_x', 'country_y','special_request'
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

    print(f"Datos cargados: X shape={X.shape}, y shape={y.shape}")
    print(f"Distribución de target: {pd.Series(y).value_counts(normalize=True)}")

    return X, y

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
    print(f"Aplicando balanceo de datos con método: {method}")
    print(f"Distribución original: {pd.Series(y).value_counts(normalize=True)}")

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

    print(f"Distribución después del balanceo: {pd.Series(y_balanced).value_counts(normalize=True)}")
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

    return metrics

def plot_feature_importance(model, feature_names, top_n=20):
    """
    Visualiza la importancia de características.

    Args:
        model: Modelo entrenado (debe tener atributo feature_importances_)
        feature_names: Lista de nombres de características
        top_n: Número de características principales a mostrar
    """
    if hasattr(model, 'feature_importances_'):
        # Obtener importancia de características
        importances = model.feature_importances_

        # Crear DataFrame para visualización
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)

        # Mostrar las características más importantes
        top_features = feature_importance.head(top_n)

        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=top_features)
        plt.title(f'Top {top_n} Características Más Importantes')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        print(f"Gráfico de importancia de características guardado como 'feature_importance.png'")

        return feature_importance
    else:
        print("El modelo no tiene atributo feature_importances_")
        return None

def save_model(pipe, filename="models/xgboost_model.pkl"):
    """
    Serializa el modelo entrenado.

    Args:
        pipe: Pipeline entrenada
        filename: Nombre del archivo para guardar
    """
    with open(os.environ.get("MODEL_PATH", filename), mode="wb") as f:
        cloudpickle.dump(pipe, f)
    print(f"Modelo guardado en {os.environ.get('MODEL_PATH', filename)}")

def nested_cross_validation(X, y, preprocessor, n_outer=5, n_inner=3):
    """
    Realiza validación cruzada anidada para evaluar el modelo de forma más robusta.

    Args:
        X: Características
        y: Target
        preprocessor: Preprocesador definido
        n_outer: Número de folds externos
        n_inner: Número de folds internos

    Returns:
        scores: Lista de métricas por fold externo
    """
    print(f"Realizando validación cruzada anidada ({n_outer}x{n_inner} folds)...")

    # Definir folds externos
    outer_cv = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=42)

    # Almacenar resultados por fold
    outer_scores = []

    # Iteración sobre folds externos
    for i, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        print(f"\nFold externo {i+1}/{n_outer}")

        # Dividir datos
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Preprocesador ajustado solo a los datos de entrenamiento
        fold_preprocessor = preprocessor.fit(X_train, y_train)
        X_train_processed = fold_preprocessor.transform(X_train)

        # Balancear solo los datos de entrenamiento
        X_train_balanced, y_train_balanced = balance_dataset(X_train_processed, y_train)

        # Parámetros base para XGBoost
        xgb_params = {
            'max_depth': 10,
            'min_child_weight': 1,
            'gamma': 0.437,
            'subsample': 0.805,
            'colsample_bytree': 0.538,
            'reg_alpha': 0.765,
            'reg_lambda': 3.645,
            'learning_rate': 0.09,
            'n_estimators': 800,
            'scale_pos_weight': 1.0,  # Ajustado porque ya balanceamos los datos
            'objective': 'binary:logistic',
            'tree_method': 'hist',
            'random_state': 42,
            'n_jobs': -1
        }

        # Definir modelo XGBoost para este fold
        model = xgb.XGBClassifier(**xgb_params)

        # Entrenar modelo con datos balanceados
        print("Entrenando modelo...")
        model.fit(X_train_balanced, y_train_balanced)

        # Transformar datos de prueba
        X_test_processed = fold_preprocessor.transform(X_test)

        # Evaluar en datos de prueba
        y_pred_proba = model.predict_proba(X_test_processed)[:, 1]

        # Encontrar umbral óptimo
        best_threshold, _ = find_optimal_threshold(y_test, y_pred_proba)
        print(f"Umbral óptimo: {best_threshold:.4f}")

        # Calcular métricas
        y_pred = (y_pred_proba >= best_threshold).astype(int)
        fold_metrics = {
            'Exactitud': accuracy_score(y_test, y_pred),
            'Puntuación F1': f1_score(y_test, y_pred),
            'Precisión': precision_score(y_test, y_pred),
            'Recuperación': recall_score(y_test, y_pred),
            'AUC ROC': roc_auc_score(y_test, y_pred_proba),
            'Umbral': best_threshold
        }

        print("Métricas del fold:")
        for metric, value in fold_metrics.items():
            print(f"  - {metric}: {value:.4f}")

        outer_scores.append(fold_metrics)

    # Calcular métricas promedio
    avg_metrics = {}
    for metric in outer_scores[0].keys():
        avg_metrics[metric] = np.mean([score[metric] for score in outer_scores])

    print("\nMétricas promedio de validación cruzada:")
    for metric, value in avg_metrics.items():
        print(f"  - {metric}: {value:.4f}")

    return outer_scores, avg_metrics

def train_final_model(X, y):
    """
    Entrena el modelo final utilizando todo el conjunto de datos y las mejores prácticas.

    Args:
        X: Características
        y: Target

    Returns:
        pipe: Pipeline entrenada final
    """
    # Dividir datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"División train/test: {X_train.shape}, {X_test.shape}")

    # Verificar correlaciones entre características
    correlated_features = check_feature_correlation(X_train)

    # Verificar correlación con target
    feature_importance = check_target_correlation(X_train, y_train)

    # Definir preprocesador
    preprocessor = get_preprocessor(X_train)

    # Aplicar preprocesamiento
    print("Aplicando preprocesamiento a datos de entrenamiento...")
    X_train_processed = preprocessor.fit_transform(X_train)

    # Balancear datos de entrenamiento
    X_train_balanced, y_train_balanced = balance_dataset(X_train_processed, y_train)

    # Parámetros optimizados para XGBoost
    xgb_params = {
        'max_depth': 8,  # Reducido para prevenir sobreajuste
        'min_child_weight': 2,  # Aumentado para prevenir sobreajuste
        'gamma': 0.4,
        'subsample': 0.8,
        'colsample_bytree': 0.5,
        'reg_alpha': 0.7,
        'reg_lambda': 3.6,
        'learning_rate': 0.05,  # Reducido para mejor generalización
        'n_estimators': 500,
        'scale_pos_weight': 1.0,  # Ajustado porque ya balanceamos los datos
        'objective': 'binary:logistic',
        'tree_method': 'hist',
        'random_state': 42,
        'n_jobs': -1
    }

    # Crear y entrenar modelo
    print("Entrenando modelo XGBoost...")
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(X_train_balanced, y_train_balanced)

    # Visualizar importancia de características
    feature_importances = plot_feature_importance(model, X_train_balanced.columns)

    # Seleccionar características importantes
    print("Seleccionando características importantes...")
    selector = SelectFromModel(model, threshold='mean', prefit=True)
    feature_idx = selector.get_support()
    selected_features = X_train_balanced.columns[feature_idx].tolist()
    print(f"Seleccionadas {len(selected_features)} características de {X_train_balanced.shape[1]}")
    print("Características seleccionadas:")
    for feature in selected_features:
        print(f"  - {feature}")

    # Aplicar preprocesamiento a datos de prueba
    X_test_processed = preprocessor.transform(X_test)

    # Evaluar en conjunto de prueba
    print("\nEvaluando en conjunto de prueba...")
    y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
    best_threshold, _ = find_optimal_threshold(y_test, y_pred_proba)
    print(f"Umbral óptimo: {best_threshold:.4f}")

    # Calcular métricas
    test_metrics = evaluate_model(model, X_test_processed, y_test, best_threshold)

    print("Métricas finales en conjunto de prueba:")
    for metric, value in test_metrics.items():
        print(f"  - {metric}: {value:.4f}")

    # Crear pipeline final
    final_pipe = CustomPipeline(preprocessor, model, best_threshold)

    return final_pipe

if __name__ == "__main__":
    print("==== SISTEMA DE PREDICCIÓN DE CANCELACIONES DE HOTEL ====")
    print("Iniciando proceso...")

    # Cargar y preparar datos
    X, y = get_X_y()

    # Ejecutar validación cruzada anidada para una evaluación realista
    nested_scores, avg_metrics = nested_cross_validation(X, y, get_preprocessor(X))

    # Entrenar modelo final con todo el conjunto de datos
    print("\n==== ENTRENAMIENTO DE MODELO FINAL ====")
    final_pipe = train_final_model(X, y)

    # Guardar modelo final
    save_model(final_pipe)

    print("\nProceso completado con éxito.")