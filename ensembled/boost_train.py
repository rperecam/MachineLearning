import os
import warnings
import numpy as np
import pandas as pd
from sklearn import set_config
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score, confusion_matrix, classification_report)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import cloudpickle

# Configura sklearn para que los transformadores devuelvan DataFrames en lugar de arrays numpy
set_config(transform_output="pandas")

# Suprime advertencias para una salida más limpia
warnings.filterwarnings('ignore')


def get_X_y():
    """
    Carga y preprocesa los datos para el entrenamiento del modelo.
    Crea la variable objetivo y define los features.
    """
    hotels = pd.read_csv(os.environ.get("HOTELS_DATA_PATH", "data/hotels.csv"))
    bookings = pd.read_csv(os.environ.get("TRAIN_DATA_PATH", "data/bookings_train.csv"))

    # Une las tablas de reservas con los hoteles
    data = pd.merge(bookings, hotels, on='hotel_id', how='left')

    # Filtra solo reservas finalizadas o canceladas (excluye las aún en estado "Booked")
    data = data[data['reservation_status'] != 'Booked'].copy()

    # Considera los "No-Show" como "Check-Out" para el target
    data['reservation_status'].replace('No-Show', 'Check-Out', inplace=True)

    # Convierte fechas a formato datetime
    for col in ['arrival_date', 'booking_date', 'reservation_status_date']:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col])

    # Días entre fecha de llegada y la fecha del estado de la reserva
    data['days_before_arrival'] = (data['arrival_date'] - data['reservation_status_date']).dt.days

    # Define el target: cancelaciones hechas con 30 días o menos de anticipación
    data['target'] = ((data['reservation_status'] == 'Canceled') & (data['days_before_arrival'] <= 30)).astype(int)

    # Crea la variable lead_time: días entre reserva y llegada
    data['lead_time'] = (data['arrival_date'] - data['booking_date']).dt.days

    # Elimina columnas que ya no son necesarias para el modelo
    drop_cols = ['reservation_status', 'reservation_status_date', 'days_before_arrival',
                 'arrival_date', 'booking_date']
    drop_cols = [col for col in drop_cols if col in data.columns]

    X = data.drop(columns=['target'] + drop_cols)
    y = data['target']
    hotel_ids = data['hotel_id']  # Se usa para agrupar en la validación cruzada

    # Distribución del target
    print("\n--- Distribución Original del Target (y) ---")
    print(y.value_counts())
    print(f"Porcentaje de clase positiva (1 = Cancelación <= 30 días): {y.mean() * 100:.2f}%")
    print("Total de registros:", len(y))
    print("-------------------------------------------\n")

    return X, y, hotel_ids


def create_pipeline(X, y):
    """
    Crea un pipeline completo con preprocesamiento y modelo.
    Incluye imputación, codificación, escalado y SMOTE para rebalancear.
    """
    # Detecta tipos de variables
    num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    bool_features = X.select_dtypes(include=["bool"]).columns.tolist()
    cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    cat_features = [col for col in cat_features if col not in bool_features]

    # Evita usar 'hotel_id' como variable numérica
    if 'hotel_id' in num_features:
        num_features.remove('hotel_id')

    # Pipeline para variables numéricas: imputación y escalado
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Pipeline para variables categóricas: imputación y one-hot encoding
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # Aplica transformaciones por tipo de columna
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_features + bool_features),
            ("cat", categorical_transformer, cat_features),
        ]
    )

    # Modelo base: XGBoost con configuración personalizada
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

    # Pipeline completo con SMOTE para reequilibrar clases
    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42, sampling_strategy=0.6, k_neighbors=5)),
        ('classifier', model)
    ])

    return pipeline


def evaluate_model(y_true, y_pred_proba, threshold=0.5):
    """
    Evalúa el modelo usando múltiples métricas.
    Imprime un reporte detallado del rendimiento.
    """
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Calcula métricas principales
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_pred_proba)

    # Matriz de confusión
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Métricas adicionales
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    specificity = tn / (tn + fp)

    # Imprime resultados
    print("\n------ Evaluación del Modelo ------")
    print(f"Umbral aplicado: {threshold:.4f}")
    print(f"\nMétricas principales:")
    print(f"→ Precision:   {precision:.4f} (De las predicciones positivas, ¿cuántas son correctas?)")
    print(f"→ Recall:      {recall:.4f} (De todos los positivos reales, ¿cuántos detectamos?)")
    print(f"→ F1-Score:    {f1:.4f} (Media armónica entre precision y recall)")
    print(f"→ AUC-ROC:     {auc_roc:.4f} (Capacidad discriminativa general del modelo)")

    print(f"\nMétricas adicionales:")
    print(f"→ Accuracy:    {accuracy:.4f} (Porcentaje general de aciertos)")
    print(f"→ Specificity: {specificity:.4f} (De los negativos reales, ¿cuántos detectamos?)")

    print(f"\nMatriz de confusión:")
    print(f"→ Verdaderos Negativos (TN): {tn} (No cancela y predecimos que no cancela)")
    print(f"→ Falsos Positivos (FP):    {fp} (No cancela pero predecimos cancelación)")
    print(f"→ Falsos Negativos (FN):    {fn} (Cancela pero no lo detectamos)")
    print(f"→ Verdaderos Positivos (TP): {tp} (Cancela y predecimos correctamente)")

    # Tasas derivadas importantes para negocios
    print(f"\nMétricas de negocio:")
    print(
        f"→ Tasa de falsa alarma:  {fp / (fp + tn):.4f} (De todas las reservas que no cancelan, ¿cuántas marcamos erróneamente?)")
    print(f"→ Tasa de pérdida:       {fn / (fn + tp):.4f} (De todas las cancelaciones, ¿cuántas no detectamos?)")

    print("\nReporte de clasificación detallado:")
    print(classification_report(y_true, y_pred))

    return threshold, f1


def find_threshold(y_true, y_pred_proba):
    """
    Encuentra el mejor umbral de decisión para maximizar el F1-score.
    """
    best_f1, best_threshold = 0, 0.5
    thresholds = np.linspace(0.1, 0.8, 40)

    print("\n------ Búsqueda de umbral óptimo ------")
    print("Umbral\t\tF1\t\tPrecision\tRecall")

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)

        print(f"{threshold:.2f}\t\t{f1:.4f}\t\t{precision:.4f}\t\t{recall:.4f}")

        if f1 > best_f1:
            best_f1, best_threshold = f1, threshold

    print(f"\nUmbral óptimo seleccionado: {best_threshold:.4f} (F1: {best_f1:.4f})")
    return best_threshold, best_f1


def save_model(pipeline, threshold, path=None):
    """
    Guarda el pipeline del modelo junto con el umbral óptimo.
    """
    model_path = path or os.environ.get("MODEL_PATH", "models/pipeline.cloudpkl")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model_package = {'pipeline': pipeline, 'threshold': threshold}
    with open(model_path, mode="wb") as f:
        cloudpickle.dump(model_package, f)
    print(f"Modelo guardado en {model_path}")


def main():
    """
    Ejecuta el proceso completo:
    - carga y preprocesamiento de datos
    - entrenamiento con validación cruzada
    - búsqueda del mejor umbral
    - ajuste final del modelo
    - guardado del modelo entrenado
    """
    print("Iniciando entrenamiento...")

    # Carga datos y define target
    X, y, hotel_ids = get_X_y()

    # Crea pipeline de preprocesamiento y modelo
    pipeline = create_pipeline(X, y)

    # Validación cruzada estratificada por grupo (hotel)
    cv = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=42)

    print(f"\nRealizando validación cruzada con {cv.n_splits} splits (estratificada por hotel_id)...")

    # Predice probabilidades en validación cruzada
    y_pred_proba = cross_val_predict(pipeline, X, y, cv=cv, groups=hotel_ids, method='predict_proba')[:, 1]

    # Encuentra el mejor umbral de decisión
    optimal_threshold, best_f1 = find_threshold(y, y_pred_proba)

    # Evalúa el modelo con el umbral óptimo
    evaluate_model(y, y_pred_proba, optimal_threshold)

    # Entrena el pipeline completo en todos los datos
    print("\nEntrenando modelo final con todos los datos...")
    pipeline.fit(X, y)

    # Guarda el pipeline junto al umbral óptimo
    save_model(pipeline, optimal_threshold)

    print("\nEntrenamiento completado. ¡El modelo está listo para inferencia!")


# Punto de entrada del script
if __name__ == "__main__":
    main()