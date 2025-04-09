import warnings
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score, precision_recall_curve, auc
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import joblib
import os
from typing import Optional, Tuple
import numpy as np

# Ignorar advertencias
warnings.filterwarnings("ignore")

# --- Mapeo de Continentes Simplificado ---
class ContinentMapper(BaseEstimator, TransformerMixin):
    def __init__(self, country_col='country_x', continent_col='continent_customer', unknown='Desconocido'):
        self.country_col = country_col
        self.continent_col = continent_col
        self.unknown = unknown
        self.mapping = {
            'SPA': 'Europa', 'FRA': 'Europa', 'POR': 'Europa', 'AUT': 'Europa', 'NLD': 'Europa', 'ITA': 'Europa', 'GBR': 'Europa', 'DEU': 'Europa', 'DNK': 'Europa', 'POL': 'Europa', 'BEL': 'Europa',
            'FIN': 'Europa', 'NOR': 'Europa', 'HUN': 'Europa', 'CHE': 'Europa', 'ROU': 'Europa', 'SWE': 'Europa', 'UKR': 'Europa', 'GRC': 'Europa', 'LUX': 'Europa', 'MLT': 'Europa', 'CYP': 'Europa',
            'SVK': 'Europa', 'SRB': 'Europa', 'LTU': 'Europa', 'BIH': 'Europa', 'MKD': 'Europa', 'BGR': 'Europa', 'CZE': 'Europa', 'EST': 'Europa', 'LVA': 'Europa', 'ISL': 'Europa', 'SVN': 'Europa',
            'ALB': 'Europa', 'LIE': 'Europa', 'MNE': 'Europa', 'AND': 'Europa', 'IRL': 'Europa', 'HRV': 'Europa', 'IMN': 'Europa', 'FRO': 'Europa', 'GIB': 'Europa', 'SMR': 'Europa', 'GGY': 'Europa', 'JEY': 'Europa', 'GEO': 'Europa',
            'JPN': 'Asia', 'ISR': 'Asia', 'CHN': 'Asia', 'IND': 'Asia', 'IRN': 'Asia', 'IRQ': 'Asia', 'PHL': 'Asia', 'MYS': 'Asia', 'SGP': 'Asia', 'TWN': 'Asia', 'THA': 'Asia',
            'LKA': 'Asia', 'KWT': 'Asia', 'JOR': 'Asia', 'TUR': 'Asia', 'ARE': 'Asia', 'KOR': 'Asia', 'UZB': 'Asia', 'KAZ': 'Asia', 'MAC': 'Asia', 'HKG': 'Asia', 'KHM': 'Asia',
            'BGD': 'Asia', 'AZE': 'Asia', 'LBN': 'Asia', 'SYR': 'Asia', 'VNM': 'Asia', 'QAT': 'Asia', 'OMN': 'Asia', 'PAK': 'Asia', 'TMP': 'Asia', 'NPL': 'Asia', 'IDN': 'Asia', 'SAU': 'Asia', 'MMR': 'Asia', 'ARM': 'Asia',
            'AGO': 'África', 'CMR': 'África', 'DZA': 'África', 'EGY': 'África', 'MAR': 'África', 'ZAF': 'África', 'MOZ': 'África', 'TUN': 'África', 'GNB': 'África', 'NGA': 'África', 'CAF': 'África',
            'KEN': 'África', 'RWA': 'África', 'CIV': 'África', 'SYC': 'África', 'ETH': 'África', 'SEN': 'África', 'GHA': 'África', 'SDN': 'África', 'GAB': 'África', 'BEN': 'África', 'ZMB': 'África',
            'MWI': 'África', 'UGA': 'África', 'ZWE': 'África', 'MUS': 'África', 'TZA': 'África', 'CPV': 'África', 'NAM': 'África', 'MDG': 'África', 'MYT': 'África', 'REU': 'África', 'BWA': 'África',
            'USA': 'América del Norte', 'MEX': 'América del Norte', 'CAN': 'América del Norte', 'CUB': 'América del Norte', 'DOM': 'América del Norte', 'PRI': 'América del Norte', 'CYM': 'América del Norte', 'BHS': 'América del Norte', 'BRB': 'América del Norte',
            'VGB': 'América del Norte', 'JAM': 'América del Norte', 'LCA': 'América del Norte', 'PAN': 'América del Norte', 'CRI': 'América del Norte', 'GTM': 'América del Norte', 'NIC': 'América del Norte', 'HND': 'América del Norte',
            'BRA': 'América del Sur', 'ARG': 'América del Sur', 'ECU': 'América del Sur', 'COL': 'América del Sur', 'PER': 'América del Sur', 'URY': 'América del Sur', 'VEN': 'América del Sur', 'CHL': 'América del Sur', 'BOL': 'América del Sur',
            'PRY': 'América del Sur', 'SUR': 'América del Sur', 'GUF': 'América del Sur', 'GUY': 'América del Sur',
            'AUS': 'Oceanía', 'NZL': 'Oceanía', 'PYF': 'Oceanía', 'NCL': 'Oceanía', 'FJI': 'Oceanía',
            'ATA': 'Antártida',
            'CN': 'Otros'
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        if self.country_col in X_copy.columns:
            X_copy[self.continent_col] = X_copy[self.country_col].map(self.mapping).fillna(self.unknown)
        else:
            X_copy[self.continent_col] = self.unknown
        return X_copy

# --- Ingeniería de Características Simplificada ---
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, date_cols=['arrival_date', 'booking_date'],
                 asset_cols=['pool_and_spa', 'restaurant', 'parking'],
                 country_cols=['country_x', 'country_y'],
                 date_format=None):
        self.date_cols = date_cols
        self.asset_cols = asset_cols
        self.country_cols = country_cols
        self.date_format = date_format

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()

        arrival_col = self.date_cols[0]
        booking_col = self.date_cols[1]

        # Verificar y calcular lead_time
        if arrival_col in X_copy.columns and booking_col in X_copy.columns:
            try:
                X_copy[arrival_col] = pd.to_datetime(X_copy[arrival_col],
                                                     format=self.date_format,
                                                     errors='coerce')
                X_copy[booking_col] = pd.to_datetime(X_copy[booking_col],
                                                     format=self.date_format,
                                                     errors='coerce')

                time_diff = X_copy[arrival_col] - X_copy[booking_col]
                X_copy['lead_time'] = time_diff.dt.days

            except Exception as e:
                print(f"Error al convertir fechas o calcular lead_time: {e}")
                X_copy['lead_time'] = np.nan

        else:
            print(f"Advertencia: Columnas de fecha '{arrival_col}' o '{booking_col}' no encontradas. 'lead_time' será NaN.")
            X_copy['lead_time'] = np.nan

        # num_assets
        present_assets = [col for col in self.asset_cols if col in X_copy.columns]
        if present_assets:
            X_copy['num_assets'] = X_copy[present_assets].fillna(0).astype(int).sum(axis=1)
        else:
            X_copy['num_assets'] = 0

        # is_foreign
        country_col_x = self.country_cols[0]
        country_col_y = self.country_cols[1]
        if country_col_x in X_copy.columns and country_col_y in X_copy.columns:
            X_copy['is_foreign'] = (X_copy[country_col_x].astype(str) != X_copy[country_col_y].astype(str)).astype(int)
            X_copy.loc[X_copy[country_col_x].isna() | X_copy[country_col_y].isna(), 'is_foreign'] = 0
        else:
            X_copy['is_foreign'] = 0

        return X_copy

# --- Manejo de Outliers (Cap/Floor) ---
class OutlierCapper(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None, factor=1.5):
        self.columns = columns
        self.factor = factor
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}

    def fit(self, X, y=None):
        if self.columns is None:
            self.columns = X.select_dtypes(include=np.number).columns.tolist()
        for col in self.columns:
            if col in X.columns:
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                self.lower_bounds_[col] = Q1 - self.factor * IQR
                self.upper_bounds_[col] = Q3 + self.factor * IQR
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col in self.columns:
            if col in self.lower_bounds_:
                X_copy[col] = np.where(X_copy[col] < self.lower_bounds_[col], self.lower_bounds_[col], X_copy[col])
                X_copy[col] = np.where(X_copy[col] > self.upper_bounds_[col], self.upper_bounds_[col], X_copy[col])
        return X_copy

# --- Ingeniería de la Variable Objetivo Simplificada ---
def engineer_target(df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    if not all(col in df.columns for col in ['reservation_status', 'arrival_date', 'reservation_status_date']):
        print("Error: Columnas necesarias para la ingeniería del target no encontradas.")
        return None, None

    df_eng = df.copy()
    try:
        df_eng['arrival_date'] = pd.to_datetime(df_eng['arrival_date'], errors='coerce')
        df_eng['reservation_status_date'] = pd.to_datetime(df_eng['reservation_status_date'], errors='coerce')
    except Exception as e:
        print(f"Error convirtiendo fechas en engineer_target (puede ser normal si ya se hizo): {e}")

    df_eng['is_canceled'] = df_eng['reservation_status'].isin(['Canceled', 'No-Show']).astype(int)
    if pd.api.types.is_datetime64_any_dtype(df_eng['arrival_date']) and pd.api.types.is_datetime64_any_dtype(df_eng['reservation_status_date']):
        df_eng['days_to_arrival'] = (df_eng['arrival_date'] - df_eng['reservation_status_date']).dt.days
    else:
        print("Advertencia: No se pudo calcular days_to_arrival, fechas no son datetime.")
        df_eng['days_to_arrival'] = np.nan

    df_eng['cancelled_last_30_days'] = (df_eng['is_canceled'] == 1) & (df_eng['days_to_arrival'] <= 30)
    y = df_eng['cancelled_last_30_days'].fillna(False).astype(int)

    columns_to_drop = ['reservation_status', 'reservation_status_date',
                       'is_canceled', 'days_to_arrival', 'cancelled_last_30_days']
    columns_to_drop = [col for col in columns_to_drop if col in df_eng.columns]
    X = df_eng.drop(columns=columns_to_drop, errors='ignore')

    return X, y

# --- Clase de Pipeline Modificada para Optimizar k_best ---
class HotelBookingPipeline:
    def __init__(self, test_size=0.3, random_state=42, variance_threshold=0.001,
                 model_type='logistic', cv_folds=5, k_range=None, outlier_columns=None):
        self.test_size = test_size
        self.random_state = random_state
        self.variance_threshold = variance_threshold
        self.model_type = model_type
        self.cv_folds = cv_folds
        # Ahora usamos un rango de k_best en lugar de un valor fijo
        self.k_range = k_range if k_range else list(range(5, 51, 5))  # Por defecto: [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        self.outlier_columns = outlier_columns
        self.pipeline = self._build_pipeline()
        self.best_model = None
        self.best_k = None
        self.metrics = None

    def _build_pipeline(self):
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, make_column_selector(dtype_include=np.number)),
                ('cat', categorical_transformer, make_column_selector(dtype_include='object'))
            ],
            remainder='drop'
        )

        # Construir la primera parte de la pipeline (hasta feature_selection)
        pipeline_steps = [
            ('feature_engineer', FeatureEngineer()),
            ('continent_mapper', ContinentMapper()),
            ('outlier_capper', OutlierCapper(columns=self.outlier_columns)),
            ('preprocessor', preprocessor),
            ('variance_threshold', VarianceThreshold(threshold=self.variance_threshold)),
            ('smote', SMOTE(random_state=self.random_state)),
            # Incluimos SelectKBest pero el valor de k será determinado por GridSearchCV
            ('feature_selection', SelectKBest(score_func=f_classif))
        ]

        # Agregar el clasificador según el tipo seleccionado
        if self.model_type == 'logistic':
            pipeline_steps.append(('classifier', LogisticRegression(random_state=self.random_state, solver='liblinear', max_iter=2000)))
            self.param_grid = {
                'feature_selection__k': self.k_range,  # Probar diferentes valores de k
                'classifier__C': [0.001, 0.01, 0.1, 1, 10],
                'classifier__penalty': ['l1', 'l2']
            }
        elif self.model_type == 'sgd':
            pipeline_steps.append(('classifier', SGDClassifier(random_state=self.random_state, loss='log_loss', max_iter=1000, tol=1e-3)))
            self.param_grid = {
                'feature_selection__k': self.k_range,  # Probar diferentes valores de k
                'classifier__alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1],
                'classifier__penalty': ['l2', 'elasticnet', 'none']
            }
        else:
            raise ValueError(f"Modelo no soportado: {self.model_type}")

        return ImbPipeline(pipeline_steps)

    def train(self, bookings_file: str, hotels_file: str):
        try:
            df_book = pd.read_csv(bookings_file)
            df_hotel = pd.read_csv(hotels_file)
        except FileNotFoundError as e:
            print(f"Error al cargar los archivos: {e}")
            return

        df = pd.merge(df_book, df_hotel, on='hotel_id', how='left').drop('hotel_id', axis=1, errors='ignore')
        df_model = df[~df['reservation_status'].isin(['Booked', np.nan])].copy()
        if df_model.empty:
            print("Advertencia: No hay datos para entrenar el modelo después de filtrar el estado 'Booked'.")
            return

        X, y = engineer_target(df_model)
        if X is None or y is None:
            return

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size,
                                                            stratify=y, random_state=self.random_state)

        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        grid_search = GridSearchCV(self.pipeline, self.param_grid, cv=cv, scoring='f1', verbose=1, error_score='raise')
        grid_search.fit(X_train, y_train)

        self.best_model = grid_search.best_estimator_
        self.best_k = grid_search.best_params_['feature_selection__k']  # Guardar el mejor valor de k

        y_pred = self.best_model.predict(X_test)
        y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]

        self.metrics = {
            'f1_score': f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'pr_auc': auc(precision_recall_curve(y_test, y_pred_proba)[1], precision_recall_curve(y_test, y_pred_proba)[0])
        }

        print(f"Mejores parámetros: {grid_search.best_params_}")
        print(f"Mejor valor de k encontrado: {self.best_k}")  # Mostrar el mejor k
        print(f"Métricas en el conjunto de prueba:")
        for metric, value in self.metrics.items():
            print(f"{metric}: {value:.4f}")

    def save_model(self, filepath: str):
        if self.best_model:
            try:
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                joblib.dump(self.best_model, filepath)
                print(f"Modelo guardado en: {filepath}")
            except Exception as e:
                print(f"Error al guardar el modelo con joblib: {e}")
        else:
            print("Advertencia: No hay un modelo entrenado para guardar.")

    def load_model(self, filepath: str):
        try:
            self.best_model = joblib.load(filepath)
            print(f"Modelo cargado desde: {filepath}")
        except FileNotFoundError:
            print(f"Error: Archivo no encontrado en: {filepath}")
        except Exception as e:
            print(f"Error al cargar el modelo con joblib: {e}")

if __name__ == '__main__':
    bookings_file_train = '../data/bookings_train.csv'
    hotels_file = '../data/hotels.csv'
    model_path = '../model/model.pkl'

    # Configuración de la pipeline con un rango de valores para k_best
    pipeline = HotelBookingPipeline(
        test_size=0.25,
        random_state=42,
        variance_threshold=0.001,
        model_type='logistic',
        cv_folds=5,
        k_range=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
        outlier_columns=('rate', 'stay_nights', 'total_guests')
    )

    pipeline.train(bookings_file=bookings_file_train, hotels_file=hotels_file)
    pipeline.save_model(filepath=model_path)