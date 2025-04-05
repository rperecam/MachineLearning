import warnings
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score, precision_recall_curve, roc_curve, auc
from sklearn.base import BaseEstimator, TransformerMixin
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import cloudpickle
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
                 country_cols=['country_x', 'country_y']):
        self.date_cols = date_cols
        self.asset_cols = asset_cols
        self.country_cols = country_cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()

        # Verificar y calcular lead_time
        if 'arrival_date' in X_copy.columns and 'booking_date' in X_copy.columns:
            X_copy['arrival_date'] = pd.to_datetime(X_copy['arrival_date'], errors='coerce')
            X_copy['booking_date'] = pd.to_datetime(X_copy['booking_date'], errors='coerce')
            X_copy['lead_time'] = (X_copy['arrival_date'] - X_copy['booking_date']).dt.days
        else:
            X_copy['lead_time'] = np.nan

        # num_assets
        present_assets = [col for col in self.asset_cols if col in X_copy.columns]
        X_copy['num_assets'] = X_copy[present_assets].fillna(0).astype(int).sum(axis=1)

        # is_foreign
        if all(col in X_copy.columns for col in self.country_cols):
            X_copy['is_foreign'] = (X_copy[self.country_cols[0]] != X_copy[self.country_cols[1]]).astype(int)
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
    df_eng['arrival_date'] = pd.to_datetime(df_eng['arrival_date'], errors='coerce')
    df_eng['reservation_status_date'] = pd.to_datetime(df_eng['reservation_status_date'], errors='coerce')

    # Simplificación de la lógica de cancelación tardía
    df_eng['is_canceled'] = df_eng['reservation_status'].isin(['Canceled', 'No-Show']).astype(int)
    df_eng['days_to_arrival'] = (df_eng['arrival_date'] - df_eng['reservation_status_date']).dt.days
    df_eng['cancelled_last_30_days'] = (df_eng['is_canceled'] == 1) & (df_eng['days_to_arrival'] <= 30)
    y = df_eng['cancelled_last_30_days'].fillna(False).astype(int)
    X = df_eng.drop(columns=['reservation_status', 'arrival_date', 'reservation_status_date',
                             'is_canceled', 'days_to_arrival'], errors='ignore')
    return X, y

# --- Clase de Pipeline Simplificada ---
class HotelBookingPipeline:
    def __init__(self, test_size=0.3, random_state=42, variance_threshold=0.001,
                 model_type='logistic', cv_folds=5, k_best=10, outlier_columns=None):
        self.test_size = test_size
        self.random_state = random_state
        self.variance_threshold = variance_threshold
        self.model_type = model_type
        self.cv_folds = cv_folds
        self.k_best = k_best
        self.outlier_columns = outlier_columns
        self.pipeline = self._build_pipeline()
        self.best_model = None
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
            remainder='drop'  # Eliminar columnas restantes que no se transforman
        )

        pipeline_steps = [
            ('feature_engineer', FeatureEngineer()),
            ('continent_mapper', ContinentMapper()),
            ('outlier_capper', OutlierCapper(columns=self.outlier_columns)),
            ('preprocessor', preprocessor),
            ('variance_threshold', VarianceThreshold(threshold=self.variance_threshold)),
            ('smote', SMOTE(random_state=self.random_state)),
            ('feature_selection', SelectKBest(score_func=f_classif, k=self.k_best))
        ]

        if self.model_type == 'logistic':
            pipeline_steps.append(('classifier', LogisticRegression(random_state=self.random_state, solver='liblinear', max_iter=2000)))
            self.param_grid = {
                'classifier__C': [0.001, 0.01, 0.1, 1, 10],
                'classifier__penalty': ['l1', 'l2']
            }
        elif self.model_type == 'sgd':
            pipeline_steps.append(('classifier', SGDClassifier(random_state=self.random_state, loss='log_loss', max_iter=1000, tol=1e-3)))
            self.param_grid = {
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
        grid_search = GridSearchCV(self.pipeline, self.param_grid, cv=cv, scoring='f1', n_jobs=-1, verbose=1, error_score='raise')
        grid_search.fit(X_train, y_train)

        self.best_model = grid_search.best_estimator_
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
        print(f"Métricas en el conjunto de prueba:")
        for metric, value in self.metrics.items():
            print(f"{metric}: {value:.4f}")

    def save_model(self, filepath: str):
        if self.best_model:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'wb') as f:
                cloudpickle.dump(self.best_model, f)
            print(f"Modelo guardado en: {filepath}")
        else:
            print("Advertencia: No hay un modelo entrenado para guardar.")

    def load_model(self, filepath: str):
        try:
            with open(filepath, 'rb') as f:
                self.best_model = cloudpickle.load(f)
            print(f"Modelo cargado desde: {filepath}")
        except FileNotFoundError:
            print(f"Error: Archivo no encontrado en: {filepath}")
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")

if __name__ == '__main__':
    # Definir las rutas de los archivos
    bookings_file_train = 'data/bookings_train.csv'
    hotels_file = 'data/hotels.csv'
    model_path = 'model/model.pkl'

    # Crear una instancia del pipeline
    pipeline = HotelBookingPipeline(
        test_size=0.25,
        random_state=42,
        variance_threshold=0.001,
        model_type='logistic',  # Puedes cambiar a 'sgd' si prefieres usar SGDClassifier
        cv_folds=5,
        k_best=22,
        outlier_columns=('rate', 'stay_nights', 'total_guests') # Especifica las columnas si deseas manejar outliers
    )

    # Entrenar el modelo
    pipeline.train(bookings_file=bookings_file_train, hotels_file=hotels_file)

    # Guardar el modelo entrenado
    pipeline.save_model(filepath=model_path)
