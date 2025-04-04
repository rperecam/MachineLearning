# -*- coding: utf-8 -*-
import warnings
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline # Changed from ImbPipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_curve, auc, precision_recall_curve, make_scorer, confusion_matrix, classification_report)
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
import seaborn as sns
# Removed SMOTE import
import sys
import cloudpickle
import os
import traceback
from typing import List, Dict, Optional, Tuple, Any

# Ignorar advertencias específicas
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Custom Transformer for Continent Mapping (Enhanced Robustness) ---
class ContinentMapper(BaseEstimator, TransformerMixin):
    """Maps country codes to continents, handling unknowns and missing columns."""
    def __init__(self, country_col='country_x', continent_col='continent_customer', unknown_value='Desconocido'):
        self.country_col = country_col
        self.continent_col = continent_col
        self.unknown_value = unknown_value
        # (Continent dictionary remains the same as before - keeping it concise here)
        self.continentes = {
            'Europa': ['ALB', 'AND', 'AUT', 'BEL', 'BGR', 'BIH', 'BLR', 'CHE', 'CYP', 'CZE', 'DEU', 'DNK', 'ESP', 'EST', 'FIN', 'FRA', 'FRO', 'GBR', 'GEO', 'GGY', 'GIB', 'GRC', 'HRV', 'HUN', 'IMN', 'IRL', 'ISL', 'ITA', 'JEY', 'LIE', 'LTU', 'LUX', 'LVA', 'MCO', 'MDA', 'MKD', 'MLT', 'MNE', 'NLD', 'NOR', 'POL', 'PRT', 'ROU', 'RUS', 'SMR', 'SRB', 'SVK', 'SVN', 'SWE', 'TUR', 'UKR', 'VAT'],
            'Asia': ['ARE', 'ARM', 'AZE', 'BGD', 'BHR', 'BRN', 'BTN', 'CHN', 'CYP', 'GEO', 'HKG', 'IDN', 'IND', 'IRN', 'IRQ', 'ISR', 'JOR', 'JPN', 'KAZ', 'KGZ', 'KHM', 'KOR', 'KWT', 'LAO', 'LBN', 'LKA', 'MAC', 'MDV', 'MMR', 'MNG', 'MYS', 'NPL', 'OMN', 'PAK', 'PHL', 'PRK', 'PSE', 'QAT', 'RUS', 'SAU', 'SGP', 'SYR', 'THA', 'TJK', 'TLS', 'TUR', 'TWN', 'UZB', 'VNM', 'YEM'],
            'África': ['AGO', 'BDI', 'BEN', 'BFA', 'BWA', 'CAF', 'CIV', 'CMR', 'COD', 'COG', 'COM', 'CPV', 'DJI', 'DZA', 'EGY', 'ERI', 'ESH', 'ETH', 'GAB', 'GHA', 'GIN', 'GMB', 'GNB', 'GNQ', 'KEN', 'LBR', 'LBY', 'LSO', 'MAR', 'MDG', 'MLI', 'MOZ', 'MRT', 'MUS', 'MWI', 'MYT', 'NAM', 'NER', 'NGA', 'REU', 'RWA', 'SDN', 'SEN', 'SLE', 'SOM', 'SSD', 'STP', 'SWZ', 'SYC', 'TCD', 'TGO', 'TUN', 'TZA', 'UGA', 'ZAF', 'ZMB', 'ZWE'],
            'América del Norte': ['ABW', 'AIA', 'ANT', 'ATG', 'BES', 'BHS', 'BLM', 'BLZ', 'BMU', 'BRB', 'CAN', 'CRI', 'CUB', 'CUW', 'CYM', 'DMA', 'DOM', 'GLP', 'GRD', 'GRL', 'GTM', 'HND', 'HTI', 'JAM', 'KNA', 'LCA', 'MAF', 'MEX', 'MSR', 'MTQ', 'NIC', 'PAN', 'PRI', 'SLV', 'SPM', 'SXM', 'TCA', 'TTO', 'USA', 'VCT', 'VGB', 'VIR'],
            'América del Sur': ['ARG', 'BOL', 'BRA', 'CHL', 'COL', 'ECU', 'FLK', 'GUF', 'GUY', 'PER', 'PRY', 'SUR', 'URY', 'VEN'],
            'Oceanía': ['ASM', 'AUS', 'COK', 'FJI', 'FSM', 'GUM', 'KIR', 'MHL', 'MNP', 'NCL', 'NFK', 'NIU', 'NRU', 'NZL', 'PLW', 'PNG', 'PYF', 'SLB', 'TKL', 'TON', 'TUV', 'VUT', 'WLF', 'WSM'],
            'Antártida': ['ATA', 'ATF'],
        }
        self.country_to_continent = {country: continent
                                     for continent, countries in self.continentes.items()
                                     for country in countries}

    def fit(self, X, y=None):
        # No fitting needed, but check column presence optimistically
        if self.country_col not in X.columns:
            warnings.warn(f"ContinentMapper Fit: Column '{self.country_col}' not found during fit. Will attempt transform regardless.", RuntimeWarning)
        return self

    def transform(self, X):
        X_copy = X.copy()
        # Create the target column first
        if self.continent_col not in X_copy.columns:
            X_copy[self.continent_col] = self.unknown_value

        if self.country_col in X_copy.columns:
            # Apply mapping, ensuring existing NaNs in country_col result in unknown_value
            X_copy[self.continent_col] = X_copy[self.country_col].apply(
                lambda code: self.country_to_continent.get(str(code).strip().upper(), self.unknown_value) if pd.notna(code) else self.unknown_value
            )
        else:
            # Column missing during transform - already filled with unknown_value
            warnings.warn(f"ContinentMapper Transform: Column '{self.country_col}' not found. Output column '{self.continent_col}' filled with '{self.unknown_value}'.", RuntimeWarning)

        return X_copy

# --- Custom Transformer for Feature Engineering (Enhanced Robustness) ---
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Performs feature engineering, handling missing input columns gracefully."""
    def __init__(self, date_cols=['arrival_date', 'booking_date', 'reservation_status_date'],
                 asset_cols=['pool_and_spa', 'restaurant', 'parking'],
                 country_cols=['country_x', 'country_y'],
                 fillna_zero_cols = ['special_requests', 'required_car_parking_spaces']):
        self.date_cols = date_cols
        self.asset_cols = asset_cols
        self.country_cols = country_cols
        self.fillna_zero_cols = fillna_zero_cols

    def fit(self, X, y=None):
        # No state needed
        return self

    def transform(self, X):
        X_copy = X.copy()

        # 1. num_assets
        present_asset_cols = [col for col in self.asset_cols if col in X_copy.columns]
        X_copy['num_assets'] = 0 # Initialize
        if present_asset_cols:
            for col in present_asset_cols:
                X_copy[col] = pd.to_numeric(X_copy[col], errors='coerce').fillna(0)
            X_copy['num_assets'] = X_copy[present_asset_cols].sum(axis=1).astype(int)
        # else: # Column 'num_assets' already initialized to 0

        # 2. is_foreign
        country_x, country_y = self.country_cols
        X_copy['is_foreign'] = 0 # Initialize
        if country_x in X_copy.columns and country_y in X_copy.columns:
            col_x_str = X_copy[country_x].astype(str).fillna('UNK_X')
            col_y_str = X_copy[country_y].astype(str).fillna('UNK_Y')
            X_copy['is_foreign'] = (col_x_str != col_y_str).astype(int)
        else:
            warnings.warn(f"FeatureEngineer Transform: Columns '{country_x}' or '{country_y}' missing. 'is_foreign' set to 0.", RuntimeWarning)

        # 3. lead_time
        arrival_date_col, booking_date_col, _ = self.date_cols
        X_copy['lead_time'] = np.nan # Initialize
        if arrival_date_col in X_copy.columns and booking_date_col in X_copy.columns:
            arrival_dt = pd.to_datetime(X_copy[arrival_date_col], errors='coerce')
            booking_dt = pd.to_datetime(X_copy[booking_date_col], errors='coerce')
            valid_dates = arrival_dt.notna() & booking_dt.notna()
            X_copy.loc[valid_dates, 'lead_time'] = (arrival_dt[valid_dates] - booking_dt[valid_dates]).dt.days
            X_copy.loc[X_copy['lead_time'] < 0, 'lead_time'] = 0 # Cap negative lead time
            if X_copy['lead_time'].isnull().any():
                warnings.warn("FeatureEngineer Transform: 'lead_time' has NaNs after calculation (missing/invalid dates). Imputer will handle.", RuntimeWarning)
        else:
            warnings.warn(f"FeatureEngineer Transform: Columns '{arrival_date_col}' or '{booking_date_col}' missing. 'lead_time' set to NaN.", RuntimeWarning)


        # 4. Fill specific NaNs with 0
        for col in self.fillna_zero_cols:
            if col in X_copy.columns:
                X_copy[col].fillna(0, inplace=True)
            else:
                # If col is expected downstream, create it filled with 0
                X_copy[col] = 0
                warnings.warn(f"FeatureEngineer Transform: Column '{col}' not found. Created and filled with 0.", RuntimeWarning)


        return X_copy

# --- Target Engineering Function (Allows NaNs) ---
def engineer_target_variable(df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    """
    Engineers the target variable 'cancelled_last_30_days'. Allows NaNs if dates are invalid.
    Returns DataFrame with features and the target Series (may contain NaNs).
    """
    df_eng = df.copy()
    required_cols = ['reservation_status', 'arrival_date', 'reservation_status_date', 'booking_date']

    if not all(col in df_eng.columns for col in required_cols):
        print(f"Error: Target engineering requires columns: {required_cols}. Found: {list(df_eng.columns)}")
        return None, None

    try:
        # Convert dates, coercing errors to NaT
        for col in ['arrival_date', 'reservation_status_date', 'booking_date']:
            df_eng[col] = pd.to_datetime(df_eng[col], errors='coerce')

        # Initialize intermediate columns
        df_eng['status_corrected'] = df_eng['reservation_status']
        df_eng["is_canceled"] = 0
        df_eng["days_diff_cancel"] = np.nan
        df_eng["cancelled_last_30_days"] = np.nan # Start with NaN

        # --- Logic ---
        mask_no_show = df_eng['reservation_status'] == 'No-Show'
        # Date comparison yields NaT if any date is NaT
        mask_same_date = df_eng['reservation_status_date'].dt.normalize() == df_eng['arrival_date'].dt.normalize()
        # Only update status where comparison is valid (not NaT)
        df_eng.loc[mask_no_show & mask_same_date.fillna(False), 'status_corrected'] = 'Check-Out'
        df_eng.loc[mask_no_show & (~mask_same_date).fillna(False), 'status_corrected'] = 'Canceled'

        df_eng["is_canceled"] = (df_eng["status_corrected"] == "Canceled").astype(int)

        # Calculate days difference only where dates are valid
        valid_diff_dates = df_eng['arrival_date'].notna() & df_eng['reservation_status_date'].notna()
        df_eng.loc[valid_diff_dates, "days_diff_cancel"] = (df_eng.loc[valid_diff_dates, "arrival_date"] - df_eng.loc[valid_diff_dates, 'reservation_status_date']).dt.days

        # Create final target where possible
        # Condition: is_canceled is 1 AND days_diff is calculated AND days_diff <= 30
        mask_target_calculable = (df_eng["is_canceled"] == 1) & df_eng["days_diff_cancel"].notna()
        mask_target_condition = mask_target_calculable & (df_eng["days_diff_cancel"] <= 30)

        # Set target to 1 where condition is met
        df_eng.loc[mask_target_condition, "cancelled_last_30_days"] = 1
        # Set target to 0 where calculable but condition not met
        df_eng.loc[mask_target_calculable & (~mask_target_condition), "cancelled_last_30_days"] = 0
        # Target remains NaN where dates were missing for calculation

        # --- Define Target and Features ---
        target_col = "cancelled_last_30_days"
        y = df_eng[target_col]

        # Drop intermediate cols
        cols_to_drop = ['reservation_status_date', 'status_corrected', 'is_canceled', 'days_diff_cancel']
        df_features = df_eng.drop(columns=cols_to_drop, errors='ignore')

        num_nan_target = y.isnull().sum()
        if num_nan_target > 0:
            warnings.warn(f"Target Engineering: {num_nan_target} rows resulted in NaN target due to missing/invalid dates.")

        return df_features, y

    except Exception as e:
        print(f"Error during target engineering: {e}")
        traceback.print_exc()
        return None, None

# --- Outlier Handler (Caps/Floors using IQR) ---
class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Handles outliers by capping/flooring based on IQR boundaries learned from training data.
    """
    def __init__(self, columns=None, lower_quantile=0.05, upper_quantile=0.95, factor=1.5):
        self.columns = columns if columns is not None else ['rate', 'total_guests', 'stay_nights', 'lead_time']
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.factor = factor
        self.boundaries_ = {}
        self.columns_found_ = []

    def fit(self, X, y=None):
        self.boundaries_ = {}
        self.columns_found_ = [col for col in self.columns if col in X.columns and pd.api.types.is_numeric_dtype(X[col])]

        if not self.columns_found_:
            warnings.warn("OutlierHandler Fit: No specified numeric columns found.", RuntimeWarning)
            return self

        X_ = X[self.columns_found_].copy()

        for col in self.columns_found_:
            try:
                Q1 = X_[col].quantile(self.lower_quantile)
                Q3 = X_[col].quantile(self.upper_quantile)
                IQR = Q3 - Q1

                if pd.isna(IQR) or IQR <= 1e-9: # Handle NaN or zero IQR
                    warnings.warn(f"OutlierHandler Fit: IQR for column '{col}' is invalid ({IQR}). Using min/max as bounds.", RuntimeWarning)
                    lower_bound = X_[col].min()
                    upper_bound = X_[col].max()
                    if pd.isna(lower_bound) or pd.isna(upper_bound): # If all NaNs
                        continue # Cannot determine bounds
                else:
                    lower_bound = Q1 - self.factor * IQR
                    upper_bound = Q3 + self.factor * IQR

                self.boundaries_[col] = (lower_bound, upper_bound)
            except Exception as e:
                warnings.warn(f"OutlierHandler Fit: Error calculating bounds for column '{col}': {e}", RuntimeWarning)
        return self

    def transform(self, X):
        """Applies capping/flooring based on fitted boundaries."""
        X_processed = X.copy()
        if not self.boundaries_:
            return X_processed # No boundaries learned

        for col, (lower_bound, upper_bound) in self.boundaries_.items():
            if col in X_processed.columns and pd.api.types.is_numeric_dtype(X_processed[col]):
                # Ensure bounds are not NaN before clipping
                valid_lower = lower_bound if not pd.isna(lower_bound) else None
                valid_upper = upper_bound if not pd.isna(upper_bound) else None
                X_processed[col] = X_processed[col].clip(lower=valid_lower, upper=valid_upper)
            elif col in X_processed.columns:
                # Column exists but isn't numeric (shouldn't happen if fit logic is correct)
                warnings.warn(f"OutlierHandler Transform: Column '{col}' found but is not numeric. Skipping capping.", RuntimeWarning)
            # else: # Column from fit not found in transform - ignore silently for inference robustness

        return X_processed

# --- Main Pipeline Class (Refactored) ---
class HotelBookingPipeline:
    def __init__(self,
                 test_size=0.25,
                 random_state=42,
                 variance_threshold=0.001,
                 outlier_cols=['rate', 'total_guests', 'stay_nights', 'lead_time'],
                 outlier_quantiles=(0.05, 0.95),
                 outlier_factor=1.5,
                 selected_k: Optional[int] = None # Specify fixed K for SelectKBest, or None to omit
                 ):
        self.test_size = test_size
        self.random_state = random_state
        self.variance_threshold_val = variance_threshold
        self.outlier_cols = outlier_cols
        self.outlier_quantiles = outlier_quantiles
        self.outlier_factor = outlier_factor
        self.selected_k = selected_k # Store chosen K

        # Pipeline components (reusable parts)
        self.feature_engineer = FeatureEngineer()
        self.continent_mapper = ContinentMapper()
        self.outlier_handler = OutlierHandler(
            columns=self.outlier_cols,
            lower_quantile=self.outlier_quantiles[0],
            upper_quantile=self.outlier_quantiles[1],
            factor=self.outlier_factor
        )

        # Core pipeline (built during training)
        self.preprocessing_pipeline: Optional[Pipeline] = None # Includes custom steps + ColumnTransformer
        self.full_pipeline: Optional[Pipeline] = None # Includes preprocessing + optional selection + model

        # Training artifacts
        self.trained_model_type: Optional[str] = None
        self.trained_columns: Optional[List[str]] = None # Columns entering ColumnTransformer
        self.preprocessor_transformer: Optional[ColumnTransformer] = None # The fitted ColumnTransformer itself
        self.selected_features_indices_: Optional[np.ndarray] = None
        self.selected_features_names_: Optional[List[str]] = None

        # Logs and Metrics
        self.pipeline_steps_log: List[str] = []
        self.metrics: Dict[str, Dict[str, float]] = {'logistic': {}, 'sgd': {}}
        self.best_params: Dict[str, Dict[str, Any]] = {'logistic': {}, 'sgd': {}} # If GridSearchCV is used

    def _log_step(self, message: str):
        print(f"Pipeline Step: {message}")
        self.pipeline_steps_log.append(message)

    # load_and_merge_data, separate_data_by_status remain the same as before

    def load_and_merge_data(self, bookings_file: str, hotels_file: str) -> Optional[pd.DataFrame]:
        """Loads and merges booking and hotel data."""
        self._log_step(f"Cargando datos desde '{bookings_file}' y '{hotels_file}'...")
        try:
            if not os.path.exists(bookings_file):
                raise FileNotFoundError(f"Bookings file not found: {bookings_file}")
            if not os.path.exists(hotels_file):
                raise FileNotFoundError(f"Hotels file not found: {hotels_file}")

            df_book = pd.read_csv(bookings_file)
            df_hotel = pd.read_csv(hotels_file)

            if 'hotel_id' not in df_book.columns or 'hotel_id' not in df_hotel.columns:
                self._log_step("Error: 'hotel_id' column missing in one or both input files for merging.")
                return None
            df = pd.merge(df_book, df_hotel, on='hotel_id', how='left')
            df.drop("hotel_id", axis=1, inplace=True, errors='ignore') # Drop key after merge
            self._log_step(f"Datos cargados y fusionados. Shape inicial: {df.shape}")
            return df
        except FileNotFoundError as e:
            self._log_step(f"Error: {e}")
            return None
        except Exception as e:
            self._log_step(f"Error inesperado en la carga o fusión de datos: {e}")
            traceback.print_exc()
            return None

    def separate_data_by_status(self, df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Separates data into modeling (Canceled/Check-Out/No-Show) and validation (Booked)."""
        if df is None or df.empty:
            self._log_step("Error: DataFrame de entrada para separate_data_by_status está vacío o es None.")
            return None, None
        if 'reservation_status' not in df.columns:
            self._log_step("Error: 'reservation_status' column not found for data separation.")
            return None, None
        try:
            df['reservation_status'] = df['reservation_status'].astype(str).fillna('Unknown')
            model_statuses = ["Canceled", "Check-Out", "No-Show"]
            df_model = df[df["reservation_status"].isin(model_statuses)].copy()
            df_validation = df[df["reservation_status"] == "Booked"].copy()
            self._log_step(f"Datos separados: {df_model.shape[0]} registros para modelado ({model_statuses}), {df_validation.shape[0]} para validación ('Booked').")

            if df_model.empty:
                self._log_step("Warning: No data available for modeling after filtering by status.")
                return None, df_validation if not df_validation.empty else None

            return df_model, df_validation
        except Exception as e:
            self._log_step(f"Error al separar los datos por estado: {e}")
            traceback.print_exc()
            return None, None


    def build_preprocessing_pipeline(self, X: pd.DataFrame) -> Optional[Pipeline]:
        """
        Builds the preprocessing pipeline including custom steps, outlier handling,
        and standard scaling/encoding. Designed to be fit on training data.
        """
        self._log_step("Construyendo el pipeline de preprocesamiento...")
        if X is None or X.empty:
            self._log_step("Error: DataFrame de entrada para build_preprocessing_pipeline está vacío.")
            return None

        try:
            # --- Determine column types AFTER custom steps but BEFORE ColumnTransformer ---
            # Apply initial custom transformers to a sample or copy to determine columns
            X_temp = self.feature_engineer.transform(X.copy())
            X_temp = self.continent_mapper.transform(X_temp)
            # Outlier handler will be applied *within* the pipeline now
            # X_temp = self.outlier_handler.transform(X_temp) # Don't apply here, just use for column ID

            self.trained_columns = list(X_temp.columns) # Columns entering ColumnTransformer stage

            numeric_cols = X_temp.select_dtypes(include=np.number).columns.tolist()
            categorical_cols = X_temp.select_dtypes(include=['object', 'category']).columns.tolist()

            # Exclude known non-features (defensive)
            cols_to_exclude = ['cancelled_last_30_days', 'id', 'booking_id']
            numeric_cols = [col for col in numeric_cols if col not in cols_to_exclude]
            categorical_cols = [col for col in categorical_cols if col not in cols_to_exclude]

            common_cols = set(numeric_cols) & set(categorical_cols)
            if common_cols:
                self._log_step(f"Warning: Columns found in both numeric/categorical: {common_cols}. Removing from categorical.")
                categorical_cols = [col for col in categorical_cols if col not in common_cols]

            # --- Define ColumnTransformer ---
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())])

            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')), # Handles NaNs and unseen values implicitly if strategy='constant' fill_value='missing'
                ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))]) # handle_unknown='ignore' is crucial for inference

            transformers_list = []
            if numeric_cols:
                transformers_list.append(('num', numeric_transformer, numeric_cols))
            if categorical_cols:
                transformers_list.append(('cat', categorical_transformer, categorical_cols))

            if not transformers_list:
                self._log_step("Error: No numeric or categorical columns found for ColumnTransformer.")
                return None

            self.preprocessor_transformer = ColumnTransformer(
                transformers=transformers_list,
                remainder='drop' # Ensures only processed columns pass through
            )
            # Store the columns targeted by the ColumnTransformer for clarity
            self._preprocessor_numeric_cols = numeric_cols
            self._preprocessor_categorical_cols = categorical_cols


            # --- Build the full PREPROCESSING Pipeline ---
            # Order: Custom FE -> Continent Map -> Outlier Cap -> ColumnTransformer
            preprocessing_pipeline = Pipeline([
                ('feature_engineer', self.feature_engineer),
                ('continent_mapper', self.continent_mapper),
                ('outlier_handler', self.outlier_handler), # Fit/Transform applied here
                ('column_transformer', self.preprocessor_transformer)
            ])

            self._log_step("Pipeline de preprocesamiento construido exitosamente.")
            return preprocessing_pipeline

        except Exception as e:
            self._log_step(f"Error construyendo el pipeline de preprocesamiento: {e}")
            traceback.print_exc()
            return None


    def build_full_model_pipeline(self, model_type: str = 'logistic') -> Optional[Pipeline]:
        """
        Builds the full model pipeline including preprocessing, variance threshold,
        optional feature selection (fixed k), and the classifier.
        Requires self.preprocessing_pipeline to be built and fitted first.
        """
        self._log_step(f"Construyendo el pipeline completo del modelo para: {model_type.upper()}")

        if self.preprocessing_pipeline is None or not hasattr(self.preprocessing_pipeline, 'steps'):
            self._log_step("Error: El pipeline de preprocesamiento base no está construido o es inválido.")
            return None
        # Check if outlier_handler has been fitted (needed for transform)
        if not hasattr(self.outlier_handler, 'boundaries_') or not self.outlier_handler.boundaries_:
            self._log_step("Error: OutlierHandler no ha sido 'fiteado' antes de construir el pipeline completo.")
            # return None # This check might be too strict if called before fitting the whole pipeline

        # --- Define Classifier ---
        if model_type == 'logistic':
            classifier = LogisticRegression(max_iter=2000, random_state=self.random_state, class_weight='balanced', solver='liblinear')
        elif model_type == 'sgd':
            classifier = SGDClassifier(max_iter=2000, tol=1e-3, random_state=self.random_state, class_weight='balanced', loss='log_loss')
        else:
            self._log_step(f"Error: Modelo '{model_type}' no soportado.")
            raise ValueError("Tipo de modelo no soportado")

        # --- Assemble Steps ---
        model_steps = [
            # The preprocessing_pipeline is now a single fitted step
            ('preprocessing', self.preprocessing_pipeline),
            ('variance_threshold', VarianceThreshold(threshold=self.variance_threshold_val))
            # SMOTE REMOVED
        ]

        # Add Feature Selection if k is specified
        if self.selected_k is not None and isinstance(self.selected_k, int) and self.selected_k > 0:
            self._log_step(f"Añadiendo SelectKBest con k={self.selected_k} al pipeline.")
            model_steps.append(('feature_selector', SelectKBest(score_func=f_classif, k=self.selected_k)))
        else:
            self._log_step("SelectKBest no se incluirá en el pipeline final (k no especificado o inválido).")


        model_steps.append(('classifier', classifier))

        try:
            full_pipeline = Pipeline(model_steps) # Use sklearn Pipeline
            self._log_step(f"Pipeline completo del modelo para {model_type.upper()} construido.")
            return full_pipeline
        except Exception as e:
            self._log_step(f"Error construyendo el Pipeline completo del modelo: {e}")
            traceback.print_exc()
            return None


    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series, model_type: str = 'logistic', use_grid_search: bool = False, cv_folds_gridsearch: int = 5) -> bool:
        """
        Trains the model. Fits the preprocessing pipeline first, then builds and fits
        the full model pipeline. Optionally uses GridSearchCV for classifier hyperparameters.

        Args:
            X_train, y_train: Training data.
            model_type: 'logistic' or 'sgd'.
            use_grid_search: If True, tune classifier hyperparameters using GridSearchCV.
                             If False, train with default hyperparameters.
            cv_folds_gridsearch: Folds for GridSearchCV if used.

        Returns:
            True if successful, False otherwise.
        """
        self._log_step(f"Iniciando entrenamiento para el modelo: {model_type.upper()} (GridSearch={use_grid_search})")

        if X_train is None or y_train is None or X_train.empty or y_train.empty:
            self._log_step("Error: Datos de entrenamiento (X_train o y_train) vacíos o None.")
            return False

        # 1. Build and Fit the Preprocessing Pipeline
        # Includes fitting OutlierHandler now
        self.preprocessing_pipeline = self.build_preprocessing_pipeline(X_train)
        if self.preprocessing_pipeline is None:
            self._log_step("Fallo al construir el pipeline de preprocesamiento. Abortando.")
            return False
        try:
            self._log_step("Ajustando (fit) el pipeline de preprocesamiento...")
            # Fit the preprocessing pipeline (including FE, mapper, outlier handler, column transformer)
            self.preprocessing_pipeline.fit(X_train, y_train)
            self._log_step("Pipeline de preprocesamiento ajustado.")
            # Store the fitted ColumnTransformer instance separately if needed later
            self.preprocessor_transformer = self.preprocessing_pipeline.named_steps['column_transformer']

        except Exception as e:
            self._log_step(f"Error ajustando el pipeline de preprocesamiento: {e}")
            traceback.print_exc()
            return False


        # 2. Build the Full Model Pipeline (using the fitted preprocessing pipeline)
        self.full_pipeline = self.build_full_model_pipeline(model_type)
        if self.full_pipeline is None:
            self._log_step("Fallo al construir el pipeline completo del modelo. Abortando.")
            return False

        # 3. Fit the Full Model Pipeline
        try:
            self.trained_model_type = model_type
            if use_grid_search:
                self._log_step(f"Iniciando GridSearchCV para hiperparámetros del clasificador ({model_type.upper()}) con {cv_folds_gridsearch} folds...")

                # --- Define Param Grid (Classifier Only) ---
                param_grid = {}
                if model_type == 'logistic':
                    param_grid = {
                        'classifier__C': [0.01, 0.1, 1, 10, 50],
                        'classifier__penalty': ['l1', 'l2']
                    }
                elif model_type == 'sgd':
                    param_grid = {
                        'classifier__alpha': [1e-5, 1e-4, 1e-3, 1e-2],
                        'classifier__penalty': ['l2', 'elasticnet'],
                        'classifier__l1_ratio': [0.1, 0.15, 0.5, 0.75]
                    }

                cv = StratifiedKFold(n_splits=cv_folds_gridsearch, shuffle=True, random_state=self.random_state)
                scorer = make_scorer(f1_score, average='binary')

                # Important: Pass the *unfitted* full pipeline definition to GridSearchCV
                # It will handle cloning and fitting internally during CV
                grid_search = GridSearchCV(
                    self.full_pipeline, # Pass the pipeline structure
                    param_grid,
                    cv=cv,
                    scoring=scorer,
                    verbose=1,
                    n_jobs=-1,
                    error_score='raise'
                )

                grid_search.fit(X_train, y_train) # Fit GridSearchCV

                self.full_pipeline = grid_search.best_estimator_ # Store the best found pipeline
                self.best_params[model_type] = grid_search.best_params_
                best_score = grid_search.best_score_

                self._log_step(f"GridSearchCV completado. Mejor F1 (CV): {best_score:.4f}")
                self._log_step(f"Mejores parámetros: {self.best_params[model_type]}")

            else:
                # Fit without GridSearchCV
                self._log_step("Ajustando (fit) el pipeline completo del modelo sin GridSearchCV...")
                self.full_pipeline.fit(X_train, y_train)
                self._log_step("Pipeline completo del modelo ajustado.")

            # --- Post-Fit: Extract Feature Selection Info (if used) ---
            if 'feature_selector' in self.full_pipeline.named_steps:
                try:
                    selector = self.full_pipeline.named_steps['feature_selector']
                    self.selected_features_indices_ = selector.get_support(indices=True)

                    # Get feature names *after* all preprocessing steps *before* selector
                    preprocessing_step = self.full_pipeline.named_steps['preprocessing']
                    variance_step = self.full_pipeline.named_steps['variance_threshold']
                    column_transformer_step = preprocessing_step.named_steps['column_transformer']

                    feature_names_after_col_transformer = column_transformer_step.get_feature_names_out()
                    variance_mask = variance_step.get_support()
                    feature_names_before_selector = feature_names_after_col_transformer[variance_mask]

                    self.selected_features_names_ = [feature_names_before_selector[i] for i in self.selected_features_indices_]
                    self._log_step(f"SelectKBest seleccionó {len(self.selected_features_names_)} características.")
                except Exception as e:
                    self._log_step(f"Warning: No se pudieron extraer los nombres de las características seleccionadas: {e}")
                    self.selected_features_indices_ = None
                    self.selected_features_names_ = None

            self._log_step(f"Modelo {model_type.upper()} entrenado exitosamente.")
            return True

        except Exception as e:
            self._log_step(f"Error durante el ajuste del pipeline completo del modelo: {e}")
            traceback.print_exc()
            return False

    def evaluate_feature_selection_impact(self, X_train: pd.DataFrame, y_train: pd.Series, model_type: str = 'logistic', cv_folds: int = 5, max_features_to_eval: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        Evaluates model performance for different numbers of selected features (k).
        Uses cross-validation on the training data.

        Args:
            X_train, y_train: Training data.
            model_type: 'logistic' or 'sgd'.
            cv_folds: Number of folds for cross-validation.
            max_features_to_eval: Maximum value of k to test. If None, tests up to all features.

        Returns:
            DataFrame with k values and corresponding mean CV F1 scores, or None if failed.
        """
        self._log_step(f"Iniciando evaluación del impacto de la selección de características (k) para modelo {model_type.upper()}...")

        if X_train is None or y_train is None or X_train.empty or y_train.empty:
            self._log_step("Error Evaluación K: Datos de entrenamiento vacíos.")
            return None

        # 1. Ensure Preprocessing Pipeline is Fitted
        if self.preprocessing_pipeline is None or not hasattr(self.preprocessing_pipeline, 'steps') or self.preprocessor_transformer is None:
            self._log_step("Error Evaluación K: Pipeline de preprocesamiento no construido o ajustado. Ajustando ahora...")
            temp_preprocessing_pipeline = self.build_preprocessing_pipeline(X_train)
            if temp_preprocessing_pipeline is None: return None
            try:
                temp_preprocessing_pipeline.fit(X_train, y_train)
                fitted_preprocessing_pipeline = temp_preprocessing_pipeline
                fitted_col_transformer = fitted_preprocessing_pipeline.named_steps['column_transformer']
            except Exception as e:
                self._log_step(f"Error ajustando preprocesamiento temporal para evaluación k: {e}")
                return None
        else:
            # Use the already fitted pipeline from the instance
            fitted_preprocessing_pipeline = self.preprocessing_pipeline
            fitted_col_transformer = self.preprocessor_transformer


        # 2. Determine the maximum number of features available AFTER preprocessing
        try:
            # Need to apply preprocessing and variance threshold to determine feature count
            temp_pipeline_pre_select = Pipeline([
                ('preprocessing', fitted_preprocessing_pipeline),
                ('variance_threshold', VarianceThreshold(threshold=self.variance_threshold_val))
            ])
            X_processed = temp_pipeline_pre_select.transform(X_train)
            n_features_available = X_processed.shape[1]
            self._log_step(f"Evaluación K: {n_features_available} características disponibles después del preprocesamiento y VarianceThreshold.")
            if n_features_available == 0:
                self._log_step("Error Evaluación K: No hay características disponibles después del preprocesamiento.")
                return None
        except Exception as e:
            self._log_step(f"Error Evaluación K: No se pudo determinar el número de características disponibles: {e}")
            traceback.print_exc()
            return None

        # 3. Define Classifier
        if model_type == 'logistic':
            classifier = LogisticRegression(max_iter=2000, random_state=self.random_state, class_weight='balanced', solver='liblinear')
        elif model_type == 'sgd':
            classifier = SGDClassifier(max_iter=2000, tol=1e-3, random_state=self.random_state, class_weight='balanced', loss='log_loss')
        else:
            self._log_step(f"Error Evaluación K: Modelo '{model_type}' no soportado.")
            return None


        # 4. Loop through k values and evaluate
        results = []
        k_values = range(1, n_features_available + 1)
        if max_features_to_eval is not None:
            k_values = range(1, min(n_features_available, max_features_to_eval) + 1)

        self._log_step(f"Evaluando k desde 1 hasta {max(k_values)} con {cv_folds} folds...")

        for k in k_values:
            # Create pipeline for this specific k
            current_pipeline = Pipeline([
                ('preprocessing', fitted_preprocessing_pipeline), # Use already fitted preprocessor
                ('variance_threshold', VarianceThreshold(threshold=self.variance_threshold_val)),
                ('feature_selector', SelectKBest(score_func=f_classif, k=k)),
                ('classifier', classifier) # Use a default classifier instance
            ])

            try:
                # Perform cross-validation
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
                scorer = make_scorer(f1_score, average='binary')
                # Important: cross_val_score needs X, y *before* preprocessing,
                # as the pipeline handles preprocessing internally.
                scores = cross_val_score(current_pipeline, X_train, y_train, cv=cv, scoring=scorer, n_jobs=-1)
                mean_f1 = np.mean(scores)
                std_f1 = np.std(scores)
                results.append({'k': k, 'mean_f1_cv': mean_f1, 'std_f1_cv': std_f1})
                print(f"  k={k}: Mean F1 = {mean_f1:.4f} (+/- {std_f1:.4f})") # Progress indicator
            except ValueError as ve:
                if "less than k" in str(ve) or "Cannot have number of splits" in str(ve):
                    print(f"  k={k}: Error en CV (probablemente datos insuficientes para k={k} o {cv_folds} splits). Deteniendo evaluación.")
                    break # Stop if CV fails for a k
                else:
                    print(f"  k={k}: Error de valor inesperado durante CV: {ve}")
                    break
            except Exception as e:
                print(f"  k={k}: Error inesperado durante CV: {e}")
                traceback.print_exc()
                break # Stop on other errors


        if not results:
            self._log_step("Evaluación K: No se obtuvieron resultados.")
            return None

        results_df = pd.DataFrame(results)
        self._log_step("Evaluación del impacto de la selección de características completada.")

        # Optional: Plot results
        try:
            plt.figure(figsize=(10, 6))
            plt.errorbar(results_df['k'], results_df['mean_f1_cv'], yerr=results_df['std_f1_cv'], marker='o', capsize=5, linestyle='-', alpha=0.8)
            plt.xlabel('Número de Características Seleccionadas (k)')
            plt.ylabel('F1 Score Promedio (CV)')
            plt.title(f'Impacto de k en F1 Score (CV={cv_folds}) - Modelo {model_type.upper()}')
            plt.grid(True)
            best_k_idx = results_df['mean_f1_cv'].idxmax()
            best_k = results_df.loc[best_k_idx, 'k']
            best_f1 = results_df.loc[best_k_idx, 'mean_f1_cv']
            plt.axvline(best_k, color='r', linestyle='--', label=f'Mejor k={best_k} (F1={best_f1:.4f})')
            plt.legend()
            plt.show()
        except Exception as plot_err:
            self._log_step(f"Warning: No se pudo generar el gráfico de evaluación de k: {plot_err}")


        return results_df

    # evaluate_model remains mostly the same, but uses self.full_pipeline
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series, plot_curves=True) -> Optional[Dict[str, float]]:
        """Evaluates the final trained model (self.full_pipeline) on the test set."""
        model_type = self.trained_model_type
        pipeline_to_eval = self.full_pipeline

        if model_type is None or pipeline_to_eval is None:
            self._log_step("Error de evaluación: El modelo final no está entrenado o no es válido.")
            return None
        if X_test is None or y_test is None or X_test.empty or y_test.empty:
            self._log_step("Error de evaluación: Datos de prueba vacíos o None.")
            return None

        self._log_step(f"Evaluando el modelo final entrenado ({model_type.upper()}) en el conjunto de prueba...")

        try:
            y_pred = pipeline_to_eval.predict(X_test)
            y_pred_proba = None
            if hasattr(pipeline_to_eval.named_steps['classifier'], 'predict_proba'):
                y_pred_proba = pipeline_to_eval.predict_proba(X_test)[:, 1]
            elif hasattr(pipeline_to_eval.named_steps['classifier'], 'decision_function'):
                y_scores = pipeline_to_eval.decision_function(X_test)
                y_pred_proba = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min() + 1e-9) # Normalize safely

            # --- Calculate Metrics ---
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            metrics_dict = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1}
            roc_auc, pr_auc = None, None

            if y_pred_proba is not None:
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                roc_auc = auc(fpr, tpr)
                metrics_dict['roc_auc'] = roc_auc
                precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
                pr_auc = auc(recall_curve, precision_curve)
                metrics_dict['pr_auc'] = pr_auc

            # --- Logging & Plotting (same as before) ---
            self._log_step("\n--- Métricas de Evaluación en Conjunto de Prueba ---")
            self._log_step(f"  Modelo: {model_type.upper()}")
            self._log_step(f"  Accuracy:  {accuracy:.4f}")
            # ... (rest of logging)
            cm = confusion_matrix(y_test, y_pred)
            self._log_step("Matriz de Confusión:")
            self._log_step(f"\n{cm}\n")
            report = classification_report(y_test, y_pred, target_names=['No Cancelado (0)', 'Cancelado (1)'], zero_division=0)
            self._log_step("Informe de Clasificación:")
            self._log_step(f"\n{report}\n")

            if plot_curves:
                # Confusion Matrix Plot
                plt.figure(figsize=(6, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Pred No Cancela', 'Pred Si Cancela'], yticklabels=['Real No Cancela', 'Real Si Cancela'])
                plt.xlabel('Predicción')
                plt.ylabel('Valor Real')
                plt.title(f'Matriz de Confusión - {model_type.upper()}')
                plt.show()

                if y_pred_proba is not None:
                    # ROC and PR Curve Plots (same code as before)
                    plt.figure(figsize=(14, 6))
                    # ROC
                    plt.subplot(1, 2, 1); plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})'); plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--'); plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05]); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f'Curva ROC - {model_type.upper()}'); plt.legend(loc="lower right"); plt.grid(True)
                    # PR
                    plt.subplot(1, 2, 2); plt.plot(recall_curve, precision_curve, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})'); no_skill = y_test.sum() / len(y_test); plt.plot([0, 1], [no_skill, no_skill], linestyle='--', color='red', label=f'No Skill ({no_skill:.2f})'); plt.xlabel('Recall'); plt.ylabel('Precision'); plt.ylim([0.0, 1.05]); plt.xlim([0.0, 1.0]); plt.title(f'Curva Precision-Recall - {model_type.upper()}'); plt.legend(loc="best"); plt.grid(True)
                    plt.tight_layout(); plt.show()

            self.metrics[model_type] = metrics_dict
            return metrics_dict

        except Exception as e:
            self._log_step(f"Error durante la evaluación del modelo {model_type.upper()}: {e}")
            traceback.print_exc()
            return None


    # predict and predict_proba use self.full_pipeline
    def predict(self, X_new: pd.DataFrame) -> Optional[np.ndarray]:
        """Makes predictions using the final trained pipeline."""
        pipeline_to_use = self.full_pipeline
        model_type = self.trained_model_type

        if pipeline_to_use is None or model_type is None:
            self._log_step("Error Predicción: Modelo final no entrenado.")
            return None
        if X_new is None or X_new.empty:
            self._log_step("Error Predicción: Datos de entrada vacíos.")
            return None

        self._log_step(f"Realizando predicciones con el modelo final entrenado ({model_type.upper()})...")
        try:
            # Pipeline handles missing columns robustness via transformers/imputers
            predictions = pipeline_to_use.predict(X_new)
            self._log_step(f"Predicciones realizadas para {len(X_new)} registros.")
            return predictions
        except Exception as e:
            self._log_step(f"Error durante la predicción: {e}")
            traceback.print_exc()
            return None

    def predict_proba(self, X_new: pd.DataFrame) -> Optional[np.ndarray]:
        """Makes probability predictions using the final trained pipeline."""
        pipeline_to_use = self.full_pipeline
        model_type = self.trained_model_type

        if pipeline_to_use is None or model_type is None:
            self._log_step("Error Predicción Prob: Modelo final no entrenado.")
            return None
        if X_new is None or X_new.empty:
            self._log_step("Error Predicción Prob: Datos de entrada vacíos.")
            return None

        try:
            classifier_step = pipeline_to_use.named_steps['classifier']
            if hasattr(classifier_step, 'predict_proba'):
                probabilities = pipeline_to_use.predict_proba(X_new)
                prob_positive_class = probabilities[:, 1]
                self._log_step(f"Predicciones de probabilidad realizadas para {len(X_new)} registros.")
                return prob_positive_class
            else:
                self._log_step(f"Error Predicción Prob: El clasificador final ({type(classifier_step).__name__}) no soporta 'predict_proba'.")
                return None
        except Exception as e:
            self._log_step(f"Error durante la predicción de probabilidad: {e}")
            traceback.print_exc()
            return None

    # save_pipeline and load_pipeline remain the same (using cloudpickle)
    def save_pipeline(self, filepath: str) -> bool:
        """Saves the entire HotelBookingPipeline object using cloudpickle."""
        self._log_step(f"Guardando el objeto pipeline completo en: {filepath}")
        if self.full_pipeline is None:
            warnings.warn("Save Warning: Intentando guardar, pero el pipeline final no parece estar entrenado.", RuntimeWarning)

        try:
            dir_name = os.path.dirname(filepath)
            if dir_name: os.makedirs(dir_name, exist_ok=True)
            with open(filepath, 'wb') as f:
                cloudpickle.dump(self, f)
            self._log_step(f"Pipeline guardado exitosamente en {filepath}")
            return True
        except Exception as e:
            self._log_step(f"Error al guardar el pipeline en {filepath}: {e}")
            traceback.print_exc()
            return False

    @staticmethod
    def load_pipeline(filepath: str) -> Optional['HotelBookingPipeline']:
        """Loads a HotelBookingPipeline object from a file."""
        print(f"Pipeline Static Step: Cargando el objeto pipeline desde: {filepath}")
        if not os.path.exists(filepath):
            print(f"Error: Archivo de pipeline no encontrado en {filepath}")
            return None
        try:
            with open(filepath, 'rb') as f:
                loaded_pipeline_object = cloudpickle.load(f)
            if isinstance(loaded_pipeline_object, HotelBookingPipeline):
                print("Pipeline cargado exitosamente.")
                return loaded_pipeline_object
            else:
                print(f"Error: El archivo cargado no es una instancia de HotelBookingPipeline. Tipo: {type(loaded_pipeline_object)}")
                return None
        except Exception as e:
            print(f"Error al cargar el pipeline desde {filepath}: {e}")
            traceback.print_exc()
            return None


# === Main Execution Example (Modified Flow) ===
if __name__ == "__main__":

    # --- Configuration ---
    BOOKINGS_FILE = 'bookings.csv'
    HOTELS_FILE = 'hotels.csv'
    PIPELINE_SAVE_PATH = 'hotel_pipeline_v2.pkl'
    MODEL_TO_TRAIN = 'logistic' # 'logistic' or 'sgd'
    PERFORM_K_EVALUATION = True # Set to True to run the feature selection impact evaluation
    MAX_K_TO_EVAL = 30 # Limit k evaluation for speed (set to None for all)
    # --- CHOOSE K_BEST based on evaluation results or domain knowledge ---
    # Example: After running evaluation, you might find k=20 is optimal
    CHOSEN_K_BEST = 20 # Set to None to skip SelectKBest in the final model
    USE_GRIDSEARCH_FINAL = False # Set True to tune final classifier hyperparameters

    # --- Data File Check (same as before) ---
    if not os.path.exists(BOOKINGS_FILE) or not os.path.exists(HOTELS_FILE):
        print("\n---! Archivos de Datos No Encontrados !---")
        # (Code to create dummy files remains the same)
        pd.DataFrame({'hotel_id': [1]*50, 'col_b': range(50), 'reservation_status': ['Canceled', 'Check-Out']*25,
                      'arrival_date': pd.to_datetime(['2024-01-15', '2024-02-10']*25),
                      'booking_date': pd.to_datetime(['2024-01-01', '2024-01-10']*25),
                      'reservation_status_date': pd.to_datetime(['2024-01-10', '2024-02-12']*25),
                      'country_x': ['ESP', 'PRT'] * 25, 'country_y': ['PRT', 'PRT'] * 25,
                      'rate': np.random.randint(50, 300, 50), 'total_guests': np.random.randint(1, 5, 50),
                      'stay_nights': np.random.randint(1, 10, 50), 'special_requests': np.random.randint(0, 3, 50),
                      'required_car_parking_spaces': [0, 1] * 25, 'some_other_col': ['A'] * 50 }
                     ).to_csv(BOOKINGS_FILE, index=False)
        pd.DataFrame({'hotel_id': [1], 'pool_and_spa': [1], 'restaurant': [0], 'parking': [1]}
                     ).to_csv(HOTELS_FILE, index=False)
        print("Archivos CSV de ejemplo creados.")
        print("--------------------------------------------\n")


    # --- 1. Instantiate Pipeline Manager ---
    # Pass the chosen K for the final model training here
    pipeline_manager = HotelBookingPipeline(
        test_size=0.25,
        random_state=42,
        selected_k=CHOSEN_K_BEST # Pass the chosen k here
    )

    # --- 2. Load and Prepare Data ---
    df_full = pipeline_manager.load_and_merge_data(BOOKINGS_FILE, HOTELS_FILE)

    if df_full is not None:
        df_model_data, df_validation_booked = pipeline_manager.separate_data_by_status(df_full)

        if df_model_data is not None:
            # --- 3. Engineer Target Variable (Allows NaNs) ---
            df_features, y_target_with_nans = engineer_target_variable(df_model_data)

            if df_features is not None and y_target_with_nans is not None:
                # --- 4. Handle Target NaNs ---
                valid_target_mask = y_target_with_nans.notna()
                if not valid_target_mask.all():
                    original_count = len(df_features)
                    df_features = df_features[valid_target_mask]
                    y_target = y_target_with_nans[valid_target_mask].astype(int) # Convert to int after removing NaNs
                    removed_count = original_count - len(df_features)
                    pipeline_manager._log_step(f"Manejo de Target NaN: Eliminados {removed_count} registros con target NaN.")
                else:
                    y_target = y_target_with_nans.astype(int)

                if df_features.empty:
                    print("Error: No quedan datos después de eliminar registros con target NaN.")
                    sys.exit()

                # --- 5. Train/Test Split ---
                pipeline_manager._log_step(f"Dividiendo datos ({df_features.shape[0]} registros) en entrenamiento/prueba...")
                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        df_features, y_target,
                        test_size=pipeline_manager.test_size,
                        random_state=pipeline_manager.random_state,
                        stratify=y_target
                    )
                    pipeline_manager._log_step(f"División completada: Entrenamiento={X_train.shape}, Prueba={X_test.shape}")
                    pipeline_manager._log_step(f"Distribución del target en Entrenamiento:\n{y_train.value_counts(normalize=True)}")
                    pipeline_manager._log_step(f"Distribución del target en Prueba:\n{y_test.value_counts(normalize=True)}")


                    # --- 6. (Optional) Evaluate Feature Selection Impact ---
                    if PERFORM_K_EVALUATION:
                        k_eval_results = pipeline_manager.evaluate_feature_selection_impact(
                            X_train.copy(), y_train.copy(), # Use copies to be safe
                            model_type=MODEL_TO_TRAIN,
                            cv_folds=5, # Use 5 folds for evaluation
                            max_features_to_eval=MAX_K_TO_EVAL
                        )
                        if k_eval_results is not None:
                            print("\n--- Resultados Evaluación Impacto K ---")
                            print(k_eval_results)
                            print("------------------------------------")
                            # Based on these results, you might update CHOSEN_K_BEST
                            # and potentially reinstantiate pipeline_manager if k changed.
                            # For this example, we assume CHOSEN_K_BEST is set beforehand.


                    # --- 7. Train the Final Model ---
                    # Note: pipeline_manager was initialized with selected_k=CHOSEN_K_BEST
                    training_success = pipeline_manager.train_model(
                        X_train, y_train,
                        model_type=MODEL_TO_TRAIN,
                        use_grid_search=USE_GRIDSEARCH_FINAL # Decide if tuning final classifier params
                    )

                    # --- 8. Evaluate the Final Model ---
                    if training_success:
                        metrics = pipeline_manager.evaluate_model(X_test, y_test, plot_curves=True)
                        if metrics: print(f"\nEvaluación final del modelo {MODEL_TO_TRAIN.upper()} completada.")

                        # --- 9. Save the Pipeline ---
                        save_success = pipeline_manager.save_pipeline(PIPELINE_SAVE_PATH)
                        if save_success: print(f"\nPipeline entrenado guardado en: {PIPELINE_SAVE_PATH}")

                        # --- 10. Example Load and Predict (using inference script logic) ---
                        print("\n--- Ejemplo de Carga y Predicción ---")
                        loaded_manager = HotelBookingPipeline.load_pipeline(PIPELINE_SAVE_PATH)
                        if loaded_manager:
                            # Predict on test set
                            preds_test = loaded_manager.predict(X_test)
                            if preds_test is not None: print(f"Predicciones (cargado) en Test set: {len(preds_test)} OK")

                            # Predict on validation ('Booked') set
                            if df_validation_booked is not None and not df_validation_booked.empty:
                                print(f"Predicciones (cargado) en 'Booked' set: {len(df_validation_booked)}...")
                                preds_booked = loaded_manager.predict(df_validation_booked)
                                if preds_booked is not None:
                                    print("  Predicciones 'Booked' OK")
                                    # print(pd.Series(preds_booked).value_counts())
                            else:
                                print("No hay datos 'Booked' para predecir.")
                        else:
                            print("Fallo al cargar el pipeline guardado.")
                    else: print("Fallo al guardar el pipeline.")
                else: print(f"\nEntrenamiento del modelo final ({MODEL_TO_TRAIN.upper()}) falló.")

            except ValueError as e:
            pipeline_manager._log_step(f"Error en train_test_split o flujo principal: {e}")
            traceback.print_exc()
        except Exception as e:
        pipeline_manager._log_step(f"Error inesperado en el flujo principal: {e}")
        traceback.print_exc()
else: print("Fallo en la ingeniería de la variable objetivo.")
else: print("No hay datos para modelar después de filtrar por estado.")
else: print("Fallo al cargar/fusionar los datos.")
print("\n--- Fin del Script Principal ---")