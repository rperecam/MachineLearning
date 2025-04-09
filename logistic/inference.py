import pandas as pd
import joblib
import os
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# Definir las clases necesarias para la deserialización

class ContinentMapper(BaseEstimator, TransformerMixin):
    def __init__(self, country_col='country_x', continent_col='continent_customer', unknown='Desconocido'):
        self.country_col = country_col
        self.continent_col = continent_col
        self.unknown = unknown
        self.mapping = {
            'SPA': 'Europa', 'FRA': 'Europa', 'POR': 'Europa',
            'USA': 'América del Norte', 'MEX': 'América del Norte',
            'BRA': 'América del Sur', 'ARG': 'América del Sur',
            'CHN': 'Asia', 'JPN': 'Asia',
            'AUS': 'Oceanía', 'NZL': 'Oceanía'
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

        # Calcular lead_time
        arrival_col = self.date_cols[0]
        booking_col = self.date_cols[1]
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
                X_copy['lead_time'] = np.nan
        else:
            X_copy['lead_time'] = np.nan

        # Calcular num_assets
        present_assets = [col for col in self.asset_cols if col in X_copy.columns]
        if present_assets:
            X_copy['num_assets'] = X_copy[present_assets].fillna(0).astype(int).sum(axis=1)
        else:
            X_copy['num_assets'] = 0

        # Calcular is_foreign
        country_col_x = self.country_cols[0]
        country_col_y = self.country_cols[1]
        if country_col_x in X_copy.columns and country_col_y in X_copy.columns:
            X_copy['is_foreign'] = (X_copy[country_col_x].astype(str) != X_copy[country_col_y].astype(str)).astype(int)
            X_copy.loc[X_copy[country_col_x].isna() | X_copy[country_col_y].isna(), 'is_foreign'] = 0
        else:
            X_copy['is_foreign'] = 0

        return X_copy

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
                X_copy[col] = np.where(X_copy[col] < self.lower_bounds_[col],
                                       self.lower_bounds_[col], X_copy[col])
                X_copy[col] = np.where(X_copy[col] > self.upper_bounds_[col],
                                       self.upper_bounds_[col], X_copy[col])
        return X_copy

# Definir las rutas de los archivos
#model_path = 'model/model.pkl'
#input_bookings_path = 'data/bookings_train.csv'
#input_hotels_path = 'data/hotels.csv'
#output_predictions_path = 'data/predictions.csv'


# Rutas desde variables de entorno
model_path = os.environ.get('MODEL_PATH')
input_bookings_path = os.environ.get('INPUT_BOOKINGS_PATH')
input_hotels_path = os.environ.get('INPUT_HOTELS_PATH')
output_predictions_path = os.environ.get('OUTPUT_PREDICTIONS_PATH')

def load_model(filepath):
    try:
        model = joblib.load(filepath)
        print(f"Modelo cargado desde: {filepath}")
        return model
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return None

def make_predictions(model, input_bookings_path, input_hotels_path, output_predictions_path):
    try:
        df_bookings = pd.read_csv(input_bookings_path)
        df_hotels = pd.read_csv(input_hotels_path)
        df_input = pd.merge(df_bookings, df_hotels, on='hotel_id', how='left')
        df_input = df_input[~df_input['reservation_status'].isin(['Booked', np.nan])].copy()

        predictions = model.predict(df_input)
        probabilities = model.predict_proba(df_input)[:, 1]

        df_output = pd.DataFrame({
            'predicted_cancellation': predictions,
            'probability_cancellation': probabilities
        })
        df_output.to_csv(output_predictions_path, index=False)
        print(f"Predicciones guardadas en: {output_predictions_path}")
    except Exception as e:
        print(f"Error durante la predicción: {e}")

if __name__ == '__main__':
    model = load_model(model_path)
    if model:
        make_predictions(model, input_bookings_path, input_hotels_path, output_predictions_path)