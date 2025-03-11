# Preprocesamiento y División de Datos para Modelado Predictivo

Este documento describe el proceso de fusión, división y preprocesamiento de datos utilizado para preparar un conjunto de datos para el entrenamiento de modelos de aprendizaje automático.

## 1. Fusión de Datos

### Descripción General

Se combinan dos conjuntos de datos, `bookings_train.csv` y `hotels.csv`, utilizando la columna `hotel_id` como clave de unión. El resultado se guarda en `data.csv`.

### Código Python

```python
import pandas as pd

# Cargar datos
bookings_df = pd.read_csv('bookings_train.csv')
hotels_df = pd.read_csv('hotels.csv')

# Fusionar DataFrames
data_df = pd.merge(bookings_df, hotels_df, on='hotel_id')

# Guardar DataFrame fusionado
data_df.to_csv('data.csv', index=False)
```

## 2. División de Datos en Entrenamiento y Validación

### Descripción General

El conjunto de datos fusionado se divide en conjuntos de entrenamiento y validación basados en el estado de la reserva y una fecha límite. El conjunto de validación se guarda en `validation.csv`.

### Código Python

```python
import pandas as pd

def split_data(df, date_limit):
    """Divide el DataFrame en conjuntos de entrenamiento y validación."""

    # División inicial
    validation_df = df[df['status'] == 'Booked'].copy()
    train_df = df[df['status'] != 'Booked'].copy()

    # Filtrado por fecha
    validation_df = validation_df[validation_df['arrival'] >= date_limit]
    train_df = train_df[train_df['arrival'] < date_limit]

    # Guardar validación
    validation_df.to_csv('validation.csv', index=False)

    return train_df, validation_df

# Ejemplo de uso
data_df = pd.read_csv('data.csv')
train_df, validation_df = split_data(data_df, '2018-01-01')
```

## 3. Preprocesamiento de Datos

### Pasos del Preprocesamiento

1.  **Eliminación de Columnas Innecesarias:** Se eliminan columnas irrelevantes.
2.  **Renombrado de Columnas:** Se ajustan los nombres para mayor claridad.
3.  **Transformación de Fechas y Creación de Variables:** Se convierten fechas y se generan nuevas variables (diferencia de días, cancelaciones).
4.  **Conversión de Columnas Booleanas:** Se transforman a valores numéricos (0 y 1).
5.  **Manejo de Valores Atípicos (Outliers):** Se identifican y tratan con IQR.
6.  **Separación de Columnas:** Se dividen en numéricas y categóricas.
7.  **Imputación de Valores Faltantes:** Se reemplazan con media o moda.
8.  **Codificación de Columnas Categóricas:** Se transforman con "one-hot encoding".
9.  **Normalización de Columnas Numéricas:** Se normalizan para escala similar.
10. **Creación de DataFrame Final:** Se genera DataFrame limpio y transformado.
11. **Guardado de DataFrame:** Se guarda en un archivo CSV.
12. **Eliminación de Columnas de Baja Correlación:** Se eliminan columnas con baja correlación a la variable objetivo.

### Técnicas y Herramientas Utilizadas

* **Pandas:** Manipulación de DataFrames.
* **NumPy:** Operaciones numéricas.
* **Scikit-learn:** Normalización, codificación e imputación.

### Resultado

Conjunto de datos limpio y transformado, listo para el entrenamiento de modelos de aprendizaje automático.
```
