import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def process_data(X_train, X_test, numerical_cols, categorical_cols):
    """
    Realiza la normalización y la codificación one-hot de los datos.

    Args:
        X_train (pd.DataFrame): DataFrame de entrenamiento.
        X_test (pd.DataFrame): DataFrame de prueba.
        numerical_cols (list): Lista de columnas numéricas.
        categorical_cols (list): Lista de columnas categóricas.

    Returns:
        tuple: DataFrames de entrenamiento y prueba transformados.
    """

    # Pipelines para transformaciones numéricas y categóricas
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
    ])

    # ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='passthrough'
    )

    # Aplicar transformaciones al conjunto de entrenamiento
    X_train_transformed = preprocessor.fit_transform(X_train)

    # Aplicar transformaciones al conjunto de prueba
    X_test_transformed = preprocessor.transform(X_test)

    # Obtener nombres de las columnas transformadas
    numerical_cols_transformed = [col for col in numerical_cols]
    categorical_cols_transformed = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_cols).tolist()
    remaining_cols = [col for col in X_train.columns if col not in numerical_cols + categorical_cols]

    transformed_columns = numerical_cols_transformed + categorical_cols_transformed + remaining_cols

    # Convertir a DataFrames de pandas
    X_train_transformed = pd.DataFrame(X_train_transformed, columns=transformed_columns)
    X_test_transformed = pd.DataFrame(X_test_transformed, columns=transformed_columns)

    return X_train_transformed, X_test_transformed