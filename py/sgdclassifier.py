import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE

def find_best_sgdclassifier_params_and_k_grid(X_train_processed, y_train, X_test_processed, y_test):
    """
    Encuentra los mejores hiperparámetros de SGDClassifier y el mejor valor de k usando GridSearchCV.

    Optimizaciones:
    1.  Manejo del desbalanceo de clases con SMOTE.
    2.  Uso de early_stopping para evitar sobreajuste en SGDClassifier.
    3.  Reporte de métricas detallado y validación cruzada.

    Args:
        X_train_processed: Características del conjunto de entrenamiento.
        y_train: Etiquetas del conjunto de entrenamiento.
        X_test_processed: Características del conjunto de prueba.
        y_test: Etiquetas del conjunto de prueba.
    """

    # 1. Manejo del desbalanceo de clases con SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train_processed, y_train)

    # 2. Pipeline con SelectKBest y SGDClassifier
    pipeline = Pipeline([
        ('selector', SelectKBest(score_func=f_classif)),
        ('model', SGDClassifier(loss='log_loss', random_state=42, early_stopping=True))
    ])

    # 3. Definición del espacio de búsqueda de hiperparámetros
    param_grid = {
        'selector__k': range(1, min(X_train_processed.shape[1] + 1, 10)),
        'model__penalty': ['l1', 'l2'],
        'model__max_iter': [1000],
        'model__learning_rate': ['optimal', 'adaptive'],  # Añadir learning_rate
        'model__eta0': [0.001, 0.01, 0.1],  # Añadir eta0
        'model__alpha': [ 0.0001, 0.001, 0.01] # Añadir alpha
    }

    # 4. Validación cruzada estratificada
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 5. Búsqueda de los mejores hiperparámetros con GridSearchCV
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='f1', verbose=10, n_jobs=-1, error_score='raise')
    grid_search.fit(X_resampled, y_resampled)

    # 6. Reporte de los mejores parámetros y score
    print("Mejores parámetros:", grid_search.best_params_)
    print("Mejor score (F1):", grid_search.best_score_)

    # 7. Obtención de las mejores características seleccionadas
    best_features = X_train_processed.columns[grid_search.best_estimator_.named_steps['selector'].get_support()]
    print("\nMejores características seleccionadas:")
    print(best_features)

    # 8. Evaluación del mejor modelo en el conjunto de prueba
    best_pipeline = grid_search.best_estimator_
    y_pred = best_pipeline.predict(X_test_processed)
    y_pred_proba = best_pipeline.predict_proba(X_test_processed)[:, 1]

    # 9. Cálculo y reporte de métricas de rendimiento
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print("\nMétricas de rendimiento en el conjunto de prueba:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-score: {f1}")
    print(f"ROC AUC: {roc_auc}")

    # 10. Validación cruzada del mejor modelo
    cv_scores = cross_val_score(best_pipeline, X_resampled, y_resampled, cv=cv, scoring='f1', n_jobs=-1)
    print(f"\nValidación cruzada (F1): {cv_scores}")
    print(f"Media de validación cruzada (F1): {np.mean(cv_scores)}")