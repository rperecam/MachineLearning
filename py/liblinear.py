import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def find_best_logistic_params_and_k_grid(X_train_processed, y_train, X_test_processed, y_test):
    """
    Encuentra los mejores hiperparámetros de LogisticRegression y el mejor valor de k usando GridSearchCV.
    """

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train_processed, y_train)

    pipeline = Pipeline([
        ('selector', SelectKBest(score_func=f_classif)),
        ('model', LogisticRegression(solver='liblinear')) # solver = liblinear.
    ])

    param_grid = {
        'selector__k': range(1, min(X_train_processed.shape[1] + 1, 10)), #Rango reducido.
        'model__C': [0.01, 0.1, 1, 10], # Rango reducido.
        'model__penalty': ['l1', 'l2'],
        'model__max_iter': [1000] # Rango reducido.
    }

    # Verificar si se está usando StratifiedKFold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Usar tqdm para monitorizar el progreso de GridSearchCV
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='f1', verbose=10)
    grid_search.fit(X_resampled, y_resampled)

    print("Mejores parámetros:", grid_search.best_params_)
    print("Mejor score (F1):", grid_search.best_score_)

    best_features = X_train_processed.columns[grid_search.best_estimator_.named_steps['selector'].get_support()]
    print("\nMejores características seleccionadas:")
    print(best_features)

    best_pipeline = grid_search.best_estimator_
    best_pipeline.fit(X_resampled, y_resampled)

    y_pred = best_pipeline.predict(X_test_processed)
    y_pred_proba = best_pipeline.predict_proba(X_test_processed)[:, 1]

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

    cv_scores = cross_val_score(best_pipeline, X_resampled, y_resampled, cv=cv, scoring='f1')
    print(f"\nValidación cruzada (F1): {cv_scores}")
    print(f"Media de validación cruzada (F1): {np.mean(cv_scores)}")