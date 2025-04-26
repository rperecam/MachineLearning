# Sistema Avanzado de Predicción de Cancelaciones Hoteleras

## Descripción General
Este proyecto implementa un sistema predictivo para identificar con alta precisión las reservas hoteleras con mayor probabilidad de cancelación con al menos 30 días de antelación, permitiendo a los establecimientos implementar estrategias proactivas para mitigar pérdidas económicas.

## El Problema

En la industria hotelera, las cancelaciones anticipadas representan un desafío crítico para la gestión eficiente del inventario y la maximización de ingresos. Las cancelaciones tardías (dentro de los 30 días previos a la llegada) son especialmente problemáticas ya que:

- Dejan poco margen para recuperar la habitación con nuevas reservas
- Afectan la planificación de recursos y personal
- Impactan negativamente en los ingresos proyectados

## Modelos Implementados

El proyecto ofrece dos enfoques complementarios:

### 1. Modelo de Regresión Logística (`logistic/`)
- Enfoque baseline con fuerte interpretabilidad
- Ingeniería de características basada en conocimiento del dominio
- Manejo de outliers y valores nulos integrado en pipeline
- Métricas clave: F1-score (0.45), Recall (0.77), AUC-ROC (0.75)

### 2. Ensemble Avanzado (`ensembled/`)
- Arquitectura stacking con múltiples algoritmos
  - XGBoost
  - LightGBM
  - RandomForest
  - Gradient Boosting
- Validación cruzada estratificada por grupos (hotel_id)
- Técnicas avanzadas para manejo de desbalanceo de clases
- Métricas clave: F1-score (0.55), Recall (0.99), AUC-ROC (0.96)

## Resultados Principales

El ensemble muestra un rendimiento superior con:
- **Recall extremadamente alto (99%)**: Detecta casi todas las cancelaciones potenciales
- **Buena precisión (38%)**: Equilibrio razonable entre detección y falsos positivos
- **Excelente capacidad discriminativa (AUC-ROC: 0.96)**: Clara separación entre clases

El modelo está optimizado para maximizar la detección de cancelaciones (recall), priorizando la identificación de todas las posibles cancelaciones aunque esto implique algunos falsos positivos.

## Estructura del Proyecto

```
proyecto/
├── data/                # Datos de entrenamiento e inferencia
├── models/              # Modelos entrenados guardados
├── ensembled/           # Implementación del modelo ensemble
│   ├── *.py             # Código fuente del modelo ensemble
│   ├── Dockerfile       # Configuración para despliegue
│   └── README.md        # Memoria detallada del ensemble
├── logistic/            # Implementación del modelo baseline
│   ├── *.py             # Código fuente del modelo logístico
│   ├── Dockerfile       # Configuración para despliegue
│   └── README.md        # Memoria detallada del modelo logístico
└── README.md            # Este archivo
```

## Uso con Docker

### Para construir la imagen:
```bash
docker build -t hotel-predictor -f ensembled/Dockerfile .
```

### Para entrenar el modelo:
```bash
docker run -v "$(pwd)/models:/app/models" hotel-predictor
```

### Para ejecutar inferencia:
```bash
docker run -e SCRIPT_TO_RUN=inference -v "$(pwd)/models:/app/models" -v "$(pwd)/data:/app/data" hotel-predictor
```

## Conclusiones

Este sistema proporciona una solución robusta al problema de cancelaciones hoteleras, con un enfoque especial en:

- **Alta sensibilidad**: Capacidad para detectar prácticamente todas las cancelaciones potenciales
- **Procesamiento integrado**: Pipeline completo desde datos crudos hasta predicciones
- **Validación rigurosa**: Estrategias avanzadas para evitar fugas de datos y sobreajuste
- **Despliegue simplificado**: Containerización completa para entornos productivos

El desarrollo iterativo, basado en lecciones aprendidas y feedback, ha permitido crear un modelo que equilibra efectividad predictiva con aplicabilidad práctica en entornos reales de negocio hotelero.