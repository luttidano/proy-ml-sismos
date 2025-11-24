# Proyecto ML Sismos

## Objetivo
Predecir la ocurrencia de réplicas fuertes tras un mainshock, estudiando ingeniería de atributos sísmicos y evitando fuga temporal (data leakage).

## Etapas
1. Ingeniería de features (conteos históricos, ratios, distancias, clasificación mainshock M>=5.5).  
2. Etiquetado y fusión de clases de ventana temporal de réplica.  
3. Análisis de relevancia y dimensionalidad (correlaciones, PCA, kNN incremental, distancias).  

## Prevención de Leakage
- Uso de operaciones acumuladas históricas (cumcount, shift) para no mirar eventos futuros.
- Definición de mainshock aislado en ventana temporal y espacial previa.

## Dimensionalidad
- Ratio muestras/variables cómodo en Etapa 1, crítico en Etapa 2 (n muy bajo).  
- PCA muestra compresibilidad (90% varianza con pocas componentes).  
- kNN: rendimiento similar con pocas features.  

## Pendiente
- Validación temporal formal.  
- Placeholders: similitud_promedio_vecinos, conflicto_modelos (implementación o eliminación).  
- Baselines LogisticRegression / RandomForest con métricas (precision, recall, F1, PR curve).  
- Comparación representaciones: todas vs PCA vs subset correlación.  

## Estructura Carpeta
- `1era_etapa/` Ingeniería inicial.  
- `2da_etapa_creacion_features/` Generación y visualización.  
- `3ra_etapa_preprocesamiento/` Notebooks de selección y dimensionalidad.  

## Reproducibilidad
Fijar random_state en splits y modelos; centralizar configuración en futuro `config.json`.

## Cómo Ejecutar
1. Instalar dependencias (pandas, numpy, scikit-learn, matplotlib).  
2. Abrir notebooks en `3ra_etapa_preprocesamiento/`.  
3. Ejecutar de arriba a abajo para regenerar análisis.  

## Licencia
Añadir licencia si se publica públicamente.
