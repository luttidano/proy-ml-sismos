# Guía de Columnas para Entrenamiento

## ❌ Columnas que NO se usan como Features

### Identificadores y Fechas
- `id_sismo_principal` → Identificador único (no aporta información)
- `Date(UTC)` → Fecha/hora (se usa solo para ordenar en split temporal)

### Targets (Lo que queremos predecir)
- `existe_replica_fuerte` → **TARGET de Etapa 1** (predicción binaria: Sí/No)
- `ventana_temporal_replica` → **TARGET de Etapa 2** (clasificación: 0-24h, 24-72h, >72h, sin réplica)

### Banderas de Filtro
- `es_mainshock` → Se usa para filtrar datos (separar Etapa 1 de Etapa 2), NO es una feature

### Placeholders Vacíos (100% NaN)
- `similitud_promedio_vecinos` → Pendiente de implementar
- `conflicto_modelos` → Pendiente de implementar

---

## ✅ Columnas que SÍ se usan como Features

### Geográficas (7 columnas)
- `Latitude` → Latitud del epicentro
- `Longitude` → Longitud del epicentro
- `Depth` → Profundidad del hipocentro (km)
- `celda_geografica` → Celda geográfica (categórica)
- `zona_sismica` → Zona sísmica (Norte, Centro, Sur)
- `distancia_a_costa_km` → Distancia a la costa
- `estacion_año` → Estación del año (categórica)

### Características del Terremoto (4 columnas)
- `Magnitude` → Magnitud del sismo
- `es_sismo_somero` → Si profundidad < 70 km (binaria)
- `intensidad_categoria` → Categoría de intensidad (categórica)
- `energia_liberada_estimada` → Energía calculada

### Actividad Histórica SIN FUGA (4 columnas)
Estas variables calculan la actividad **hasta la fecha del evento**, mirando hacia el pasado.

- `actividad_M5_15d` → Sismos M≥5 en últimos 15 días
- `actividad_M6_30d` → Sismos M≥6 en últimos 30 días
- `actividad_M7_90d` → Sismos M≥7 en últimos 90 días
- `actividad_reciente_completa` → Total de actividad reciente

### Contexto Regional (3 columnas)
- `brecha_magnitud_zona` → Diferencia con sismo previo en la zona
- `sismos_previos_celda` → Conteo histórico incremental en celda
- `densidad_sismica_zona` → Densidad de eventos en la zona

### Ratios Históricos SIN FUGA (3 columnas)
Calculados con datos históricos **hasta la fecha del evento**.

- `ratio_replicas_24h` → Proporción histórica de réplicas en 0-24h
- `ratio_replicas_48h` → Proporción histórica de réplicas en 24-48h
- `ratio_replicas_72h` → Proporción histórica de réplicas en 48-72h

### Umbral Calculado (1 columna)
- `magnitud_umbral` → Umbral de magnitud para la zona

---

## 📊 Resumen

**Total de features numéricas utilizables:** ~21 columnas  
(Depende de si usas las categóricas con encoding o las excluyes)

**Columnas categóricas que requieren encoding:**
- `celda_geografica`
- `zona_sismica`
- `intensidad_categoria`
- `estacion_año`

**Tratamiento recomendado:**
1. **Opción A (Simple):** Usar solo features numéricas → ~17 features
2. **Opción B (Completa):** Aplicar One-Hot Encoding a categóricas → ~30-40 features

---

## ⚠️ Validación Temporal Obligatoria

**NUNCA** usar split aleatorio (train_test_split con shuffle=True) porque:
- Mezcla eventos pasados y futuros.
- El modelo "aprende" del futuro sin darse cuenta (fuga temporal).

**Siempre** usar `split_temporal()`:
- Entrena con eventos **antiguos** (ej. 2012-2018).
- Prueba con eventos **recientes** (ej. 2019-2020).
- Simula predicción real: predecir el futuro con datos del pasado.

---

## 🔧 Uso en Código

```python
from utils_validacion import obtener_columnas_numericas, limpiar_columnas_vacias

# Cargar y limpiar
df = pd.read_csv('seismic_features_fusion_final.csv')
df, cols_vacias = limpiar_columnas_vacias(df)

# Obtener features numéricas automáticamente
# (excluye targets, IDs, placeholders)
features = obtener_columnas_numericas(df)

print(f"Features a usar: {len(features)}")
print(features)
```

**Salida esperada:**
```
[Features] 17 columnas numéricas seleccionadas.
[Excluidas] 8 columnas (targets, IDs, placeholders).
['Latitude', 'Longitude', 'Depth', 'Magnitude', 'distancia_a_costa_km', ...]
```
