# ...existing code...
# === Ingeniería de atributos y exportación ===
import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt
from datetime import timedelta

df = pd.read_csv('1era_etapa/seismic_data_clean.csv')

# Asegurar Date(UTC) en datetime y dataset ordenado
df = df.copy()
df['Date(UTC)'] = pd.to_datetime(df['Date(UTC)'])
df = df.sort_values('Date(UTC)').reset_index(drop=True)

# -------------------------
# Helpers
# -------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2.0)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2.0)**2
    c = 2 * asin(sqrt(a))
    return R * c

def estacion_surhem(m):
    # Verano: Dic-Feb, Otoño: Mar-May, Invierno: Jun-Ago, Primavera: Sep-Nov
    if m in (12, 1, 2):
        return 'Verano'
    if m in (3, 4, 5):
        return 'Otoño'
    if m in (6, 7, 8):
        return 'Invierno'
    return 'Primavera'

def zona_sismica_por_lat(lat):
    # Ajuste de signos para latitudes negativas (Chile):
    # Norte: lat >= -30; Centro: -40 <= lat < -30; Sur: lat < -40
    if lat >= -30:
        return 'Norte'
    if lat >= -40:
        return 'Centro'
    return 'Sur'

def categoria_intensidad(m):
    if m < 6.0:
        return 'Suave'
    if m < 7.0:
        return 'Moderado'
    if m < 8.0:
        return 'Fuerte'
    return 'Extremo'

# Anclajes costeros (aprox. a lo largo de la costa de Chile continental)
COAST_ANCHORS = [
    (-18.5, -70.35),  # Arica
    (-23.65, -70.40), # Antofagasta
    (-29.90, -71.25), # Coquimbo
    (-33.05, -71.63), # Valparaíso
    (-36.82, -73.05), # Concepción
    (-39.82, -73.24), # Valdivia
    (-41.47, -72.94), # Puerto Montt
    (-42.50, -73.80), # Chiloé
    (-45.40, -73.10), # Aysén
    (-52.97, -71.07)  # Magallanes
]

def distancia_costa_aprox_km(lat, lon):
    return min(haversine_km(lat, lon, a[0], a[1]) for a in COAST_ANCHORS)

# -------------------------
# Columnas base y features simples
# -------------------------
df['cell_lat'] = df['Latitude'].round().astype(int)
df['cell_lon'] = df['Longitude'].round().astype(int)
df['celda_geografica'] = df['cell_lat'].astype(str) + '_' + df['cell_lon'].astype(str)

df['id_sismo_principal'] = (
    df['Date(UTC)'].dt.strftime('%Y-%m-%dT%H:%M:%S') + '_' +
    df['Latitude'].round(2).astype(str) + '_' +
    df['Longitude'].round(2).astype(str)
)

df['magnitud_umbral'] = df['Magnitude'] - 1.0
df['energia_liberada_estimada'] = 10 ** (1.5 * df['Magnitude'] + 4.8)

df['zona_sismica'] = df['Latitude'].apply(zona_sismica_por_lat)
df['distancia_a_costa_km'] = df.apply(lambda r: distancia_costa_aprox_km(r['Latitude'], r['Longitude']), axis=1)
df['estacion_año'] = df['Date(UTC)'].dt.month.apply(estacion_surhem)
df['es_sismo_somero'] = (df['Depth'] < 70).astype(int)
df['intensidad_categoria'] = df['Magnitude'].apply(categoria_intensidad)

"""Densidad sísmica histórica incremental (sin fuga temporal).
Para cada evento se usa SOLO el número de sismos previos en la misma celda.
Área aprox celda: 111km * 111km * cos(lat_centro) = 12321 * cos(lat_centro)
"""
cell_lat_center = df.groupby('celda_geografica')['cell_lat'].first().to_dict()

def area_celda_km2(celda):
    latc = cell_lat_center[celda]
    return 12321.0 * abs(cos(radians(latc)))

# Conteo acumulado de previos en la celda (excluye el actual) ordenado por tiempo
df = df.sort_values('Date(UTC)').reset_index(drop=True)
df['sismos_previos_celda'] = df.groupby('celda_geografica').cumcount()
df['area_celda_km2'] = df['celda_geografica'].apply(area_celda_km2)
df['densidad_sismica_zona'] = df['sismos_previos_celda'] / df['area_celda_km2']

# -------------------------
# Identificación de mainshocks (umbral magnitud >=5.5)
# Criterio: evento con Magnitude >= 5.5 que NO tiene en las últimas 72h
# (ventana retrospectiva) otro evento con Magnitude >= la suya dentro de 100 km.
# Esto evita marcar réplicas grandes dentro de una secuencia previa como nuevos mainshocks.
# -------------------------
UMBRAL_MAINSHOCK = 5.5
WINDOW_HOURS = 72
RADIUS_KM = 100.0

df['es_mainshock'] = 0

latitudes = df['Latitude'].values
longitudes = df['Longitude'].values
fechas = df['Date(UTC)'].values
magnitudes = df['Magnitude'].values

for i in range(len(df)):
    mag_i = magnitudes[i]
    if mag_i < UMBRAL_MAINSHOCK:
        continue
    t_i = fechas[i]
    lat_i = latitudes[i]
    lon_i = longitudes[i]
    # Filtrar previos dentro de la ventana temporal
    # (fechas es un numpy array de dtype datetime64)
    t_min = t_i - np.timedelta64(int(WINDOW_HOURS*60*60), 's')
    # Índices previos candidatos
    # Usamos i porque df está ordenado por tiempo
    prev_indices = range(0, i)
    es_main = True
    for j in prev_indices:
        if fechas[j] < t_min:
            continue
        # Sólo considerar previos en ventana temporal
        if magnitudes[j] >= mag_i:
            # Calcular distancia
            d = haversine_km(lat_i, lon_i, latitudes[j], longitudes[j])
            if d <= RADIUS_KM:
                es_main = False
                break
    if es_main:
        df.at[i, 'es_mainshock'] = 1

# -------------------------
# Actividad reciente (AMLIADA)
# >M5 en 15d, >M6 en 30d, >M7 en 90d, en la MISMA celda_geografica
# -------------------------
df['actividad_M5_15d'] = 0
df['actividad_M6_30d'] = 0
df['actividad_M7_90d'] = 0

for celda, g in df.groupby('celda_geografica', group_keys=False):
    g = g.sort_values('Date(UTC)')
    idxs = g.index.values
    times = g['Date(UTC)'].values
    mags = g['Magnitude'].values
    # Para cada evento, contar previos en ventanas
    for i, idx in enumerate(idxs):
        t = g.loc[idx, 'Date(UTC)']
        # máscaras de previos (estrictamente antes)
        prev_mask = g['Date(UTC)'] < t

        m5_15 = g.loc[prev_mask & (g['Date(UTC)'] >= t - timedelta(days=15)) & (g['Magnitude'] > 5.0)].shape[0]
        m6_30 = g.loc[prev_mask & (g['Date(UTC)'] >= t - timedelta(days=30)) & (g['Magnitude'] > 6.0)].shape[0]
        m7_90 = g.loc[prev_mask & (g['Date(UTC)'] >= t - timedelta(days=90)) & (g['Magnitude'] > 7.0)].shape[0]

        df.at[idx, 'actividad_M5_15d'] = m5_15
        df.at[idx, 'actividad_M6_30d'] = m6_30
        df.at[idx, 'actividad_M7_90d'] = m7_90

df['actividad_reciente_completa'] = df['actividad_M5_15d'] + df['actividad_M6_30d'] + df['actividad_M7_90d']

# -------------------------
# Brecha con magnitud histórica de la celda (antes del evento)
# -------------------------
# Cálculo de máximo histórico previo por celda (sin fuga) usando transform para conservar índice
df['max_hist_celda_previo'] = (
        df.groupby('celda_geografica')['Magnitude']
            .transform(lambda s: s.cummax().shift(1))
)
df['brecha_magnitud_zona'] = df['Magnitude'] - df['max_hist_celda_previo'].fillna(df['Magnitude'])
df.loc[df['max_hist_celda_previo'].isna(), 'brecha_magnitud_zona'] = 0.0

# -------------------------
# Targets y ventanas de réplica fuerte (misma celda, umbral = M-1.0)
# existe_replica_fuerte (<=72h), ventana_temporal_replica (1..4)
# Además flags para ratios históricos (24/48/72h)
# -------------------------
df['existe_replica_fuerte'] = 0
df['ventana_temporal_replica'] = np.nan
df['has_replica_24h'] = 0
df['has_replica_48h'] = 0
df['has_replica_72h'] = 0

for celda, g in df.groupby('celda_geografica', group_keys=False):
    g = g.sort_values('Date(UTC)')
    for idx in g.index:
        t0 = g.loc[idx, 'Date(UTC)']
        mthr = g.loc[idx, 'magnitud_umbral']

        future = g[(g['Date(UTC)'] > t0)]  # futuros estrictos
        strong_future = future[future['Magnitude'] >= mthr]

        if strong_future.empty:
            continue

        # Primera réplica fuerte futura
        earliest_idx = strong_future['Date(UTC)'].idxmin()
        delta_h = (strong_future.loc[earliest_idx, 'Date(UTC)'] - t0).total_seconds() / 3600.0

        # Ventana (1:0-24, 2:24-48, 3:48-72, 4:>72)
        if delta_h <= 24:
            df.at[idx, 'ventana_temporal_replica'] = 1
            df.at[idx, 'has_replica_24h'] = 1
            df.at[idx, 'has_replica_48h'] = 1
            df.at[idx, 'has_replica_72h'] = 1
            df.at[idx, 'existe_replica_fuerte'] = 1
        elif delta_h <= 48:
            df.at[idx, 'ventana_temporal_replica'] = 2
            df.at[idx, 'has_replica_24h'] = 0
            df.at[idx, 'has_replica_48h'] = 1
            df.at[idx, 'has_replica_72h'] = 1
            df.at[idx, 'existe_replica_fuerte'] = 1
        elif delta_h <= 72:
            df.at[idx, 'ventana_temporal_replica'] = 3
            df.at[idx, 'has_replica_24h'] = 0
            df.at[idx, 'has_replica_48h'] = 0
            df.at[idx, 'has_replica_72h'] = 1
            df.at[idx, 'existe_replica_fuerte'] = 1
        else:
            df.at[idx, 'ventana_temporal_replica'] = 4
            # existe_replica_fuerte se mantiene 0 (porque es >72h)

# Asignar clase 5 a eventos sin ninguna réplica fuerte futura (permanecía NaN)
df['ventana_temporal_replica'] = df['ventana_temporal_replica'].fillna(5).astype(int)

# -------------------------
# Ratios históricos por celda (usando SOLO eventos previos)
# ratio_replicas_24h / 48h / 72h
# -------------------------
def expanding_ratio_prev(s):
        # media acumulada de previos: shift(1) para excluir el actual
        return s.shift(1).expanding().mean()

# Usar groupby.transform para conservar el índice original y evitar incompatibilidades
df['ratio_replicas_24h'] = (
        df.groupby('celda_geografica')['has_replica_24h']
            .transform(lambda s: s.shift(1).expanding().mean())
            .fillna(0.0)
)
df['ratio_replicas_48h'] = (
        df.groupby('celda_geografica')['has_replica_48h']
            .transform(lambda s: s.shift(1).expanding().mean())
            .fillna(0.0)
)
df['ratio_replicas_72h'] = (
        df.groupby('celda_geografica')['has_replica_72h']
            .transform(lambda s: s.shift(1).expanding().mean())
            .fillna(0.0)
)

# -------------------------
# Columnas pendientes (se calculan después de entrenar modelos)
# -------------------------
df['similitud_promedio_vecinos'] = np.nan
df['conflicto_modelos'] = np.nan

# -------------------------
# Selección de columnas finales y exportación
# -------------------------
cols_finales = [
    # ID y originales
    'id_sismo_principal','Date(UTC)','Latitude','Longitude','Depth','Magnitude',
    # Geografía y tiempo
    'celda_geografica','zona_sismica','distancia_a_costa_km','estacion_año',
    # Técnicas
    'es_sismo_somero','intensidad_categoria','energia_liberada_estimada','es_mainshock',
    # Actividad / historial
    'actividad_M5_15d','actividad_M6_30d','actividad_M7_90d','actividad_reciente_completa',
    'brecha_magnitud_zona','sismos_previos_celda','densidad_sismica_zona','ratio_replicas_24h','ratio_replicas_48h','ratio_replicas_72h',
    # Targets
    'magnitud_umbral','existe_replica_fuerte','ventana_temporal_replica',
    # Control (pendiente)
    'similitud_promedio_vecinos','conflicto_modelos'
]

features_df = df[cols_finales].copy()

# Guardar CSV de features
out_path = r'c:\Users\PC\Desktop\PROY ML\2da_etapa_creacion_features\seismic_features.csv'
features_df.to_csv(out_path, index=False, encoding='utf-8')
print(f'Features exportadas a: {out_path}')
features_df.head(10)
