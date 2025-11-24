"""
Utilidades para Validación Temporal y Preprocesamiento
========================================================
Funciones reutilizables para entrenar modelos con división temporal correcta.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def limpiar_columnas_vacias(df):
    """
    Elimina columnas que tienen 100% de valores NaN (placeholders vacíos).
    
    Parámetros:
    -----------
    df : pandas.DataFrame
        Dataset a limpiar
        
    Retorna:
    --------
    df_limpio : pandas.DataFrame
        Dataset sin columnas totalmente vacías
    columnas_eliminadas : list
        Lista de nombres de columnas eliminadas
    """
    columnas_vacias = [col for col in df.columns if df[col].isna().all()]
    if columnas_vacias:
        print(f'[Limpieza] Columnas 100% NaN eliminadas: {columnas_vacias}')
        df_limpio = df.drop(columns=columnas_vacias)
    else:
        print('[Limpieza] No se encontraron columnas 100% NaN.')
        df_limpio = df.copy()
    
    return df_limpio, columnas_vacias


def split_temporal(df, col_fecha='Date(UTC)', porcentaje_train=0.7):
    """
    Divide dataset en entrenamiento y prueba usando orden temporal.
    
    Parámetros:
    -----------
    df : pandas.DataFrame
        Dataset con columna de fecha/hora
    col_fecha : str
        Nombre de la columna con timestamps
    porcentaje_train : float
        Porcentaje de datos para entrenamiento (el resto va a prueba)
        
    Retorna:
    --------
    train : pandas.DataFrame
        Datos de entrenamiento (eventos más antiguos)
    test : pandas.DataFrame
        Datos de prueba (eventos más recientes)
    """
    # Asegurar que la columna de fecha es tipo datetime
    if df[col_fecha].dtype != 'datetime64[ns]':
        df[col_fecha] = pd.to_datetime(df[col_fecha])
    
    # Ordenar por fecha
    df_sorted = df.sort_values(col_fecha).reset_index(drop=True)
    
    # Calcular punto de corte
    n_train = int(len(df_sorted) * porcentaje_train)
    
    train = df_sorted.iloc[:n_train].copy()
    test = df_sorted.iloc[n_train:].copy()
    
    fecha_corte = train[col_fecha].max()
    print(f'[Split Temporal] Train: {len(train)} eventos hasta {fecha_corte}')
    print(f'[Split Temporal] Test: {len(test)} eventos posteriores')
    
    return train, test


def preparar_datos(df, columnas_features, col_target, imputar=True, escalar=True):
    """
    Prepara datos para entrenamiento: imputación, escalado y separación X/y.
    
    Parámetros:
    -----------
    df : pandas.DataFrame
        Dataset completo
    columnas_features : list
        Lista de nombres de columnas a usar como features
    col_target : str
        Nombre de la columna target
    imputar : bool
        Si True, rellena NaN con medianas
    escalar : bool
        Si True, aplica StandardScaler
        
    Retorna:
    --------
    X : pandas.DataFrame o numpy.ndarray
        Features preparadas
    y : pandas.Series
        Target
    scaler : StandardScaler o None
        Escalador ajustado (None si escalar=False)
    """
    # Separar features y target
    X = df[columnas_features].copy()
    y = df[col_target].copy()
    
    # Imputación con medianas
    if imputar:
        medianas = X.median()
        X = X.fillna(medianas)
        n_imputados = df[columnas_features].isna().sum().sum()
        if n_imputados > 0:
            print(f'[Imputación] {n_imputados} valores NaN rellenados con medianas.')
    
    # Escalado
    scaler = None
    if escalar:
        scaler = StandardScaler()
        X = pd.DataFrame(
            scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        print('[Escalado] Features estandarizadas (media=0, std=1).')
    
    return X, y, scaler


def preparar_train_test(train_df, test_df, columnas_features, col_target):
    """
    Prepara conjuntos de entrenamiento y prueba con el mismo escalador.
    
    Parámetros:
    -----------
    train_df : pandas.DataFrame
        Dataset de entrenamiento
    test_df : pandas.DataFrame
        Dataset de prueba
    columnas_features : list
        Columnas a usar como features
    col_target : str
        Columna target
        
    Retorna:
    --------
    X_train, X_test : numpy.ndarray
        Features escaladas
    y_train, y_test : numpy.ndarray
        Targets
    scaler : StandardScaler
        Escalador ajustado en train
    """
    # Separar features y target
    X_train_raw = train_df[columnas_features].copy()
    X_test_raw = test_df[columnas_features].copy()
    y_train = train_df[col_target].values
    y_test = test_df[col_target].values
    
    # Imputar medianas (calculadas solo en train)
    medianas_train = X_train_raw.median()
    X_train = X_train_raw.fillna(medianas_train)
    X_test = X_test_raw.fillna(medianas_train)  # Usar medianas de train
    
    # Escalar (fit solo en train, transform en ambos)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f'[Preparación] Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}')
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def obtener_columnas_numericas(df, ignorar=None):
    """
    Extrae nombres de columnas numéricas, excluyendo targets, identificadores y placeholders.
    
    Parámetros:
    -----------
    df : pandas.DataFrame
        Dataset
    ignorar : list, optional
        Columnas adicionales a excluir. Si None, usa lista por defecto.
        
    Retorna:
    --------
    columnas_numericas : list
        Lista de nombres de columnas numéricas útiles
    """
    # Columnas que NUNCA deben usarse como features
    excluir_siempre = [
        'id_sismo_principal',          # Identificador único
        'Date(UTC)',                    # Fecha (solo para split temporal)
        'DateTime',                     # Alias de fecha
        'existe_replica_fuerte',        # TARGET Etapa 1
        'ventana_temporal_replica',     # TARGET Etapa 2
        'es_mainshock',                 # Bandera de filtro (no es feature)
        'similitud_promedio_vecinos',   # Placeholder vacío (100% NaN)
        'conflicto_modelos'             # Placeholder vacío (100% NaN)
    ]
    
    # Combinar con columnas adicionales a ignorar
    if ignorar is not None:
        excluir_siempre.extend(ignorar)
    
    # Extraer solo numéricas que no estén en la lista de exclusión
    columnas_numericas = [
        col for col in df.select_dtypes(include=['int64', 'float64']).columns
        if col not in excluir_siempre
    ]
    
    print(f'[Features] {len(columnas_numericas)} columnas numéricas seleccionadas.')
    print(f'[Excluidas] {len(excluir_siempre)} columnas (targets, IDs, placeholders).')
    
    return columnas_numericas
