"""
DEMOSTRACIÓN: Sistema Cascada de Predicción de Réplicas Sísmicas
Predice si habrá réplica fuerte y cuándo ocurrirá
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys

# Agregar ruta para importar la clase SistemaCascada
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# Definir clase SistemaCascada (necesaria para deserializar el .pkl)
class SistemaCascada:
    """Sistema cascada de 2 etapas para predicción de réplicas sísmicas."""
    
    def __init__(self, modelo_etapa1, scaler_etapa1, imputer_etapa1, features_etapa1,
                 modelo_etapa2, scaler_etapa2, imputer_etapa2, features_etapa2,
                 umbral_etapa1=0.5):
        self.modelo_etapa1 = modelo_etapa1
        self.scaler_etapa1 = scaler_etapa1
        self.imputer_etapa1 = imputer_etapa1
        self.features_etapa1 = features_etapa1
        self.modelo_etapa2 = modelo_etapa2
        self.scaler_etapa2 = scaler_etapa2
        self.imputer_etapa2 = imputer_etapa2
        self.features_etapa2 = features_etapa2
        self.umbral_etapa1 = umbral_etapa1
        self.mapeo_temporal = {0: '0-24h', 1: '24-72h', 2: '>72h', 3: 'Sin réplica'}
    
    def preprocesar_etapa1(self, df):
        X = df[self.features_etapa1].copy()
        X_imputed = self.imputer_etapa1.transform(X)
        X_scaled = self.scaler_etapa1.transform(X_imputed)
        return X_scaled
    
    def preprocesar_etapa2(self, df):
        X = df[self.features_etapa2].copy()
        X_imputed = self.imputer_etapa2.transform(X)
        X_scaled = self.scaler_etapa2.transform(X_imputed)
        return X_scaled
    
    def predecir(self, df, retornar_detalles=False):
        n_eventos = len(df)
        X_etapa1 = self.preprocesar_etapa1(df)
        proba_etapa1 = self.modelo_etapa1.predict_proba(X_etapa1)[:, 1]
        pred_etapa1 = (proba_etapa1 >= self.umbral_etapa1).astype(int)
        
        predicciones_finales = np.full(n_eventos, 3, dtype=int)
        predicciones_temporales = np.full(n_eventos, -1, dtype=int)
        probas_etapa2 = np.zeros((n_eventos, 4))
        
        indices_con_replica = np.where(pred_etapa1 == 1)[0]
        
        if len(indices_con_replica) > 0:
            df_con_replica = df.iloc[indices_con_replica]
            X_etapa2 = self.preprocesar_etapa2(df_con_replica)
            pred_temporal = self.modelo_etapa2.predict(X_etapa2)
            proba_temporal = self.modelo_etapa2.predict_proba(X_etapa2)
            
            predicciones_finales[indices_con_replica] = pred_temporal
            predicciones_temporales[indices_con_replica] = pred_temporal
            
            for i, idx in enumerate(indices_con_replica):
                clases_etapa2 = self.modelo_etapa2.classes_
                for j, clase in enumerate(clases_etapa2):
                    probas_etapa2[idx, clase] = proba_temporal[i, j]
        
        if not retornar_detalles:
            return predicciones_finales
        else:
            return {
                'predicciones_finales': predicciones_finales,
                'pred_etapa1': pred_etapa1,
                'proba_etapa1': proba_etapa1,
                'pred_etapa2': predicciones_temporales,
                'proba_etapa2': probas_etapa2,
                'n_evaluados_etapa2': len(indices_con_replica),
                'indices_etapa2': indices_con_replica
            }
    
    def interpretar_prediccion(self, prediccion):
        if prediccion == 3:
            return "Sin réplica fuerte"
        else:
            return f"Réplica fuerte en {self.mapeo_temporal[prediccion]}"

# ============================================================================
# 1. CARGAR SISTEMA CASCADA COMPLETO
# ============================================================================
print("═" * 70)
print("🌍 SISTEMA DE PREDICCIÓN DE RÉPLICAS SÍSMICAS")
print("═" * 70)

# Cargar sistema guardado
ruta_sistema = os.path.join('4ta_etapa_modelado', 'sistema_cascada_COMPLETO.pkl')
sistema_completo = joblib.load(ruta_sistema)
sistema = sistema_completo['sistema']

print("\n✅ Sistema cascada cargado correctamente")
print(f"   - Modelo Etapa 1 (binario): {type(sistema.modelo_etapa1).__name__}")
print(f"   - Modelo Etapa 2 (temporal): {type(sistema.modelo_etapa2).__name__}")
print(f"   - Umbral activación Etapa 2: {sistema.umbral_etapa1}")

# ============================================================================
# 2. CREAR EVENTO DE EJEMPLO (Simular nuevo terremoto)
# ============================================================================
print("\n" + "═" * 70)
print("📍 NUEVO EVENTO SÍSMICO DETECTADO")
print("═" * 70)

# Cargar un evento real del dataset para usar como ejemplo
ruta_csv = os.path.join('3ra_etapa_preprocesamiento', 'seismic_features_fusion_final.csv')
master = pd.read_csv(ruta_csv)
mainshocks = master[master['es_mainshock'] == 1].copy()

# ============ SELECTOR INTELIGENTE DE CASOS ============
# Encuentra automáticamente casos interesantes para demostrar

print("\n🔍 Buscando casos interesantes en el dataset...")

# Buscar eventos con réplica por ventana temporal
eventos_con_replica = mainshocks[mainshocks['existe_replica_fuerte'] == 1]
eventos_sin_replica = mainshocks[mainshocks['existe_replica_fuerte'] == 0]

if len(eventos_con_replica) > 0:
    # Buscar por cada ventana temporal
    ventana_0_24h = eventos_con_replica[eventos_con_replica['ventana_temporal_replica'] == 0]
    ventana_24_72h = eventos_con_replica[eventos_con_replica['ventana_temporal_replica'] == 1]
    ventana_mas_72h = eventos_con_replica[eventos_con_replica['ventana_temporal_replica'] == 2]
    
    print(f"   ✓ Eventos con réplica 0-24h: {len(ventana_0_24h)}")
    print(f"   ✓ Eventos con réplica 24-72h: {len(ventana_24_72h)}")
    print(f"   ✓ Eventos con réplica >72h: {len(ventana_mas_72h)}")
    print(f"   ✓ Eventos sin réplica: {len(eventos_sin_replica)}")

# OPCIONES PARA PROBAR (cambia el número 1, 2, 3 o 4):
CASO_A_MOSTRAR = 1  # ← CAMBIA ESTE NÚMERO (1, 2, 3 o 4)

if CASO_A_MOSTRAR == 1 and len(ventana_24_72h) > 0:
    # Caso 1: Evento con réplica 24-72h
    idx_seleccionado = ventana_24_72h.index[0]
    print(f"\n📌 CASO SELECCIONADO: Evento con réplica 24-72h (índice {idx_seleccionado})")
    
elif CASO_A_MOSTRAR == 2 and len(ventana_mas_72h) > 0:
    # Caso 2: Evento con réplica >72h
    idx_seleccionado = ventana_mas_72h.index[0]
    print(f"\n📌 CASO SELECCIONADO: Evento con réplica >72h (índice {idx_seleccionado})")
    
elif CASO_A_MOSTRAR == 3 and len(ventana_0_24h) > 0:
    # Caso 3: Evento con réplica 0-24h (inmediata)
    idx_seleccionado = ventana_0_24h.index[0]
    print(f"\n📌 CASO SELECCIONADO: Evento con réplica 0-24h (índice {idx_seleccionado})")
    
elif CASO_A_MOSTRAR == 4 and len(eventos_sin_replica) > 0:
    # Caso 4: Evento sin réplica
    idx_seleccionado = eventos_sin_replica.index[0]
    print(f"\n📌 CASO SELECCIONADO: Evento SIN réplica (índice {idx_seleccionado})")
    
else:
    # Fallback: primer evento disponible
    idx_seleccionado = mainshocks.index[0]
    print(f"\n📌 CASO SELECCIONADO: Primer evento disponible (índice {idx_seleccionado})")

evento_nuevo = mainshocks.loc[[idx_seleccionado]].copy()

# Mostrar información del evento
print(f"\n📊 DATOS DEL TERREMOTO:")
print(f"   Fecha: {evento_nuevo['Date(UTC)'].values[0]}")
print(f"   Ubicación: Lat {evento_nuevo['Latitude'].values[0]:.2f}°, Lon {evento_nuevo['Longitude'].values[0]:.2f}°")
print(f"   Magnitud: {evento_nuevo['Magnitude'].values[0]}")
print(f"   Profundidad: {evento_nuevo['Depth'].values[0]} km")
print(f"   Distancia a costa: {evento_nuevo['distancia_a_costa_km'].values[0]:.1f} km")
print(f"   Energía liberada: {evento_nuevo['energia_liberada_estimada'].values[0]:.2e} J")

# ============================================================================
# 3. EJECUTAR PREDICCIÓN CON SISTEMA CASCADA
# ============================================================================
print("\n" + "═" * 70)
print("🤖 EJECUTANDO PREDICCIÓN EN SISTEMA CASCADA")
print("═" * 70)

# Realizar predicción con detalles
resultado = sistema.predecir(evento_nuevo, retornar_detalles=True)

# --- ETAPA 1: Predicción Binaria ---
print("\n🔹 ETAPA 1: ¿HABRÁ RÉPLICA FUERTE?")
print("-" * 70)

prob_replica = resultado['proba_etapa1'][0]
pred_replica = resultado['pred_etapa1'][0]

print(f"   Probabilidad de réplica fuerte: {prob_replica:.1%}")
print(f"   Umbral de decisión: {sistema.umbral_etapa1}")

if pred_replica == 1:
    print(f"   ⚠️  PREDICCIÓN: SÍ HABRÁ RÉPLICA FUERTE")
else:
    print(f"   ✅ PREDICCIÓN: NO SE ESPERA RÉPLICA FUERTE")

# --- ETAPA 2: Predicción Temporal (solo si Etapa 1 dice Sí) ---
if pred_replica == 1:
    print("\n🔹 ETAPA 2: ¿CUÁNDO OCURRIRÁ LA RÉPLICA?")
    print("-" * 70)
    
    pred_temporal = resultado['predicciones_finales'][0]
    interpretacion = sistema.interpretar_prediccion(pred_temporal)
    
    print(f"   Ventana temporal predicha: {interpretacion}")
    
    # Mostrar probabilidades por ventana
    probas_temp = resultado['proba_etapa2'][0]
    print(f"\n   Probabilidades por ventana temporal:")
    mapeo = {0: '0-24h', 1: '24-72h', 2: '>72h'}
    for clase, prob in enumerate(probas_temp):
        if prob > 0:
            print(f"      {mapeo.get(clase, 'N/A')}: {prob:.1%}")
else:
    print("\n   ⏭️  ETAPA 2: No ejecutada (predicción Etapa 1 = No réplica)")

# ============================================================================
# 4. COMPARAR CON LA REALIDAD (si disponible)
# ============================================================================
print("\n" + "═" * 70)
print("📖 COMPARACIÓN CON LA REALIDAD")
print("═" * 70)

realidad_replica = evento_nuevo['existe_replica_fuerte'].values[0]

if realidad_replica == 1:
    ventana_real = evento_nuevo['ventana_temporal_replica'].values[0]
    mapeo_ventanas = {0: '0-24h', 1: '24-72h', 2: '>72h'}
    
    print(f"\n   ✅ REALIDAD: SÍ ocurrió réplica fuerte")
    print(f"   Ventana temporal real: {mapeo_ventanas.get(ventana_real, 'Desconocida')}")
    
    # Evaluar precisión
    prediccion_final = resultado['predicciones_finales'][0]
    
    if pred_replica == 0:
        print(f"\n   ❌ RESULTADO: PREDICCIÓN INCORRECTA")
        print(f"      El modelo NO detectó la réplica (Falso Negativo)")
        print(f"      ⚠️ PELIGROSO: Se perdió una alerta importante")
    else:
        if prediccion_final == ventana_real:
            print(f"\n   ✅ RESULTADO: PREDICCIÓN TOTALMENTE CORRECTA")
            print(f"      Detectó la réplica Y acertó la ventana temporal")
        else:
            print(f"\n   ⚠️ RESULTADO: PREDICCIÓN PARCIALMENTE CORRECTA")
            print(f"      Detectó la réplica pero erró la ventana temporal")
            print(f"      Predicho: {mapeo_ventanas.get(prediccion_final)}, Real: {mapeo_ventanas.get(ventana_real)}")
else:
    print(f"\n   ℹ️  REALIDAD: NO ocurrió réplica fuerte")
    
    if pred_replica == 1:
        print(f"\n   ⚠️ RESULTADO: FALSA ALARMA")
        print(f"      El modelo predijo réplica pero no ocurrió")
        print(f"      (En seguridad sísmica, es preferible esto a no alertar)")
    else:
        print(f"\n   ✅ RESULTADO: PREDICCIÓN CORRECTA")
        print(f"      El modelo acertó que NO habría réplica")

# ============================================================================
# 5. RECOMENDACIONES PARA PROTECCIÓN CIVIL
# ============================================================================
print("\n" + "═" * 70)
print("🚨 RECOMENDACIONES PARA PROTECCIÓN CIVIL")
print("═" * 70)

if pred_replica == 1:
    pred_final = resultado['predicciones_finales'][0]
    
    if pred_final == 0:  # 0-24h
        print("\n   🔴 ALERTA MÁXIMA: Réplica esperada en las próximas 24 horas")
        print("   • Evacuar edificios dañados INMEDIATAMENTE")
        print("   • Activar protocolos de emergencia")
        print("   • Reforzar monitoreo sísmico continuo")
    elif pred_final == 1:  # 24-72h
        print("\n   🟠 ALERTA ALTA: Réplica esperada entre 24-72 horas")
        print("   • Inspeccionar infraestructura crítica")
        print("   • Preparar albergues temporales")
        print("   • Mantener equipos de rescate en alerta")
    elif pred_final == 2:  # >72h
        print("\n   🟡 ALERTA MODERADA: Réplica esperada después de 72 horas")
        print("   • Monitoreo sísmico extendido")
        print("   • Evaluación de daños estructurales")
        print("   • Comunicación preventiva a la población")
else:
    print("\n   🟢 ALERTA BAJA: No se espera réplica fuerte inmediata")
    print("   • Mantener monitoreo sísmico rutinario")
    print("   • Evaluar daños menores")
    print("   • Comunicar calma a la población")

# ============================================================================
# 6. INFORMACIÓN TÉCNICA DEL MODELO
# ============================================================================
#print("\n" + "═" * 70)
#print("ℹ️  INFORMACIÓN TÉCNICA DEL SISTEMA")
#print("═" * 70)

#print(f"\n   📊 Rendimiento esperado (según K-Fold CV):")
#print(f"      • Etapa 1 - Recall: ~87.5% (detecta ~9 de 10 réplicas)")
#print(f"      • Etapa 1 - Precision: ~22.2% (1 de 5 alarmas es correcta)")
#print(f"      • Sistema optimizado para NO perder réplicas reales")
#print(f"\n   ⚠️ Limitación reconocida:")
#print(f"      • Dataset pequeño: 236 mainshocks (25 con réplica)")
#print(f"      • Proyecto académico - NO para uso en producción real")
#print(f"      • Requiere validación con más datos y expertos sísmicos")

print("\n" + "═" * 70)
print("✅")
print("═" * 70)