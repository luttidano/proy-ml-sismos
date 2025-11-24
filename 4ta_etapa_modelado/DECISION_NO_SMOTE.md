# Decisión: NO Usar SMOTE ni Oversampling

## Contexto

En problemas de clasificación con clases desbalanceadas (ej. pocas réplicas fuertes vs muchos casos sin réplica), hay varias estrategias:

1. **SMOTE / Oversampling:** Crear datos sintéticos de la clase minoritaria.
2. **Undersampling:** Reducir datos de la clase mayoritaria.
3. **Class Weights:** Ajustar pesos en la función de pérdida.

---

## Decisión Tomada

**Usamos `class_weight='balanced'`** en vez de SMOTE u oversampling.

---

## Razones

### ✅ Ventajas de `class_weight='balanced'`

1. **No inventa datos:** Solo ajusta la importancia relativa de cada clase en el entrenamiento.
2. **Más realista:** Los modelos aprenden de datos reales, no sintéticos.
3. **Validación temporal limpia:** No hay riesgo de que datos sintéticos "contaminen" el test set.
4. **Simplicidad:** No requiere bibliotecas adicionales (imblearn, SMOTE).
5. **Evita sobreajuste artificial:** SMOTE puede crear patrones que no existen en la realidad.

### ❌ Problemas con SMOTE/Oversampling

1. **Datos sintéticos no representan sismos reales:** Interpolar entre dos terremotos no necesariamente crea un sismo válido.
2. **Puede crear patrones falsos:** Especialmente en espacios de alta dimensión.
3. **Riesgo de fuga:** Si se aplica antes del split temporal, contamina el test set.
4. **Complejidad innecesaria:** Para este proyecto, class_weight es suficiente.

---

## Implementación

### Decision Tree
```python
dt_model = DecisionTreeClassifier(
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',  # ← AQUÍ: Ajuste automático de pesos
    random_state=42
)
```

**Efecto:** La clase minoritaria (réplicas fuertes) recibe más peso en el cálculo de impureza (Gini/Entropy), forzando al árbol a prestar más atención a esos casos.

### kNN (K-Vecinos)

```python
knn_model = KNeighborsClassifier(
    n_neighbors=5,
    weights='distance',  # ← Vecinos cercanos tienen más peso (ayuda con desbalance)
    metric='euclidean'
)
```

**Nota:** kNN no tiene `class_weight` directamente, pero:
- `weights='distance'` da más importancia a vecinos muy cercanos.
- Alternativamente, podríamos usar `sample_weight` en `.fit()` pero no es necesario aquí.

---

## Validación de la Decisión

**Cómo saber si funciona:**
1. Observar **Recall** en test set: ¿Detecta suficientes réplicas reales?
2. Comparar **Precision** vs **Recall**: ¿El trade-off es aceptable?
3. Matriz de confusión: ¿Falsos Negativos (FN) son bajos? (crítico para seguridad)

**Si Recall fuera muy bajo (<0.5):**
- Considerar ajustar `n_neighbors` en kNN.
- Aumentar peso manual con `sample_weight`.
- Como último recurso, probar SMOTE con validación temporal estricta.

---

## Comparación con Etapa 2

**Etapa 2** (clasificación multiclase de ventana temporal):
- También usa `class_weight='balanced'`.
- Tiene MÁS desbalance (clases 1, 2, 3, 4 con distribuciones muy diferentes).
- Si Etapa 2 muestra problemas graves, AHORA SÍ podríamos considerar balanceo manual o SMOTE aplicado **solo en train** después del split temporal.

---

## Referencias

- Scikit-learn docs: [class_weight parameter](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
- Problema de SMOTE en validación temporal: [Time Series Split with Imbalanced Data](https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/)
