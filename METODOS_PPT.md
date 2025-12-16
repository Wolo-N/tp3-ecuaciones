# SECCIÓN MÉTODOS - Presentación PPT

## ESTRUCTURA SUGERIDA PARA SLIDES

---

## SLIDE 1: MÉTODO DE VOLÚMENES FINITOS (FVM)

### Discretización Espacial

**Elementos Q9 Isoparamétricos**
- 9 nodos por elemento (4 esquinas + 4 lados + 1 centro)
- Interpolación cuadrática en 2D
- Permite geometrías curvas (ahusamiento y redondeo)

**Malla estructurada:**
- Coordenadas naturales: (ξ, η) ∈ [-1, 1]²
- Malla física: 21×21 nodos → 36 volúmenes de control (6×6)
- Nodos centrales: sampling de 1 cada 2 nodos

**Mapeo isoparamétrico:**
```
x(ξ,η) = Σ Nᵢ(ξ,η) · xᵢ
y(ξ,η) = Σ Nᵢ(ξ,η) · yᵢ
```

---

## SLIDE 2: CÁLCULO DE GEOMETRÍA

### Áreas y Volúmenes por FVM

**Áreas 2D (Método de Heron):**
- División de bloques 3×3 en triángulos
- Ordenamiento antihorario desde nodo central
- Fórmula de Heron para cada triángulo:
  ```
  s = (a + b + c)/2
  A_triángulo = √[s(s-a)(s-b)(s-c)]
  A_total = Σ A_triángulos
  ```

**Volúmenes 3D:**
```
V = A₂D × L_extrusión  [m³]
```

**Áreas de caras:**
- **Internas** (conducción): Longitud de arista compartida × L_extrusión
- **Superficiales** (convección): 2×A₂D + perímetro_frontera × L_extrusión

---

## SLIDE 3: ECUACIÓN DE BALANCE TÉRMICO

### Balance de Energía por Volumen de Control

**Estado transitorio:**
```
ρ cₚ V (dT/dt) = Q_conducción + Q_convección
```

**Conducción (Ley de Fourier):**
```
Q_cond = Σⱼ k Aᵢⱼ (Tⱼ - Tᵢ)/dᵢⱼ
```

**Convección (Ley de Newton):**
```
Q_conv = -h A_surf (Tᵢ - T∞)
```

**Condiciones de borde:**
- Base: Dirichlet (T = 90°C)
- Superficies expuestas: Convección uniforme (h = 300 W/m²·K)

---

## SLIDE 4: INTEGRACIÓN TEMPORAL

### Método Forward Euler Explícito

**Discretización temporal:**
```
Tᵢⁿ⁺¹ = Tᵢⁿ + Δt/(ρ cₚ V) × [Q_cond + Q_conv]
```

**Criterio de estabilidad:**
```
Δt ≤ Δt_max = 0.5 × (ρ cₚ V)/(Σ k A/d + h A_surf)
```

**Convergencia:**
```
||Tⁿ⁺¹ - Tⁿ||∞ < 10⁻⁴
```

**Características:**
- Esquema explícito de primer orden
- Factor de seguridad: 0.5
- Ajuste automático de Δt para estabilidad

---

## SLIDE 5: SOLVER ESTACIONARIO

### Sistema Lineal Directo

**Para optimización (más rápido que transitorio):**

En estado estacionario (∂T/∂t = 0):
```
0 = Σⱼ k Aᵢⱼ (Tⱼ - Tᵢ)/dᵢⱼ - h A_surf (Tᵢ - T∞)
```

**Sistema matricial [A]{T} = {b}:**
- Matriz dispersa (sparse) 36×36
- ~99% de ceros (solo 5 vecinos/nodo)
- Solver: scipy.sparse.linalg.spsolve (UMFPACK)
- Simétrico y definido positivo

**Ventajas:**
- Solución en 1 paso (no iterativo)
- 100× más rápido que solver transitorio
- Ideal para optimización paramétrica

---

## SLIDE 6: OPTIMIZACIÓN

### Búsqueda Exhaustiva Multi-dimensional

**Espacio de diseño (4D):**
| Parámetro | Rango | Valores |
|-----------|-------|---------|
| Espesor (t) | 5-10 μm | 15 |
| Altura (h) | 15-25 mm | 10 |
| Ahusamiento (tip_ratio) | 0.05-1.0 | 10 |
| Redondeo (round_factor) | 0.0-1.0 | 8 |

**Total: 12,000 diseños evaluados**

**Función objetivo:**
```
Minimizar: m_total = n_aletas × m_aleta
```

**Restricciones:**
1. Calor disipado: Q_fin > 0
2. Aletas necesarias: n_req = ⌈Q_required/Q_fin⌉
3. Factibilidad geométrica: n_req ≤ ⌊BASE_SIZE/(t+GAP)⌋

---

## SLIDE 7: PARÁMETROS FÍSICOS

### Propiedades del Material y Condiciones

**Aluminio:**
- Densidad: ρ = 2700 kg/m³
- Capacidad calorífica: cₚ = 900 J/kg·K
- Conductividad térmica: k = 205 W/m·K

**Condiciones operativas:**
- Coeficiente de convección: h = 300 W/m²·K
- Temperatura ambiente: T∞ = 45°C (peor caso)
- Temperatura base (CPU): T_base = 90°C
- Calor a disipar: Q_required = 500 W

**Geometría:**
- Base del disipador: 50 mm × 50 mm
- Gap entre aletas: 1 mm
- Longitud de aletas: 50 mm

---

## SLIDE 8: VALIDACIÓN

### Verificación de Balance Térmico

**Balance energético:**
```
Q_entrada (Base→Cuerpo) = Q_salida (Cuerpo→Ambiente)
```

**Cálculo:**
- Q_cond_in: Suma de flujos conductivos cruzando frontera Base→Cuerpo
- Q_conv_body: Suma de flujos convectivos del cuerpo al ambiente
- Error relativo: |Q_in - Q_out|/max(|Q_in|, |Q_out|) × 100%

**Criterio de aceptación:** Error < 1%

---

## DIAGRAMA DE FLUJO DEL MÉTODO

```
┌─────────────────────────────────────┐
│ 1. GEOMETRÍA (Q9)                   │
│    - Definir puntos de control      │
│    - Mapeo isoparamétrico           │
│    - Generar malla 11×11            │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 2. DISCRETIZACIÓN FVM               │
│    - Seleccionar nodos centrales    │
│    - Calcular vecindarios           │
│    - Computar áreas/volúmenes       │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 3. SOLVER TÉRMICO                   │
│    ┌─────────────┬─────────────┐    │
│    │ Transitorio │ Estacionario│    │
│    │ (Forward    │ (Sistema    │    │
│    │  Euler)     │  Lineal)    │    │
│    └─────────────┴─────────────┘    │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 4. POST-PROCESAMIENTO               │
│    - Calcular Q_disipado            │
│    - Verificar balance térmico      │
│    - Determinar masa y factibilidad │
└─────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│ 5. OPTIMIZACIÓN                     │
│    - Barrido paramétrico 4D         │
│    - Minimizar masa total           │
│    - Guardar diseño óptimo          │
└─────────────────────────────────────┘
```

---

## ECUACIONES CLAVE PARA SLIDES

### 1. Funciones de forma Q9 (producto tensorial)
```
Nᵢ(ξ,η) = Lₚ(ξ) · Mᵧ(η)

L₁ = ξ(ξ-1)/2    M₁ = η(η-1)/2
L₂ = 1 - ξ²      M₂ = 1 - η²
L₃ = ξ(ξ+1)/2    M₃ = η(η+1)/2
```

### 2. Balance FVM discreto
```
ρ cₚ Vᵢ (Tᵢⁿ⁺¹ - Tᵢⁿ)/Δt = Σⱼ k Aᵢⱼ (Tⱼⁿ - Tᵢⁿ)/dᵢⱼ - h A_surf,i (Tᵢⁿ - T∞)
```

### 3. Sistema estacionario
```
[A]{T} = {b}

Aᵢᵢ = -(Σⱼ k Aᵢⱼ/dᵢⱼ + h A_surf,i)
Aᵢⱼ = k Aᵢⱼ/dᵢⱼ
bᵢ = -h A_surf,i T∞
```

### 4. Calor total disipado
```
Q_fin = Σᵢ h A_surf,i (Tᵢ - T∞)
```

---

## PUNTOS DESTACADOS PARA ÉNFASIS VERBAL

1. **Uso de elementos Q9**: Permite geometrías complejas (ahusamiento y redondeo) con precisión cuadrática

2. **Doble solver**:
   - Transitorio para validación física
   - Estacionario para optimización rápida (100× más rápido)

3. **Estabilidad automática**: El código ajusta Δt automáticamente según el criterio de estabilidad

4. **Validación rigurosa**: Balance térmico verifica conservación de energía

5. **Optimización exhaustiva**: 12,000 diseños evaluados para encontrar el óptimo global

6. **Implementación eficiente**: Matrices dispersas (sparse) para reducir uso de memoria

---

## RECURSOS VISUALES SUGERIDOS

Para cada slide, considerar:

1. **Geometría Q9**: Diagrama del elemento con 9 nodos numerados
2. **Malla FVM**: Imagen de la malla con volúmenes de control coloreados
3. **Balance térmico**: Diagrama con flechas rojas (conducción) y azules (convección)
4. **Forward Euler**: Timeline mostrando pasos temporales
5. **Sistema lineal**: Visualización de la matriz sparse (patrón de sparsity)
6. **Optimización**: Gráfico 3D del espacio de diseño con óptimo marcado
7. **Resultados**: Heatmap de temperaturas de la geometría óptima

---

## ARCHIVOS DE REFERENCIA

| Concepto | Archivo | Función/Líneas |
|----------|---------|----------------|
| Q9 shape functions | finGridwCenters.py | líneas 10-47 |
| FVM grid | finGridwCenters.py | líneas 347-597 |
| Forward Euler | finEuler.py | líneas 71-193 |
| Steady solver | finEuler.py | líneas 318-441 |
| Optimization | optimumFinder.py | líneas 53-154 |
| Visualization | plots.py | líneas 50-180 |

---

## NOTAS PARA EL PRESENTADOR

- Enfatizar que FVM es una técnica industrial estándar
- Q9 es más sofisticado que elementos lineales (mayor precisión)
- Forward Euler es didáctico pero limitado por estabilidad
- El solver estacionario es la clave para optimización práctica
- 12,000 evaluaciones serían imposibles con solver transitorio
- Balance térmico confirma correctitud numérica del método
