# Cálculo de Producción de H2V Espacial con Calliope

## 📋 Descripción

Este notebook ejecuta el modelo **Calliope** punto por punto en la grilla del Valle de Aconcagua para calcular la producción óptima de hidrógeno verde considerando:

- ✅ **Optimización económica** (minimiza costos totales del sistema)
- ✅ **Capacidades óptimas** de PV, electrolizador y almacenamiento
- ✅ **LCOH** (Levelized Cost of Hydrogen) calculado con CAPEX + OPEX
- ✅ **Balance energético** completo: PV → Electrolyzer → H2 Storage → Demand
- ✅ **Validación espacial** por escenario y año

## 🔧 Diferencia con Enfoque Directo

### ❌ Enfoque directo (NO usado):
```python
# Cálculo simplificado sin optimización
E_pv = CF × P_nominal × horas
H2 = E_pv / SEC
```
**Problemas**: No optimiza capacidades, no considera almacenamiento, no calcula LCOH real.

### ✅ Enfoque con Calliope (USADO):
```python
# Optimización completa del sistema
model = calliope.Model('model_config.yml')
model.run()  # Resuelve LP para minimizar costos
kpis = compute_kpis(model.results)
```
**Ventajas**: 
- Calcula capacidades óptimas PV/Electrolyzer/Storage
- Minimiza LCOH considerando CAPEX, OPEX, lifetime
- Balancea energía temporalmente con almacenamiento
- Respeta constraints técnicos del electrolizador

## 🚀 Uso

### 1. Ejecución rápida (1 escenario, 1 año)
```bash
cd /home/aninotna/magister/tesis/justh2_pipeline/scripts/calliope_v6
jupyter notebook calculate_h2v_production_spatial.ipynb
```

Ejecutar celdas 1-15. Procesa ~400 puntos en ~2-4 horas.

**Configuración actual**:
- Escenario: `ssp245`
- Año: `2030`
- Grilla: 20×20 puntos

### 2. Ejecución completa (todos los escenarios)

Descomentar celda 16 y ejecutar:
- Escenarios: `ssp245`, `ssp370`, `ssp585`
- Años: `2030`, `2050`, `2070`, `2100`
- **⚠️ Tiempo estimado**: 24-48 horas para ~1600 puntos × 12 configuraciones

## 📊 Outputs

### Archivos generados (por escenario/año):

1. **CSV completo**:
   ```
   h2v_calliope_results_ssp245_2030.csv
   ```
   Columnas: `lat`, `lon`, `h2_prod_kg`, `h2_prod_ton`, `cap_pv_mw`, `cap_electrolyzer_mw`, `lcoh_usd_per_kg`, `water_m3`, `cf_electrolyzer`, etc.

2. **NetCDF espacial**:
   ```
   h2v_calliope_results_ssp245_2030.nc
   ```
   Variables: `h2_prod_ton(lat, lon)`, `lcoh_usd_per_kg(lat, lon)`, `cap_pv_mw(lat, lon)`

3. **GeoJSON** (para mapas web):
   ```
   h2v_calliope_results_ssp245_2030.geojson
   ```

4. **Mapas PNG**:
   ```
   h2v_calliope_spatial_maps_ssp245_2030.png
   ```
   4 paneles: Producción H2, Cap. PV, LCOH, Consumo Agua

5. **Estadísticas**:
   ```
   h2v_calliope_statistics_ssp245_2030.csv
   ```
   Media, mediana, std, min, max de todas las variables

## 🔍 Resultados Esperados

### Rangos típicos (Valle de Aconcagua, CF ~22%):

| Variable | Rango Esperado |
|----------|----------------|
| Producción H2 | 3-5 ton/MWp/año |
| Cap. PV óptima | 0.5-2 MW |
| Cap. Electrolyzer | 0.3-1.5 MW |
| LCOH | $2.5-4.5/kg H2 |
| Consumo agua | 30-50 m³/MWp/año |
| CF Electrolyzer | 18-25% |

## 📝 Interpretación

### LCOH más bajo indica:
- ✅ Mejor recurso solar (mayor CF)
- ✅ Mejor economía de escala
- ✅ Ubicación óptima para planta H2V

### Producción H2 alta indica:
- ✅ Alto potencial solar
- ✅ Capacidades instaladas mayores (si LCOH es competitivo)

### Validaciones automáticas:
- ✓ Balance energético: `E_pv >= E_electrolyzer`
- ✓ Consumo agua: `~9 L/kg H2`
- ✓ CF electrolyzer: `≈ CF_pv × η_sistema`
- ✓ LCOH: considera CAPEX, OPEX, lifetime, interés

## 🔗 Integración con Calliope

El notebook usa la configuración existente en `calliope_v6/`:

- `model_config.yml`: configuración general
- `techs.yml`: parámetros PV, electrolizador, storage
- `locations.yml`: locaciones (se modifica dinámicamente por punto)
- `utils_calliope.py`: funciones `compute_kpis()`

### Flujo de datos:
```
PV CF (NetCDF) 
  → Punto (lat, lon) 
    → CSV temporal 
      → Calliope Model 
        → Optimización LP 
          → Results (xarray) 
            → compute_kpis() 
              → DataFrame 
                → Export
```

## 🎯 Próximos Pasos

1. **Validación multi-año**: Ejecutar celda 16 para todos los escenarios
2. **Análisis de tendencias**: Comparar 2030 vs 2050 vs 2100
3. **Sensibilidad económica**: Variar CAPEX/OPEX del electrolizador
4. **Integración hídrica**: Cruzar con disponibilidad de agua real
5. **Conflictos territoriales**: Superponer con datos de MapBioma/INDH

## 📚 Referencias

- [Calliope Documentation](https://calliope.readthedocs.io/)
- `techs.yml`: Parámetros PEM basados en literatura 2023-2024
- `utils_calliope.py`: Funciones de extracción de KPIs

## ⚠️ Notas Importantes

1. **Tiempo de ejecución**: Cada punto toma ~10-30 segundos (Calliope LP solver)
2. **Memoria**: Usar máquina con >8GB RAM para grillas grandes
3. **Solver**: Usa CBC por defecto (gratuito), puede cambiar a Gurobi para velocidad
4. **Chunks**: Procesar por bloques si la grilla es muy grande (>50×50)
5. **Validar**: Siempre revisar que no haya errores en la optimización (infeasibilities)

## 🐛 Troubleshooting

### Error: "Model infeasible"
- Revisar demanda H2 vs capacidad PV/electrolyzer
- Verificar que CF no sea todo ceros/NaN

### Error: "Solver not found"
- Instalar CBC: `conda install -c conda-forge coincbc`

### Error: "Memory error"
- Reducir tamaño de grilla o procesar por bloques
- Usar subset temporal más pequeño

---

**Autor**: Calliope v6 Pipeline - H2V Valle de Aconcagua  
**Fecha**: Octubre 2025  
**Contacto**: Ver repositorio principal
