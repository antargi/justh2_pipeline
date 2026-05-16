# 🌞 PROMPT: Bias Correction de RSDS (Radiación Solar) - ACCESS-CM2

## 📋 Contexto General

Necesito implementar **bias correction de radiación solar (RSDS)** para el modelo CMIP6 **ACCESS-CM2** usando **Quantile Mapping** contra observaciones de radiación solar del Valle de Aconcagua, Chile.

Los datos corregidos se usarán para:
1. Análisis de potencial solar fotovoltaico
2. Modelado energético con Calliope
3. Cálculo de indicadores de resiliencia climática
4. Proyecciones de producción de hidrógeno verde

---

## 🎯 Objetivo

Crear un **pipeline reproducible** de bias correction usando **xclim.sdba** (Quantile Mapping) que:
1. Corrija datos **historical** (para validación)
2. Aplique la misma corrección a **SSP245, SSP370, SSP585** (proyecciones futuras)
3. Preserve las **tendencias de cambio climático** (usar DQM si es posible, sino EQM)
4. Genere archivos NetCDF listos para análisis posterior

---

## 📊 Datos Disponibles

### 1. Observaciones (Referencia)
- **Fuente**: Solar.minenergia.cl (satelital)
- **Ruta**: `/home/aninotna/magister/tesis/justh2_pipeline/data/solar/solar_diario_grilla.zarr`
- **Variable**: `ghi` (Global Horizontal Irradiance)
- **Frecuencia**: Diaria
- **Período**: 2004-01-01 a 2016-12-30 (~13 años)
- **Región**: Valle de Aconcagua, Chile
  - Latitud: -33.23° a -32.28° (20 puntos, ~0.05° resolución)
  - Longitud: -71.88° a -70.03° (20 puntos, ~0.10° resolución)
- **Unidades**: W/m² (promedio diario)
- **Dimensiones**: `(date, lat, lon)` → renombrar `date` a `time`

### 2. CMIP6 ACCESS-CM2 - Historical
- **Ruta**: `/home/aninotna/magister/tesis/justh2_pipeline/data/cmip6/rsds/historical/`
- **Archivo**: `rsds_Amon_access_cm2_historical_r1i1p1f1_gn_185001-201412.nc`
- **Variable**: `rsds` (Surface Downwelling Shortwave Radiation)
- **Frecuencia**: Mensual (Amon)
- **Período**: 1850-01-16 a 2014-12-16
- **Unidades**: W/m²
- **Grilla**: Global (~1.25° x 1.875°)
- **Dimensiones**: `(time, lat, lon)`

### 3. CMIP6 ACCESS-CM2 - Escenarios SSP
- **Base**: `/home/aninotna/magister/tesis/justh2_pipeline/data/cmip6/rsds/`
- **SSP245**: `ssp245/access_cm2/rsds_Amon_ACCESS-CM2_ssp245_r1i1p1f1_gn_*.nc`
- **SSP370**: `ssp370/access_cm2/rsds_Amon_ACCESS-CM2_ssp370_r1i1p1f1_gn_*.nc`
- **SSP585**: `ssp585/access_cm2/rsds_Amon_ACCESS-CM2_ssp585_r1i1p1f1_gn_*.nc`
- **Período**: 2015-01-16 a 2100-12-16

---

## 🔧 Metodología Requerida

### Paso 1: Preparación de Observaciones
```python
# 1. Cargar zarr de observaciones GHI
obs = xr.open_zarr('/path/to/solar_diario_grilla.zarr')
ghi_daily = obs['ghi']

# 2. Renombrar dimensión temporal
if 'date' in ghi_daily.dims:
    ghi_daily = ghi_daily.rename({'date': 'time'})

# 3. Convertir a MENSUAL (promedio mensual de W/m²)
obs_monthly = ghi_daily.resample(time='MS').mean('time', skipna=True)

# 4. Renombrar variable para compatibilidad
obs_monthly = obs_monthly.rename('rsds')

# 5. Período de calibración (máximo traslape con CMIP6)
obs_calib = obs_monthly.sel(time=slice('2004-01-01', '2014-12-31'))
# Resultado: ~132 meses (11 años)
```

**Punto crítico**: Las observaciones están en W/m² promedio diario. Al convertir a mensual con `.mean()`, mantenemos las unidades correctas (W/m² promedio mensual).

### Paso 2: Preparación de CMIP6 Historical
```python
# 1. Cargar archivo(s) historical
hist_files = glob.glob('/path/to/rsds_Amon_access_cm2_historical*.nc')
hist = xr.open_mfdataset(hist_files, combine='by_coords')['rsds']

# 2. Ya es mensual (Amon), no necesita resample

# 3. Normalizar coordenadas
# - Renombrar latitude/longitude → lat/lon si es necesario
# - Convertir lon de [0, 360] a [-180, 180] si es necesario
# - Ordenar lat ascendente

# 4. Periodo de calibración (mismo que observaciones)
hist_calib = hist.sel(time=slice('2004-01-01', '2014-12-31'))

# 5. REGRIDDING a la grilla de observaciones
# Usar xESMF o interpolación bilineal
hist_regrid = hist_calib.interp(
    lat=obs_calib.lat,
    lon=obs_calib.lon,
    method='linear'
)
```

**Punto crítico**: El regridding debe hacerse **antes** de entrenar el Quantile Mapping. Ambos datasets deben tener la misma grilla espacial.

### Paso 3: Alineación Temporal y Manejo de Calendarios
```python
# CMIP6 puede usar calendarios no-estándar (360_day, noleap, etc.)
# Las observaciones usan calendario estándar (gregorian)

# Opción 1: Usar cftime para manejar calendarios diferentes
import cftime

# Opción 2: Convertir a índice mensual común (YYYY-MM-01)
# Esto ignora el día exacto y solo usa año-mes

def harmonize_monthly_time(ref, hist):
    """
    Alinea tiempo mensual usando solo año-mes, 
    ignorando días y calendarios.
    """
    # Extraer año-mes de ambos
    ref_ym = [(y, m) for y, m in zip(ref.time.dt.year.values, 
                                      ref.time.dt.month.values)]
    hist_ym = [(y, m) for y, m in zip(hist.time.dt.year.values, 
                                       hist.time.dt.month.values)]
    
    # Encontrar intersección
    common = sorted(set(ref_ym) & set(hist_ym))
    
    # Seleccionar índices correspondientes
    ref_idx = [i for i, ym in enumerate(ref_ym) if ym in common]
    hist_idx = [i for i, ym in enumerate(hist_ym) if ym in common]
    
    ref_aligned = ref.isel(time=ref_idx)
    hist_aligned = hist.isel(time=hist_idx)
    
    # Reasignar índice temporal común (datetime64)
    new_time = pd.to_datetime([f"{y:04d}-{m:02d}-01" for y, m in common])
    ref_aligned = ref_aligned.assign_coords(time=('time', new_time))
    hist_aligned = hist_aligned.assign_coords(time=('time', new_time))
    
    return ref_aligned, hist_aligned

# Aplicar
obs_aligned, hist_aligned = harmonize_monthly_time(obs_calib, hist_regrid)
```

**Punto crítico**: Los calendarios diferentes entre obs y CMIP6 pueden causar problemas. Alinear por año-mes es robusto.

### Paso 4: Enmascarar Celdas con Datos Insuficientes
```python
def mask_insufficient_data(ref, hist, min_months=60, min_per_month=6):
    """
    Enmascara celdas con datos insuficientes.
    - min_months: mínimo de meses válidos en total
    - min_per_month: mínimo por cada mes del año (1-12)
    """
    # Criterio total
    ok_total = ((ref.notnull().sum('time') >= min_months) & 
                (hist.notnull().sum('time') >= min_months))
    
    # Criterio por mes del año
    ref_by_month = ref.notnull().groupby('time.month').sum('time')
    hist_by_month = hist.notnull().groupby('time.month').sum('time')
    ok_monthly = ((ref_by_month >= min_per_month).all('month') & 
                  (hist_by_month >= min_per_month).all('month'))
    
    # Combinar
    mask = ok_total & ok_monthly
    
    return ref.where(mask), hist.where(mask)

# Aplicar
obs_masked, hist_masked = mask_insufficient_data(obs_aligned, hist_aligned)
```

**Punto crítico**: Esto evita entrenar QM en celdas con datos esporádicos.

### Paso 5: Chunking para Dask (Eficiencia)
```python
def chunk_for_sdba(da, max_xy=50):
    """
    Rechunk para SDBA: tiempo debe ser un solo chunk.
    """
    chunks = {
        'time': -1,  # TODO el tiempo en un chunk
        'lat': min(da.sizes['lat'], max_xy),
        'lon': min(da.sizes['lon'], max_xy)
    }
    return da.chunk(chunks)

obs_chunked = chunk_for_sdba(obs_masked)
hist_chunked = chunk_for_sdba(hist_masked)
```

**Punto crítico**: xclim.sdba requiere que la dimensión `time` esté en un solo chunk.

### Paso 6: Entrenar Quantile Mapping
```python
from xclim import sdba

# Configurar agrupación (por mes del año)
grouper = sdba.Grouper('time.month')

# Opción 1: Detrended Quantile Mapping (DQM) - PREFERIDO
# Preserva tendencias de cambio climático
try:
    QM = sdba.DetrendedQuantileMapping.train(
        ref=obs_chunked,      # Observaciones (GHI mensual)
        hist=hist_chunked,    # CMIP6 historical regridded
        nquantiles=50,        # Número de cuantiles
        group=grouper,        # Agrupar por mes
        kind='+'              # Tipo de tendencia (aditiva para temperatura/radiación)
    )
    method_used = 'DQM'
    print("✅ DQM entrenado exitosamente")
except Exception as e:
    print(f"⚠️ DQM falló: {e}")
    print("   Usando EQM como alternativa...")
    
    # Opción 2: Empirical Quantile Mapping (EQM) - FALLBACK
    QM = sdba.EmpiricalQuantileMapping.train(
        ref=obs_chunked,
        hist=hist_chunked,
        nquantiles=50,
        group=grouper
    )
    method_used = 'EQM'
    print("✅ EQM entrenado exitosamente")
```

**Puntos críticos**:
- `ref=` debe ser las **observaciones** (GHI)
- `hist=` debe ser el **modelo** a corregir (CMIP6 historical)
- `nquantiles=50` es un buen balance (más cuantiles = más detalle, más lento)
- `group='time.month'` entrena parámetros diferentes para cada mes
- DQM es mejor que EQM porque preserva tendencias

### Paso 7: Aplicar Corrección a Historical (Validación)
```python
# Aplicar a historical completo (no solo periodo calibración)
hist_full = xr.open_mfdataset(hist_files)['rsds']
hist_full_regrid = hist_full.interp(lat=obs_calib.lat, lon=obs_calib.lon)
hist_full_chunked = chunk_for_sdba(hist_full_regrid)

# Ajustar
hist_corrected = QM.adjust(
    sim=hist_full_chunked,
    extrapolation='constant'  # Manejar valores fuera del rango de entrenamiento
)

# Asegurar no-negatividad (radiación no puede ser negativa)
hist_corrected = hist_corrected.clip(min=0)

# Guardar
hist_corrected.attrs['units'] = 'W m-2'
hist_corrected.attrs['bias_correction_method'] = method_used
hist_corrected.attrs['bias_correction_period'] = '2004-2014'
hist_corrected.attrs['reference_data'] = 'GHI solar.minenergia.cl'

output_path = '/path/to/rsds_ACCESS-CM2_historical_bias_corrected.nc'
encoding = {
    'rsds': {
        'dtype': 'float32',
        '_FillValue': -9999.0,
        'zlib': True,
        'complevel': 4
    }
}
hist_corrected.to_netcdf(output_path, encoding=encoding)
```

**Punto crítico**: El objeto `QM` entrenado se puede reutilizar para corregir otros períodos (SSPs).

### Paso 8: Aplicar Corrección a Escenarios SSP
```python
# Función para procesar cada SSP
def correct_ssp(ssp_name, qm_model):
    """
    Aplica bias correction a un escenario SSP.
    """
    # Cargar SSP
    ssp_pattern = f'/path/to/ssp{ssp_name}/access_cm2/rsds_Amon_ACCESS-CM2_ssp{ssp_name}*.nc'
    ssp_files = glob.glob(ssp_pattern)
    ssp = xr.open_mfdataset(ssp_files)['rsds']
    
    # Regridding
    ssp_regrid = ssp.interp(lat=obs_calib.lat, lon=obs_calib.lon)
    
    # Chunking
    ssp_chunked = chunk_for_sdba(ssp_regrid)
    
    # Ajustar (usando el MISMO modelo QM entrenado con historical)
    ssp_corrected = qm_model.adjust(sim=ssp_chunked, extrapolation='constant')
    
    # No-negatividad
    ssp_corrected = ssp_corrected.clip(min=0)
    
    # Metadata
    ssp_corrected.attrs['units'] = 'W m-2'
    ssp_corrected.attrs['bias_correction_method'] = method_used
    ssp_corrected.attrs['bias_correction_trained_on'] = 'historical 2004-2014'
    
    # Guardar
    output = f'/path/to/rsds_ACCESS-CM2_ssp{ssp_name}_bias_corrected.nc'
    encoding = {'rsds': {'dtype': 'float32', '_FillValue': -9999.0, 
                         'zlib': True, 'complevel': 4}}
    ssp_corrected.to_netcdf(output, encoding=encoding)
    
    return ssp_corrected

# Aplicar a todos los SSPs
ssp245_bc = correct_ssp('245', QM)
ssp370_bc = correct_ssp('370', QM)
ssp585_bc = correct_ssp('585', QM)
```

**Punto crítico**: El **mismo** modelo QM se usa para todos los SSPs. Esto preserva las diferencias entre escenarios.

### Paso 9: Validación
```python
# Comparar historical original vs corregido vs observaciones
# (solo periodo de calibración)

obs_val = obs_aligned.sel(time=slice('2004', '2014'))
hist_orig_val = hist_aligned.sel(time=slice('2004', '2014'))
hist_corr_val = hist_corrected.sel(time=slice('2004', '2014'))

# Estadísticos
print("📊 Validación (2004-2014):")
print(f"Observaciones - Media: {float(obs_val.mean()):.2f} W/m²")
print(f"Original - Media: {float(hist_orig_val.mean()):.2f} W/m²")
print(f"Corregido - Media: {float(hist_corr_val.mean()):.2f} W/m²")

# Sesgo
orig_bias = float(hist_orig_val.mean() - obs_val.mean())
corr_bias = float(hist_corr_val.mean() - obs_val.mean())
print(f"\nSesgo original: {orig_bias:+.2f} W/m²")
print(f"Sesgo corregido: {corr_bias:+.2f} W/m²")
print(f"Reducción: {100*(1 - abs(corr_bias)/abs(orig_bias)):.1f}%")

# Gráfico
fig, ax = plt.subplots(figsize=(12, 5))
obs_val.mean(['lat','lon']).plot(ax=ax, label='Obs', color='black', lw=2)
hist_orig_val.mean(['lat','lon']).plot(ax=ax, label='Original', color='red', alpha=0.7)
hist_corr_val.mean(['lat','lon']).plot(ax=ax, label='Corregido', color='blue')
ax.legend()
ax.set_title('Validación Bias Correction - RSDS (2004-2014)')
plt.show()
```

---

## 📂 Estructura de Archivos de Salida

```
/home/aninotna/magister/tesis/justh2_pipeline/data/cmip6/rsds/bias_corrected_qm/
├── rsds_ACCESS-CM2_historical_bias_corrected_2004-2014.nc
├── rsds_ACCESS-CM2_ssp245_bias_corrected_2015-2100.nc
├── rsds_ACCESS-CM2_ssp370_bias_corrected_2015-2100.nc
├── rsds_ACCESS-CM2_ssp585_bias_corrected_2015-2100.nc
└── bias_correction_metadata.json  # Metadatos del proceso
```

**Metadata JSON sugerido**:
```json
{
  "model": "ACCESS-CM2",
  "variable": "rsds",
  "reference_data": "GHI solar.minenergia.cl",
  "reference_period": "2004-01-01 to 2014-12-31",
  "method": "DetrendedQuantileMapping",
  "nquantiles": 50,
  "grouping": "time.month",
  "training_duration": "132 months",
  "region": "Valle de Aconcagua, Chile",
  "lat_range": [-33.23, -32.28],
  "lon_range": [-71.88, -70.03],
  "output_grid": "obs_grid (0.05° x 0.10°)",
  "scenarios_corrected": ["historical", "ssp245", "ssp370", "ssp585"],
  "date_created": "2025-10-19",
  "created_by": "bias_correction_pipeline.ipynb"
}
```

---

## ⚠️ Problemas Comunes y Soluciones

### Problema 1: NaN en las diferencias
**Causa**: Grillas no alineadas
**Solución**: Verificar que regridding se hizo correctamente y que las coordenadas son idénticas

### Problema 2: Datos corregidos idénticos al original
**Causa**: 
- No se llamó `.adjust()`
- Se guardó el archivo equivocado
- El modelo QM no se entrenó correctamente

**Solución**: Verificar paso a paso el pipeline

### Problema 3: Errores de calendario
**Causa**: CMIP6 usa calendario 360_day o noleap
**Solución**: Usar función `harmonize_monthly_time()` para alinear por año-mes

### Problema 4: "time dimension must be a single chunk"
**Causa**: Dask chunking inadecuado
**Solución**: Usar `chunk_for_sdba()` antes de `.train()` y `.adjust()`

### Problema 5: Valores negativos en radiación
**Causa**: QM puede generar valores ligeramente negativos
**Solución**: Aplicar `.clip(min=0)` después de la corrección

---

## ✅ Checklist de Verificación

Antes de dar por válido el bias correction, verificar:

- [ ] Las observaciones están en escala mensual
- [ ] Historical y observaciones tienen el mismo período (2004-2014)
- [ ] Ambos datasets tienen la misma grilla espacial
- [ ] Los calendarios están alineados
- [ ] El chunking es correcto (time = un solo chunk)
- [ ] El modelo QM se entrenó sin errores
- [ ] La diferencia abs media (corregido - original) > 0.1 W/m²
- [ ] El sesgo se redujo (|bias_corregido| < |bias_original|)
- [ ] No hay valores negativos en RSDS corregido
- [ ] Los archivos NetCDF tienen metadatos completos
- [ ] El mismo modelo QM se usó para todos los SSPs

---

## 📝 Notas Finales

1. **Unidades**: GHI observado y RSDS de CMIP6 ambos están en W/m². Al convertir de diario a mensual, usar `.mean()` mantiene las unidades correctas.

2. **Resolución espacial**: Los datos corregidos quedarán en la grilla de observaciones (~0.05° x 0.10°), que es mucho más fina que la grilla original de CMIP6 (~1.25° x 1.875°). Esto es deseable para análisis local.

3. **Preservación de tendencias**: DQM preserva la señal de cambio climático (tendencias). Por eso es preferible a EQM cuando sea posible.

4. **Reutilización**: El objeto `QM` entrenado puede guardarse como pickle o zarr para reutilización posterior sin re-entrenar.

5. **Validación cruzada**: Idealmente, entrenar con 2004-2009 y validar con 2010-2014 para evaluar generalización. Pero con solo 11 años de datos, es mejor usar todo el período para entrenar.

---

## 🚀 Siguiente Paso

Con los datos RSDS bias-corrected, podrás:
1. Calcular **capacity factors** de PV solar
2. Generar **series temporales de generación** para Calliope
3. Calcular **indicadores de variabilidad** solar
4. Evaluar **impactos de cambio climático** en potencial solar

¿Necesitas ayuda implementando este pipeline?
