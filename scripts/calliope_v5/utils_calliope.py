import pandas as pd
import numpy as np

# Función que convierte Index/MultiIndex a una Serie de strings y aplica .str.contains(pat).
# Entrada: index_like (Index o MultiIndex de pandas), pat (string/regex)
# Salida: boolean mask (np.array) del mismo largo que el índice.
def _to_str_mask(index_like, pat: str):
    if isinstance(index_like, pd.MultiIndex):
        str_idx = index_like.map(lambda tpl: '::'.join(map(str, tpl)))
    else:
        str_idx = index_like.astype(str)
    return pd.Series(str_idx).str.contains(pat, regex=True).values


# Función que extrae los labels de una dimensión de un DataArray y los convierte en un pandas.Index.
# Entrada: da (xarray.DataArray), dim_name (str)
# Salida: pandas.Index con los labels de la dimensión.
def _labels(da, dim_name):
    return da[dim_name].to_pandas()

# Función que selecciona elementos de una dimensión de un DataArray según un mask aplicado a sus labels y suma sus valores.
# Entrada: da (xarray.DataArray), dim_name (str), mask (función que recibe labels y devuelve booleanos)
# Salida: float con la suma de los valores seleccionados o 0.0 si la dimensión no existe.
def _sum_sel(da, dim_name, mask):
    labels = _labels(da, dim_name)
    lab_list = labels[mask(labels)].tolist()
    if dim_name in da.dims:
        return float(da.sel({dim_name: lab_list}).sum())
    return 0.0

# Función que suma los valores de energy_cap en un dataset filtrando por regex en el índice.
# Entrada: ds (xarray.Dataset con variable energy_cap), pat_regex (str, patrón regex)
# Salida: float con la suma de energy_cap para las coincidencias.
def _energy_cap(ds, pat_regex):
    s = ds.energy_cap.to_series()
    if s.empty:
        return 0.0
    mask = _to_str_mask(s.index, pat_regex)
    return float(s[mask].sum())

# Función que suma los valores de storage_cap en un dataset filtrando por regex en el índice.
# Entrada: ds (xarray.Dataset con variable storage_cap), pat_regex (str, patrón regex)
# Salida: float con la suma de storage_cap para las coincidencias.
def _storage_cap(ds, pat_regex):
    s = ds.storage_cap.to_series()
    if s.empty:
        return 0.0
    mask = _to_str_mask(s.index, pat_regex)
    return float(s[mask].sum())

# Función que suma los valores de cost en un dataset filtrando por regex en el índice.
# Entrada: ds (xarray.Dataset con variable cost), pat_regex (str, patrón regex)
# Salida: float con la suma de cost para las coincidencias.
def _cost_sum_by_pattern(ds, pat_regex):
    s = ds.cost.to_series()
    if s.empty:
        return 0.0
    mask = _to_str_mask(s.index, pat_regex)
    return float(s[mask].sum())

# Función que calcula indicadores clave (KPIs) de un modelo Calliope.
# Entrada: ds (xarray.Dataset con resultados de Calliope)
# Salida: dict con métricas de H2, capacidades, consumos, costos y LCOH.
def compute_kpis(ds):
    # --- H2 servido (demanda) ---
    da_con = ds.carrier_con
    dem_h2 = _sum_sel(
        da_con, 'loc_tech_carriers_con',
        lambda lab: lab.str.endswith('::hydrogen') & lab.str.contains('demand_h2')
    )
    h2_served = abs(dem_h2)  # demanda viene negativa en Calliope

    # --- H2 producido (electrolizador) ---
    da_prod = ds.carrier_prod
    h2_prod = _sum_sel(
        da_prod, 'loc_tech_carriers_prod',
        lambda lab: lab.str.endswith('::hydrogen') & lab.str.contains('electrolyzer')
    )

    # --- Consumos del electrolizador ---
    elec_to_e = _sum_sel(
        da_con, 'loc_tech_carriers_con',
        lambda lab: lab.str.endswith('::electricity') & lab.str.contains('electrolyzer')
    )
    water_to_e = _sum_sel(
        da_con, 'loc_tech_carriers_con',
        lambda lab: lab.str.endswith('::water') & lab.str.contains('electrolyzer')
    )

    # --- Capacidades instaladas ---
    cap_pv   = _energy_cap(ds, r'::pv$')
    cap_el   = _energy_cap(ds, r'electrolyzer')
    cap_line = _energy_cap(ds, r'ac_line')
    cap_h2st = _storage_cap(ds, r'h2_store')

    # --- Utilización del electrolizador (CF aproximado) ---
    n_hours = ds.dims.get('timesteps', 1)
    cf_el = (h2_prod / (cap_el * n_hours)) if (cap_el > 0 and n_hours > 0) else np.nan

    # --- Costos ---
    total_cost = float(ds.cost.sum())
    cost_pv    = _cost_sum_by_pattern(ds, r'::pv$')
    cost_el    = _cost_sum_by_pattern(ds, r'electrolyzer')
    cost_line  = _cost_sum_by_pattern(ds, r'ac_line')
    cost_des   = _cost_sum_by_pattern(ds, r'desalination')
    cost_h2st  = _cost_sum_by_pattern(ds, r'h2_store')
    cost_water = _cost_sum_by_pattern(ds, r'water_supply|seawater_supply')

    # --- LCOH ---
    LCOH_MWh = (total_cost / h2_served) if h2_served > 0 else np.nan
    LCOH_kg  = (LCOH_MWh / 33.33)       if h2_served > 0 else np.nan

    return {
        'H2_served_MWh': h2_served,
        'H2_produced_MWh': h2_prod,
        'Electrolyzer_consumption': {
            'electricity_MWh': abs(elec_to_e),
            'water_units':     abs(water_to_e)
        },
        'Capacity_installed': {
            'PV_MW': cap_pv,
            'Electrolyzer_MW': cap_el,
            'AC_line_MW': cap_line,
            'H2_storage_MWh': cap_h2st
        },
        'Electrolyzer_CF_approx': cf_el,
        'Cost_total_$': total_cost,
        'Cost_breakdown_$': {
            'PV': cost_pv,
            'Electrolyzer': cost_el,
            'AC_line': cost_line,
            'Desalination': cost_des,
            'H2_storage': cost_h2st,
            'Water_supply': cost_water
        },
        'LCOH_$per_MWh': LCOH_MWh,
        'LCOH_$per_kg':  LCOH_kg
    }



def pv_cf_mean(ds, loc="PV_SITE", tech="pv"):
    """Promedio del factor de capacidad PV a partir del recurso."""
    loc_tech = f"{loc}::{tech}"
    if "resource" in ds.data_vars:
        dim = "loc_techs_finite_resource"  # según tu ds
        if loc_tech in ds[dim]:
            da = ds["resource"].sel({dim: loc_tech})
            return float(da.mean().values)
    return np.nan



def lcoh_per_year(ds):
    """Calcula LCOH anual usando cost_var y producción de H2."""
    # --- Costos variables por timestep ---
    if "cost_var" not in ds:
        raise ValueError("El dataset no tiene 'cost_var'. Revisa save_results.")
    cost_ts = ds.cost_var.sel(costs="monetary").sum("loc_techs_om_cost")  # (timesteps,)

    # --- Producción de H2 por timestep ---
    # Filtrar solo las tecnologías que entregan hydrogen
    h2_locs = [k for k in ds.loc_tech_carriers_prod.values if k.endswith("::hydrogen")]
    h2_served_ts = ds.carrier_prod.sel(loc_tech_carriers_prod=h2_locs).sum("loc_tech_carriers_prod")

    # --- DataFrame y agrupación anual ---
    df = pd.DataFrame({
        "cost": cost_ts.values,
        "h2": h2_served_ts.values
    }, index=pd.to_datetime(ds.timesteps.values))

    annual = df.resample("Y").sum()
    annual["LCOH_$per_MWh"] = annual["cost"] / annual["h2"]
    annual["LCOH_$per_kg"] = annual["LCOH_$per_MWh"] / 33.33
    return annual
