"""
Pipeline completo de perfilado y visualización de clusters con etiquetas climáticas.

Este módulo encapsula TODO el flujo del notebook 07_experiments_2_cluster_profiling_1511.ipynb:
1. Cálculo de importancias de variables por cluster
2. Etiquetado climático automático (temperatura, precipitación, extremos)
3. Overlay de información energética (H₂ producción)
4. Desambiguación de clusters con etiquetas similares
5. Visualización espacial con mapas interpretativos

Uso típico:
-----------
from cluster_profiling_pipeline import ClusterProfilingPipeline

# Inicializar pipeline con modelos entrenados
pipeline = ClusterProfilingPipeline(
    models_dict=MODELS,  # {'AE': model_ae, 'VAE': model_vae}
    k_clusters=10
)

# Generar perfiles completos para todos los escenarios
results = pipeline.run_full_profiling(
    data_blocks={'BASE': X_BASE, 'SSP245': X245, ...},
    coords_df=coords_df,
    feature_names=feature_names
)

# Visualizar espacialmente
pipeline.plot_all_scenarios(results, save_dir='plots/')
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.cm import ScalarMappable
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
import re
warnings.filterwarnings('ignore')


class ClusterProfilingPipeline:
    """
    Pipeline completo para perfilado y visualización de clusters con etiquetas climáticas.
    
    Parámetros:
    -----------
    models_dict : dict
        Diccionario con modelos entrenados {'AE': model, 'VAE': model}
    k_clusters : int
        Número de clusters para KMeans
    feature_names : list[str], opcional
        Nombres de variables (se puede pasar después)
    device : str
        Dispositivo PyTorch ('cuda' o 'cpu')
    """
    
    def __init__(
        self,
        models_dict: Dict[str, nn.Module],
        k_clusters: int = 10,
        feature_names: Optional[List[str]] = None,
        device: str = 'cpu'
    ):
        self.models_dict = models_dict
        self.k_clusters = k_clusters
        self.feature_names = feature_names
        self.device = torch.device(device)
        
        # Diccionarios para resultados
        self.latents = {}
        self.labels = {}
        self.importances = {}
        self.cluster_profiles = {}
        
        # Configuración para etiquetado climático
        self._setup_climate_patterns()
        self._setup_h2_detection()
    
    def _setup_climate_patterns(self):
        """Configura patrones regex para identificar variables climáticas."""
        self.PAT = {
            "tmax": re.compile(r"(tmax|tx|tasmax)", re.I),
            "tmin": re.compile(r"(tmin|tn|tasmin)", re.I),
            "pr":   re.compile(r"(^pr$|precip|prcptot|rain|ppt)", re.I),
            "rx1":  re.compile(r"(rx1(day)?)", re.I),
            "rx5":  re.compile(r"(rx5(day)?)", re.I),
            "r95":  re.compile(r"(r95p)", re.I),
            "r99":  re.compile(r"(r99p)", re.I),
            "sdii": re.compile(r"(sdii)", re.I),
            "cdd":  re.compile(r"(cdd)", re.I),
            "spi":  re.compile(r"(spi)", re.I),
            "var":  re.compile(r"(std|var|variab|cv)", re.I),
        }
        
        self.TH = {
            "high":  0.6,
            "low":  -0.6,
            "very":  1.1,
            "mid":   0.4,
        }
    
    def _setup_h2_detection(self):
        """Configura detección de columnas de H₂."""
        self.rx_h2_mean = re.compile(r"^calliope_h2_prod_ton_decadal_mean_(\d{4})$", re.I)
        self.rx_h2_stdT = re.compile(r"^calliope_h2_prod_ton_std_T$", re.I)
        self.decasel = {"BASE": "2020", "T245": "2080", "T370": "2080", "T585": "2080"}
    
    def encode_to_latent(self, model_key: str, X_data: np.ndarray) -> np.ndarray:
        """Encodea datos al espacio latente usando el modelo especificado."""
        model = self.models_dict[model_key]
        model.eval()
        
        X_tensor = torch.FloatTensor(X_data).to(self.device)
        
        with torch.no_grad():
            if model_key == 'VAE':
                _, mu, _ = model(X_tensor)
                Z = mu.cpu().numpy()
            else:  # AE
                Z = model.encoder(X_tensor).cpu().numpy()
        
        return Z
    
    def apply_clustering(self, Z_latent: np.ndarray) -> Tuple[np.ndarray, KMeans]:
        """Aplica KMeans al espacio latente escalado."""
        # Escalar espacio latente
        scaler = StandardScaler()
        Z_scaled = scaler.fit_transform(Z_latent)
        
        # KMeans
        kmeans = KMeans(
            n_clusters=self.k_clusters,
            random_state=42,
            n_init=50,
            max_iter=500
        )
        labels = kmeans.fit_predict(Z_scaled)
        
        return labels, kmeans
    
    def calculate_feature_importance(
        self,
        X_data: np.ndarray,
        labels: np.ndarray
    ) -> np.ndarray:
        """
        Calcula importancia de variables por cluster usando Random Forest.
        
        Returns:
        --------
        imp_matrix : np.ndarray (k_clusters, n_features)
            Matriz de importancias estandarizadas (z-scores)
        """
        n_features = X_data.shape[1]
        imp_matrix = np.zeros((self.k_clusters, n_features))
        
        for k in range(self.k_clusters):
            # Crear target binario: este cluster vs. resto
            y_binary = (labels == k).astype(int)
            
            if y_binary.sum() < 10:  # Cluster muy pequeño
                continue
            
            # Entrenar Random Forest
            rf = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
            rf.fit(X_data, y_binary)
            
            # Importancias crudas
            imp_raw = rf.feature_importances_
            
            # Estandarizar a z-scores
            mean_imp = imp_raw.mean()
            std_imp = imp_raw.std()
            if std_imp > 1e-8:
                imp_matrix[k, :] = (imp_raw - mean_imp) / std_imp
            else:
                imp_matrix[k, :] = imp_raw
        
        return imp_matrix
    
    def find_cols(self, names: List[str], rx: re.Pattern) -> List[int]:
        """Encuentra índices de columnas que coinciden con un patrón regex."""
        return [i for i, c in enumerate(names) if rx.search(c)]
    
    def mean_group(self, imp_row: np.ndarray, idxs: List[int]) -> float:
        """Calcula media de importancias de un grupo de variables."""
        if not idxs:
            return np.nan
        vals = imp_row[idxs]
        return float(np.nanmean(vals)) if len(vals) else np.nan
    
    def tags_from_importance_row(self, imp_row: np.ndarray, GROUPS: Dict) -> List[str]:
        """
        Genera etiquetas climáticas desde un vector de importancias.
        Evita contradicciones (e.g., no "cálido" y "frío" simultáneamente).
        """
        tags = []
        
        # Calcular importancias promedio por grupo
        z_tmax = self.mean_group(imp_row, GROUPS["tmax"])
        z_tmin = self.mean_group(imp_row, GROUPS["tmin"])
        z_pr   = self.mean_group(imp_row, GROUPS["pr"])
        z_rx   = np.nanmean([self.mean_group(imp_row, GROUPS["rx1"]),
                            self.mean_group(imp_row, GROUPS["rx5"])])
        z_r95  = self.mean_group(imp_row, GROUPS["r95"])
        z_r99  = self.mean_group(imp_row, GROUPS["r99"])
        z_sdii = self.mean_group(imp_row, GROUPS["sdii"])
        z_cdd  = self.mean_group(imp_row, GROUPS["cdd"])
        z_spi  = self.mean_group(imp_row, GROUPS["spi"])
        z_var  = self.mean_group(imp_row, GROUPS["var"])
        
        # === Temperatura (mutuamente excluyente) ===
        # Tomamos el que tenga mayor |z| entre tmax y tmin para decidir signo
        z_temp_vals = [v for v in [z_tmax, z_tmin] if not np.isnan(v)]
        if z_temp_vals:
            z_temp = max(z_temp_vals, key=lambda v: abs(v))
            if z_temp >= self.TH["high"]:
                tags.append("cálido")
            elif z_temp <= self.TH["low"]:
                tags.append("frío")
            # (si cae entre [-0.6, 0.6] no etiquetamos temperatura)
        
        # === Precipitación (mutuamente excluyente) ===
        if not np.isnan(z_pr):
            if z_pr >= self.TH["high"]:
                tags.append("húmedo")
            elif z_pr <= self.TH["low"]:
                tags.append("seco")
        
        # === Sequía/persistencia seca ===
        if not np.isnan(z_cdd) and z_cdd >= self.TH["high"]:
            tags.append("sequía/prolongado")
        
        # === Extremos de precipitación (cualquiera alto activa) ===
        z_extreme = np.nanmax([z for z in [z_rx, z_r95, z_r99, z_sdii] if not np.isnan(z)]) \
                    if any([not np.isnan(z) for z in [z_rx, z_r95, z_r99, z_sdii]]) else np.nan
        if not np.isnan(z_extreme) and z_extreme >= self.TH["high"]:
            tags.append("extremos de lluvia")
        
        # === SPI (negativo = seco relativo) ===
        if not np.isnan(z_spi) and z_spi <= self.TH["low"]:
            tags.append("anomalía seca (SPI)")
        
        # === Variabilidad (si aplicara; ya excluiste *_std_T del ranking principal) ===
        if not np.isnan(z_var) and z_var >= self.TH["high"]:
            tags.append("alta variabilidad")
        
        # === Etiquetas compuestas: REEMPLAZAR básicas por compuestas (evita redundancia) ===
        # Crear etiquetas compuestas y remover componentes básicos si corresponde
        temp_tag = None
        prec_tag = None
        
        # Identificar etiquetas de temperatura y precipitación
        if "cálido" in tags:
            temp_tag = "cálido"
        elif "frío" in tags:
            temp_tag = "frío"
        
        if "seco" in tags:
            prec_tag = "seco"
        elif "húmedo" in tags:
            prec_tag = "húmedo"
        
        # Si hay ambas dimensiones (temp + precip), crear compuesta y remover básicas
        if temp_tag and prec_tag:
            compound_tag = f"{temp_tag}-{prec_tag}"
            # Remover las básicas
            tags = [t for t in tags if t not in [temp_tag, prec_tag]]
            # Agregar la compuesta
            tags.append(compound_tag)
        
        # === Limpieza duplicados conservando orden ===
        out, seen = [], set()
        for t in tags:
            if t not in seen:
                out.append(t)
                seen.add(t)
        
        return out if out else ["perfil neutro/mixto"]
    
    def simplify_variable_name(self, var_name: str) -> str:
        """Simplifica nombres de variables para etiquetas más concisas."""
        # Remover prefijos comunes PRIMERO
        var_name = var_name.replace("historical_", "").replace("ssp245_", "").replace("ssp370_", "").replace("ssp585_", "")
        var_name = var_name.replace("climate_", "").replace("topo_", "").replace("landuse_", "")
        
        # CASO ESPECIAL H₂: aplicar reglas específicas PRIMERO (antes de reglas genéricas)
        if "calliope_h2_prod_ton_decadal_mean" in var_name:
            return "H₂prod"
        if "calliope_h2_prod_ton_std_T" in var_name:
            return "H₂var"
        
        # Mapeo de nombres técnicos a nombres ULTRA-CORTOS para etiquetas en mapas
        simplifications = {
            "tasmax": "Tmax",
            "tmax": "Tmax",
            "tasmin": "Tmin",
            "tmin": "Tmin",
            "pr_prcptot": "prec",
            "prcptot": "prec",
            "pr": "prec",
            "rx1day": "lluvMax1d",
            "rx5day": "lluvMax5d",
            "r95ptot": "prec95",
            "r99ptot": "prec99",
            "sdii": "intLluv",
            "cdd": "díasSecos",
            "spi": "SPI",
            "elevation": "elev",
            "slope": "pend",
            "aspect": "orient",
            "water_glacier": "glaciar",
            "urban_infra": "urbano",
            "agriculture": "agric",
            "forest": "bosque",
            "grassland": "pasto",
            "shrubland": "matorral",
            "restricted": "restric",
            "calliope_h2": "H₂",
            "_decadal_mean": "",
            "_mean_decadal": "",
            "_decadal": "",
            "_dec_": "",
            "_std_T": "var",
        }
        
        # Aplicar simplificaciones
        var_simplified = var_name
        for pattern, replacement in simplifications.items():
            var_simplified = var_simplified.replace(pattern, replacement)
        
        # Limpiar underscores y guiones al inicio/final
        var_simplified = var_simplified.strip('_-')
        
        # Truncar si aún es muy largo (para casos extremos)
        if len(var_simplified) > 12:
            var_simplified = var_simplified[:10] + ".."
        
        return var_simplified
    
    def get_top_drivers_summary(self, imp_row: np.ndarray, feature_names: List[str], top_n: int = 3) -> List[str]:
        """Extrae top N variables más influyentes con dirección (↑/↓)."""
        abs_imp = np.abs(imp_row)
        top_indices = np.argsort(abs_imp)[::-1][:top_n]
        
        drivers = []
        for idx in top_indices:
            imp_val = imp_row[idx]
            if abs(imp_val) < 0.3:
                continue
            direction = "↑" if imp_val > 0 else "↓"
            var_simple = self.simplify_variable_name(feature_names[idx])
            drivers.append(f"{direction}{var_simple}")
        
        return drivers
    
    def label_table_from_importance(
        self,
        imp_matrix: np.ndarray,
        scenario_tag: str,
        model_tag: str,
        feature_names: List[str],
        mask_incl: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """Genera tabla de etiquetas climáticas desde matriz de importancias."""
        if mask_incl is None:
            mask_incl = np.array([
                (not n.startswith("calliope_")) and (not n.endswith("_std_T"))
                for n in feature_names
            ])
        
        # Mapear grupos de variables (sin aplicar mask_incl - usar índices originales)
        GROUPS = {k: self.find_cols(feature_names, rx) for k, rx in self.PAT.items()}
        
        rows = []
        for k in range(self.k_clusters):
            # Etiquetas climáticas
            tags = self.tags_from_importance_row(imp_matrix[k], GROUPS)
            label_clima = ", ".join(tags)
            
            # Top drivers
            drivers = self.get_top_drivers_summary(imp_matrix[k], feature_names, top_n=3)
            label_drivers = " | ".join(drivers) if drivers else "drivers débiles"
            
            label_compact = f"{label_clima} [{label_drivers}]"
            
            rows.append({
                "model": model_tag,
                "scenario": scenario_tag,
                "cluster": k,
                "label": label_clima,
                "drivers": label_drivers,
                "label_compact": label_compact
            })
        
        df = pd.DataFrame(rows)
        
        # ✅ AGREGAR DESAMBIGUACIÓN PARA CLUSTERS CON ETIQUETAS DUPLICADAS
        df = self.add_cluster_disambiguation(df, imp_matrix, feature_names)
        
        return df
    
    def find_discriminative_variable_enhanced(
        self,
        imp_matrix: np.ndarray,
        cluster_ids: List[int],
        feature_names: List[str]
    ) -> Dict[int, Optional[str]]:
        """
        VERSIÓN AVANZADA: Desambigua clusters considerando INTENSIDAD y CONTEXTO ADICIONAL.
        
        Estrategia:
        1. Identificar variable con mayor varianza entre clusters
        2. Si varios clusters comparten la misma variable:
           - Primero buscar contexto adicional para TODOS
           - Si todos tienen el MISMO contexto → usar INTENSIDAD
           - Si tienen contextos diferentes → usar CONTEXTO
        """
        if len(cluster_ids) <= 1:
            return {cluster_ids[0]: None}
        
        # Extraer sub-matriz y ajustar feature_names
        sub_imp = imp_matrix[cluster_ids, :]
        n_features_actual = sub_imp.shape[1]
        feature_names_adj = feature_names[:n_features_actual] if len(feature_names) > n_features_actual else feature_names
        
        # Excluir H₂
        h2_indices = [i for i, name in enumerate(feature_names_adj) if 'calliope_h2' in name.lower()]
        
        # Calcular varianza entre clusters
        var_between = np.var(sub_imp, axis=0)
        for idx in h2_indices:
            if idx < len(var_between):
                var_between[idx] = -np.inf
        
        # Priorizar topografía x2
        topo_keywords = ['elevation', 'elev', 'slope', 'pend', 'glacier', 'glaciar', 'aspect', 'orient']
        for i, name in enumerate(feature_names_adj):
            if i < len(var_between) and any(kw in name.lower() for kw in topo_keywords):
                if var_between[i] > 0:
                    var_between[i] *= 2.0
        
        # Ordenar variables por varianza descendente
        most_discriminative_idx = np.argsort(var_between)[::-1]
        
        # PASO 1: Asignar variable discriminativa principal a cada cluster
        primary_vars = {}
        for cluster_id in cluster_ids:
            cluster_imp = imp_matrix[cluster_id, :]
            
            for var_idx in most_discriminative_idx[:20]:
                if var_idx in h2_indices or var_idx >= len(cluster_imp):
                    continue
                
                imp_val = cluster_imp[var_idx]
                if abs(imp_val) > 0.25:
                    var_name = self.simplify_variable_name(feature_names_adj[var_idx])
                    direction = "↑" if imp_val > 0 else "↓"
                    primary_vars[cluster_id] = {
                        'var_name': var_name,
                        'var_idx': var_idx,
                        'direction': direction,
                        'value': imp_val
                    }
                    break
        
        # PASO 2: Detectar colisiones (misma variable en múltiples clusters)
        var_usage = {}
        for cid, info in primary_vars.items():
            key = f"{info['direction']}{info['var_name']}"
            if key not in var_usage:
                var_usage[key] = []
            var_usage[key].append((cid, info['value']))
        
        # PASO 3: Resolver colisiones con intensidad + contexto
        discriminative_features = {}
        
        for key, clusters_with_var in var_usage.items():
            if len(clusters_with_var) == 1:
                # No hay colisión, usar variable sola
                cid, _ = clusters_with_var[0]
                var_info = primary_vars[cid]
                var_short = var_info['var_name'][:8]
                discriminative_features[cid] = f"{var_info['direction']}{var_short}"
            else:
                # HAY COLISIÓN: agregar intensidad + contexto
                # Ordenar por valor absoluto de importancia
                sorted_clusters = sorted(clusters_with_var, key=lambda x: abs(x[1]))
                
                # Asignar intensidades relativas
                n_clusters = len(sorted_clusters)
                intensity_labels_map = {
                    2: ["bajo", "alto"],
                    3: ["bajo", "medio", "alto"],
                    4: ["muy-bajo", "bajo", "alto", "muy-alto"],
                    5: ["extremo-bajo", "bajo", "medio", "alto", "extremo-alto"],
                    6: ["mínimo", "muy-bajo", "bajo", "alto", "muy-alto", "máximo"]
                }
                intensity_labels = intensity_labels_map.get(n_clusters, [f"int{r+1}" for r in range(n_clusters)])
                
                # PRIMERO: buscar contexto para TODOS los clusters en colisión
                contexts = {}
                for rank, (cid, val) in enumerate(sorted_clusters):
                    cluster_imp = imp_matrix[cid, :]
                    context_keywords = {
                        'glaciar': ['glacier', 'glaciar'],
                        'urbano': ['urban', 'urbano'],
                        'agrícola': ['agric', 'agriculture'],
                        'bosque': ['forest', 'bosque'],
                        'costa': ['water', 'ocean'],
                        'montaña': ['elevation', 'elev']
                    }
                    
                    context_tag = None
                    max_context_imp = 0.3  # Umbral mínimo para contexto
                    
                    for context_name, keywords in context_keywords.items():
                        for idx, fname in enumerate(feature_names_adj):
                            if idx >= len(cluster_imp):
                                continue
                            if any(kw in fname.lower() for kw in keywords):
                                imp_val = abs(cluster_imp[idx])
                                if imp_val > max_context_imp:
                                    max_context_imp = imp_val
                                    context_tag = context_name
                    
                    contexts[cid] = context_tag
                
                # VERIFICAR si todos tienen el MISMO contexto (o ninguno)
                unique_contexts = set(contexts.values())
                all_same_context = len(unique_contexts) == 1
                
                # CONSTRUIR etiquetas según resultado
                for rank, (cid, val) in enumerate(sorted_clusters):
                    var_info = primary_vars[cid]
                    var_name_full = var_info['var_name']
                    # Truncar solo si es muy largo (más de 10 caracteres)
                    var_short = var_name_full[:10] if len(var_name_full) > 10 else var_name_full
                    intensity = intensity_labels[rank]
                    context_tag = contexts[cid]
                    
                    if all_same_context or context_tag is None:
                        # Mismo contexto para todos O ningún contexto: usar INTENSIDAD
                        discriminative_features[cid] = f"{var_info['direction']}{var_short}-{intensity}"
                    else:
                        # Contextos diferentes: usar CONTEXTO (más informativo)
                        discriminative_features[cid] = f"{var_info['direction']}{var_short}+{context_tag}"
        
        # Clusters sin asignación: usar letra
        for cid in cluster_ids:
            if cid not in discriminative_features:
                letter_idx = cluster_ids.index(cid)
                discriminative_features[cid] = chr(65 + letter_idx)
        
        return discriminative_features
    
    def find_discriminative_variable(
        self,
        imp_matrix: np.ndarray,
        cluster_ids: List[int],
        feature_names: List[str]
    ) -> Dict[int, Optional[str]]:
        """
        Encuentra la variable que MEJOR DIFERENCIA clusters con misma etiqueta climática.
        
        Parámetros:
        -----------
        imp_matrix : np.ndarray
            Matriz de importancias (K, n_features)
        cluster_ids : list
            IDs de los clusters a diferenciar
        feature_names : list
            Nombres de las variables
        
        Returns:
        --------
        dict : {cluster_id: "sufijo_corto"} o {cluster_id: None}
        """
        if len(cluster_ids) <= 1:
            return {cluster_ids[0]: None}
        
        # Extraer sub-matriz solo para estos clusters
        sub_imp = imp_matrix[cluster_ids, :]
        n_features_actual = sub_imp.shape[1]
        
        # Ajustar feature_names si es más largo que la matriz
        if len(feature_names) > n_features_actual:
            feature_names_adj = feature_names[:n_features_actual]
        else:
            feature_names_adj = feature_names
        
        # FILTRAR variables de H₂
        h2_indices = [i for i, name in enumerate(feature_names_adj) if 'calliope_h2' in name.lower()]
        
        # Calcular varianza entre clusters (mayor varianza = mejor discriminación)
        var_between = np.var(sub_imp, axis=0)
        if len(h2_indices) > 0:
            for idx in h2_indices:
                if idx < len(var_between):
                    var_between[idx] = -np.inf
        
        # PRIORIZAR variables topográficas (cruciales para diferenciación espacial)
        topo_keywords = ['elevation', 'elev', 'slope', 'pend', 'glacier', 'glaciar', 'aspect', 'orient']
        for i, name in enumerate(feature_names_adj):
            if i < len(var_between) and any(kw in name.lower() for kw in topo_keywords):
                if var_between[i] > 0:
                    var_between[i] *= 2.0  # Bonus x2 para topografía
        
        # Ordenar por varianza (discriminación)
        most_discriminative_idx = np.argsort(var_between)[::-1]
        
        # Para cada cluster, encontrar su característica MÁS DISTINTIVA
        discriminative_features = {}
        
        for cluster_id in cluster_ids:
            cluster_imp = imp_matrix[cluster_id, :]
            found = False
            
            # Buscar en las top 20 variables más discriminativas
            for var_idx in most_discriminative_idx[:20]:
                if var_idx in h2_indices or var_idx >= len(cluster_imp):
                    continue
                
                imp_val = cluster_imp[var_idx]
                
                # UMBRAL REDUCIDO: 0.25 para capturar diferencias sutiles
                if abs(imp_val) > 0.25:
                    var_name = self.simplify_variable_name(feature_names_adj[var_idx])
                    direction_symbol = "↑" if imp_val > 0 else "↓"
                    var_short = var_name[:8] if len(var_name) > 8 else var_name
                    discriminative_features[cluster_id] = f"{direction_symbol}{var_short}"
                    found = True
                    break
            
            # Si UMBRAL 0.25 falló, buscar la variable con MAYOR DIFERENCIA vs otros clusters del grupo
            if not found:
                cluster_mean_group = np.mean(sub_imp, axis=0)
                diff_from_group = np.abs(cluster_imp - cluster_mean_group)
                
                # Excluir H₂
                for idx in h2_indices:
                    if idx < len(diff_from_group):
                        diff_from_group[idx] = -np.inf
                
                best_var_idx = np.argmax(diff_from_group)
                
                if best_var_idx < len(feature_names_adj) and diff_from_group[best_var_idx] > 0.1:
                    var_name = self.simplify_variable_name(feature_names_adj[best_var_idx])
                    imp_val = cluster_imp[best_var_idx]
                    direction_symbol = "↑" if imp_val > 0 else "↓"
                    var_short = var_name[:8] if len(var_name) > 8 else var_name
                    discriminative_features[cluster_id] = f"{direction_symbol}{var_short}"
                else:
                    # Último recurso: letra alfabética
                    letter_idx = cluster_ids.index(cluster_id)
                    discriminative_features[cluster_id] = chr(65 + letter_idx)
        
        return discriminative_features

    def add_cluster_disambiguation(
        self,
        labels_df: pd.DataFrame,
        imp_matrix: np.ndarray,
        feature_names: List[str]
    ) -> pd.DataFrame:
        """
        Agrega sufijos distintivos a clusters con etiquetas climáticas duplicadas.
        
        Parámetros:
        -----------
        labels_df : pd.DataFrame
            DataFrame con columnas: cluster, label, drivers, label_compact
        imp_matrix : np.ndarray
            Matriz de importancias para identificar diferencias
        feature_names : list
            Nombres de las variables
        
        Returns:
        --------
        pd.DataFrame con columna adicional 'label_disambiguated'
        """
        df = labels_df.copy()
        
        # Agrupar clusters por etiqueta climática base
        label_groups = df.groupby('label')['cluster'].apply(list).to_dict()
        
        # Para cada grupo con duplicados, encontrar variables discriminativas
        # USAR find_discriminative_variable_enhanced SI EXISTE, sino fallback a versión original
        disambiguation = {}
        for base_label, cluster_list in label_groups.items():
            if len(cluster_list) > 1:
                # Hay duplicados, usar desambiguación avanzada con intensidad + contexto
                try:
                    discrim = self.find_discriminative_variable_enhanced(imp_matrix, cluster_list, feature_names)
                except Exception as e:
                    # Fallback a versión original si enhanced no está disponible
                    print(f"[DEBUG] Enhanced falló para '{base_label}': {e}, usando fallback")
                    discrim = self.find_discriminative_variable(imp_matrix, cluster_list, feature_names)
                disambiguation.update(discrim)
            else:
                # No hay duplicados, no agregar sufijo
                disambiguation[cluster_list[0]] = None
        
        # Crear etiquetas desambiguadas
        def create_disambiguated_label(row):
            cluster_id = row['cluster']
            base_label = row['label']
            suffix = disambiguation.get(cluster_id)
            
            if suffix:
                return f"{base_label} ({suffix})"
            else:
                return base_label
        
        df['label_disambiguated'] = df.apply(create_disambiguated_label, axis=1)
        
        return df

    def find_h2_cols(self, names: List[str]) -> Tuple[Dict[str, int], Optional[int]]:
        """Identifica columnas de producción de H₂."""
        mean_cols = {}
        std_idx = None
        for i, n in enumerate(names):
            m = self.rx_h2_mean.match(n)
            if m:
                mean_cols[m.group(1)] = i
            if self.rx_h2_stdT.match(n):
                std_idx = i
        return mean_cols, std_idx
    
    def add_h2_overlay(
        self,
        labels_df: pd.DataFrame,
        tag: str,
        X: np.ndarray,
        labels: np.ndarray,
        feature_names: List[str]
    ) -> pd.DataFrame:
        """Agrega información de H₂ (potencial y estabilidad) a etiquetas."""
        h2_mean_cols, h2_std_idx = self.find_h2_cols(feature_names)
        
        yy = self.decasel.get(tag)
        if yy not in h2_mean_cols:
            labels_df = labels_df.copy()
            labels_df["h2_rank"] = "NA"
            labels_df["h2_stability"] = "NA"
            labels_df["label_complete"] = labels_df["label_compact"]
            return labels_df
        
        # Calcular ranking de H₂
        mean_idx = h2_mean_cols[yy]
        scen_vals = X[:, mean_idx]
        valid_vals = scen_vals[~np.isnan(scen_vals)]
        
        if len(valid_vals) == 0:
            labels_df["h2_rank"] = "NA"
            labels_df["h2_stability"] = "NA"
            labels_df["label_complete"] = labels_df["label_compact"]
            return labels_df
        
        p33, p66 = np.percentile(valid_vals, [33, 66])
        
        # Ranking por cluster
        h2_ranks = []
        for k in range(self.k_clusters):
            idx = (labels == k)
            if not np.any(idx):
                h2_ranks.append("sin datos")
                continue
            
            cluster_vals = X[idx, mean_idx]
            valid_cluster = cluster_vals[~np.isnan(cluster_vals)]
            
            if len(valid_cluster) == 0:
                h2_ranks.append("sin datos")
            else:
                med = float(np.median(valid_cluster))
                if med <= p33:
                    h2_ranks.append("bajo")
                elif med >= p66:
                    h2_ranks.append("alto")
                else:
                    h2_ranks.append("medio")
        
        labels_df["h2_rank"] = h2_ranks
        labels_df["h2_stability"] = "NA"  # Simplificado por ahora
        
        # Crear etiqueta completa
        labels_df["label_complete"] = labels_df.apply(
            lambda r: f"{r['label_compact']} | H₂: {r['h2_rank']}" 
            if r['h2_rank'] != "NA" else r['label_compact'],
            axis=1
        )
        
        return labels_df
    
    def plot_clusters_spatial(
        self,
        labels_array: np.ndarray,
        lat_vals: np.ndarray,
        lon_vals: np.ndarray,
        title: str,
        cluster_labels_dict: Optional[Dict[int, str]] = None,
        alpha: float = 0.75,
        figsize: Tuple[int, int] = (12, 10),
        save_path: Optional[str] = None
    ):
        """Visualización espacial de clusters con etiquetas."""
        labels_arr = np.asarray(labels_array)
        
        # Truncar si es necesario
        n_coords = len(lat_vals)
        if len(labels_arr) > n_coords:
            labels_arr = labels_arr[:n_coords]
        
        valid_mask = ~pd.isna(labels_arr)
        
        if not valid_mask.any():
            print(f"Sin datos válidos para {title}")
            return
        
        unique_vals = np.sort(np.unique(labels_arr[valid_mask]))
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Paleta de colores
        color_palette = plt.get_cmap('tab20', 20)
        cluster_colors = [color_palette(int(val) % 20) for val in unique_vals]
        discrete_cmap = ListedColormap(cluster_colors)
        
        val_to_idx = {val: idx for idx, val in enumerate(unique_vals)}
        
        try:
            # Proyección Web Mercator
            try:
                from pyproj import Transformer
                transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
                xs, ys = transformer.transform(lon_vals, lat_vals)
            except:
                xs = lon_vals * 111320
                ys = lat_vals * 110540
            
            xs_valid = xs[valid_mask]
            ys_valid = ys[valid_mask]
            labels_valid = labels_arr[valid_mask]
            
            # Grid interpolation
            grid_res = min(150, max(80, int(np.sqrt(len(xs_valid)) * 2)))
            grid_x = np.linspace(xs.min(), xs.max(), grid_res)
            grid_y = np.linspace(ys.min(), ys.max(), grid_res)
            GX, GY = np.meshgrid(grid_x, grid_y)
            extent_map = (grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max())
            
            # KNN interpolation
            coords_valid = np.column_stack([xs_valid, ys_valid])
            int_targets = np.vectorize(val_to_idx.get)(labels_valid)
            
            n_neighbors = max(1, min(len(int_targets), int(np.sqrt(len(int_targets)))))
            clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")
            clf.fit(coords_valid, int_targets)
            
            grid_points = np.column_stack([GX.ravel(), GY.ravel()])
            pred = clf.predict(grid_points).reshape(GX.shape)
            
            ax.set_xlim(extent_map[0], extent_map[1])
            ax.set_ylim(extent_map[2], extent_map[3])
            
            # Basemap (opcional)
            try:
                import contextily as ctx
                ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, 
                              crs="EPSG:3857", alpha=1.0, zoom='auto')
            except:
                pass
            
            # Plot clusters
            boundaries = np.arange(len(unique_vals) + 1) - 0.5
            norm = BoundaryNorm(boundaries, discrete_cmap.N)
            
            ax.imshow(pred, extent=extent_map, origin="lower", cmap=discrete_cmap,
                     norm=norm, alpha=alpha, zorder=3)
            ax.set_axis_off()
            ax.set_title(title, fontsize=13, pad=15, fontweight='bold')
            
            # Colorbar
            mappable = ScalarMappable(norm=norm, cmap=discrete_cmap)
            cbar = fig.colorbar(mappable, ax=ax, fraction=0.035, pad=0.02,
                               ticks=np.arange(len(unique_vals)))
            cbar.set_ticklabels([str(int(val)) for val in unique_vals], fontsize=8)
            cbar.set_label("Cluster ID", fontsize=11)
            
            # Anotaciones con labels
            if cluster_labels_dict:
                for val in unique_vals:
                    cluster_id = int(val)
                    cluster_mask = labels_valid == val
                    
                    if cluster_mask.sum() > 0:
                        centroid_x = xs_valid[cluster_mask].mean()
                        centroid_y = ys_valid[cluster_mask].mean()
                        
                        label_text = cluster_labels_dict.get(cluster_id, str(cluster_id))
                        
                        ax.annotate(
                            label_text,
                            xy=(centroid_x, centroid_y),
                            fontsize=7,
                            fontweight='bold',
                            color='black',
                            ha='center',
                            va='center',
                            bbox=dict(
                                boxstyle='round,pad=0.4',
                                facecolor='white',
                                edgecolor='gray',
                                alpha=0.85,
                                linewidth=0.8
                            ),
                            zorder=10
                        )
            
            fig.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Figura guardada: {save_path}")
            
            plt.show()
            
        except Exception as err:
            print(f"Error en plot_clusters_spatial: {err}")
            import traceback
            traceback.print_exc()
            plt.close(fig)
    
    def run_full_profiling(
        self,
        data_blocks: Dict[str, np.ndarray],
        coords_df: pd.DataFrame,
        feature_names: List[str],
        scenarios: List[str] = ['BASE', 'SSP245', 'SSP370', 'SSP585']
    ) -> Dict:
        """
        Pipeline completo: encoding → clustering → importancias → etiquetado → H₂ overlay.
        
        Parameters:
        -----------
        data_blocks : dict
            {'BASE': X_BASE, 'SSP245': X245, ...}
        coords_df : pd.DataFrame
            Coordenadas espaciales
        feature_names : list[str]
            Nombres de variables
        scenarios : list[str]
            Escenarios a procesar
        
        Returns:
        --------
        results : dict
            Diccionario completo con todos los resultados por modelo y escenario
        """
        self.feature_names = feature_names
        results = {}
        
        for model_key, model in self.models_dict.items():
            print(f"\n{'='*80}")
            print(f"PROCESANDO MODELO: {model_key}")
            print(f"{'='*80}")
            
            model_results = {}
            
            for scenario_name in scenarios:
                if scenario_name not in data_blocks:
                    print(f"  ⚠ Escenario {scenario_name} no encontrado, saltando...")
                    continue
                
                print(f"\n  Escenario: {scenario_name}")
                X_data = data_blocks[scenario_name]
                
                # 1. Encoding
                print(f"    1/5. Encoding al espacio latente...")
                Z_latent = self.encode_to_latent(model_key, X_data)
                
                # 2. Clustering
                print(f"    2/5. Aplicando KMeans (K={self.k_clusters})...")
                labels, kmeans = self.apply_clustering(Z_latent)
                
                # 3. Importancias
                print(f"    3/5. Calculando importancias de variables...")
                imp_matrix = self.calculate_feature_importance(X_data, labels)
                
                # 4. Etiquetado climático
                print(f"    4/5. Generando etiquetas climáticas...")
                labels_df = self.label_table_from_importance(
                    imp_matrix, scenario_name, model_key, feature_names
                )
                
                # 5. Overlay H₂
                print(f"    5/5. Agregando información de H₂...")
                labels_df = self.add_h2_overlay(
                    labels_df, scenario_name, X_data, labels, feature_names
                )
                
                # Guardar resultados
                model_results[scenario_name] = {
                    'X_data': X_data,
                    'Z_latent': Z_latent,
                    'labels': labels,
                    'kmeans': kmeans,
                    'importances': imp_matrix,
                    'labels_df': labels_df,
                    'coords': coords_df
                }
            
            results[model_key] = model_results
        
        print(f"\n{'='*80}")
        print(f"✓ PROCESAMIENTO COMPLETO")
        print(f"{'='*80}\n")
        
        return results
    
    def plot_spatial_comparison_labeled(
        self,
        labels_base: np.ndarray,
        labels_target: np.ndarray,
        labels_dict_base: Dict[int, str],
        labels_dict_target: Dict[int, str],
        lat_vals: np.ndarray,
        lon_vals: np.ndarray,
        title_base: str = "BASE (2020-2029)",
        title_target: str = "TARGET (2090-2100)",
        suptitle: str = "Comparación BASE → TARGET",
        alpha: float = 0.75,
        n_neighbors: int = 15,
        figsize: Tuple[int, int] = (18, 7),
        save_path: Optional[str] = None
    ):
        """
        Visualización espacial de comparación BASE → TARGET con etiquetas climáticas.
        
        Muestra dos paneles lado a lado (BASE y TARGET) con interpolación espacial
        y una leyenda que indica las transiciones de perfiles climáticos.
        
        Parámetros:
        -----------
        labels_base : np.ndarray
            Etiquetas de cluster para período BASE
        labels_target : np.ndarray
            Etiquetas de cluster para período TARGET
        labels_dict_base : dict
            Diccionario {cluster_id: etiqueta_climática} para BASE
        labels_dict_target : dict
            Diccionario {cluster_id: etiqueta_climática} para TARGET
        lat_vals : np.ndarray
            Coordenadas latitud
        lon_vals : np.ndarray
            Coordenadas longitud
        title_base : str
            Título del panel BASE
        title_target : str
            Título del panel TARGET
        suptitle : str
            Título general de la figura
        alpha : float
            Transparencia de la capa de clusters (0-1)
        n_neighbors : int
            Número de vecinos para interpolación KNN
        figsize : tuple
            Tamaño de la figura (ancho, alto)
        save_path : str, opcional
            Ruta para guardar la figura
        """
        try:
            fig, axes = plt.subplots(1, 2, figsize=figsize)
            
            # Transformar coordenadas a Web Mercator
            try:
                from pyproj import Transformer
                transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
                xs, ys = transformer.transform(lon_vals, lat_vals)
            except:
                xs, ys = lon_vals * 111320, lat_vals * 110540
            
            # Crear grid de interpolación
            grid_res = 100
            grid_x = np.linspace(xs.min(), xs.max(), grid_res)
            grid_y = np.linspace(ys.min(), ys.max(), grid_res)
            GX, GY = np.meshgrid(grid_x, grid_y)
            extent = (grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max())
            
            # Colormap para clusters
            n_unique_labels = len(set(labels_dict_base.keys()) | set(labels_dict_target.keys()))
            discrete_cmap = ListedColormap(plt.get_cmap("tab10", n_unique_labels)(np.arange(n_unique_labels)))
            boundaries = np.arange(n_unique_labels + 1) - 0.5
            norm = BoundaryNorm(boundaries, discrete_cmap.N)
            
            # === PANEL BASE ===
            valid_mask_base = ~(np.isnan(labels_base) | np.isinf(labels_base))
            coords_base = np.column_stack([xs[valid_mask_base], ys[valid_mask_base]])
            labels_arr_base = labels_base[valid_mask_base]
            
            clf_base = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")
            clf_base.fit(coords_base, labels_arr_base)
            
            grid_points = np.column_stack([GX.ravel(), GY.ravel()])
            pred_base = clf_base.predict(grid_points).reshape(GX.shape)
            
            axes[0].set_xlim(extent[0], extent[1])
            axes[0].set_ylim(extent[2], extent[3])
            
            # Basemap opcional
            try:
                import contextily as ctx
                ctx.add_basemap(axes[0], source=ctx.providers.CartoDB.Positron, 
                              crs="EPSG:3857", alpha=1.0, attribution_size=6)
            except:
                pass
            
            axes[0].imshow(pred_base, extent=extent, origin="lower", 
                          cmap=discrete_cmap, norm=norm, alpha=alpha, zorder=3)
            axes[0].set_axis_off()
            axes[0].set_title(title_base, fontsize=12, pad=10, fontweight='bold')
            
            # === PANEL TARGET ===
            valid_mask_target = ~(np.isnan(labels_target) | np.isinf(labels_target))
            coords_target = np.column_stack([xs[valid_mask_target], ys[valid_mask_target]])
            labels_arr_target = labels_target[valid_mask_target]
            
            clf_target = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")
            clf_target.fit(coords_target, labels_arr_target)
            pred_target = clf_target.predict(grid_points).reshape(GX.shape)
            
            axes[1].set_xlim(extent[0], extent[1])
            axes[1].set_ylim(extent[2], extent[3])
            
            try:
                ctx.add_basemap(axes[1], source=ctx.providers.CartoDB.Positron, 
                              crs="EPSG:3857", alpha=1.0, attribution_size=6)
            except:
                pass
            
            axes[1].imshow(pred_target, extent=extent, origin="lower", 
                          cmap=discrete_cmap, norm=norm, alpha=alpha, zorder=3)
            axes[1].set_axis_off()
            axes[1].set_title(title_target, fontsize=12, pad=10, fontweight='bold')
            
            # === LEYENDA CON ETIQUETAS CLIMÁTICAS ===
            from matplotlib.patches import Patch
            
            # Recopilar todas las etiquetas únicas de ambos paneles
            all_cluster_ids = set(labels_dict_base.keys()) | set(labels_dict_target.keys())
            
            legend_elements = []
            for cluster_id in sorted(all_cluster_ids):
                color = discrete_cmap(cluster_id)
                
                label_base = labels_dict_base.get(cluster_id, "")
                label_target = labels_dict_target.get(cluster_id, "")
                
                # Construir etiqueta según disponibilidad
                if label_base == label_target and label_base:
                    label_text = f"C{cluster_id}: {label_base}"
                elif label_base and label_target:
                    label_text = f"C{cluster_id}: {label_base} → {label_target}"
                elif label_base:
                    label_text = f"C{cluster_id}: {label_base} (solo BASE)"
                else:
                    label_text = f"C{cluster_id}: {label_target} (solo TARGET)"
                
                legend_elements.append(Patch(facecolor=color, edgecolor='black', label=label_text))
            
            # Agregar leyenda debajo de la figura
            fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, -0.05),
                      ncol=1, fontsize=9, frameon=True, fancybox=True, shadow=True)
            
            fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=0.98)
            fig.tight_layout(rect=[0, 0.05, 1, 0.96])
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"Figura guardada: {save_path}")
            
            plt.show()
            
        except Exception as err:
            print(f"Error en plot_spatial_comparison_labeled: {err}")
            import traceback
            traceback.print_exc()
            plt.close(fig)
    
    def plot_all_scenarios(
        self,
        results: Dict,
        save_dir: Optional[str] = None,
        scenarios: List[str] = ['BASE', 'SSP245', 'SSP370', 'SSP585']
    ):
        """Genera visualizaciones espaciales para todos los modelos y escenarios."""
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
        
        for model_key, model_results in results.items():
            print(f"\n{'='*80}")
            print(f"VISUALIZANDO: {model_key}")
            print(f"{'='*80}")
            
            for scenario_name in scenarios:
                if scenario_name not in model_results:
                    continue
                
                scenario_data = model_results[scenario_name]
                labels_df = scenario_data['labels_df']
                labels = scenario_data['labels']
                coords = scenario_data['coords']
                
                # Crear diccionario de etiquetas
                cluster_labels_dict = {}
                for _, row in labels_df.iterrows():
                    cluster_id = row['cluster']
                    label_text = row.get('label_complete', row.get('label_compact', row['label']))
                    cluster_labels_dict[cluster_id] = label_text
                
                # Plot
                title = f"{model_key} | {scenario_name}"
                save_file = str(save_path / f"spatial_{model_key}_{scenario_name}.png") if save_dir else None
                
                self.plot_clusters_spatial(
                    labels_array=labels,
                    lat_vals=coords['lat'].values,
                    lon_vals=coords['lon'].values,
                    title=title,
                    cluster_labels_dict=cluster_labels_dict,
                    alpha=0.75,
                    figsize=(12, 10),
                    save_path=save_file
                )
        
        print(f"\n✓ Visualizaciones completas")


if __name__ == '__main__':
    print("Módulo cluster_profiling_pipeline.py")
    print("Uso:")
    print("  from cluster_profiling_pipeline import ClusterProfilingPipeline")
    print("  pipeline = ClusterProfilingPipeline(models_dict, k_clusters=10)")
    print("  results = pipeline.run_full_profiling(data_blocks, coords_df, feature_names)")
