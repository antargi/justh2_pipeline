"""
Diagrama de arquitectura energética del sistema Calliope v6
Visualización técnica para sección 3.4 de la tesis
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np
from pathlib import Path

# Configuración de estilo (igual que notebook de referencia)
plt.style.use('seaborn-v0_8-paper')

# Paleta de colores pasteles (del notebook de referencia)
COLORS = {
    'pv': '#FFFFBA',           # Amarillo pastel (solar)
    'electrolyzer': '#BAFFC9',  # Verde menta (conversión)
    'storage': '#BAE1FF',       # Azul cielo (almacenamiento)
    'demand': '#FFB3BA',        # Rosa pastel (demanda)
    'water': '#B3FFE6',         # Turquesa (agua)
    'transmission': '#C9B3FF',  # Lavanda (transmisión)
    'desalination': '#B3D9FF',  # Azul claro (desalación)
    'carrier_elec': '#FFE5CC',  # Beige (electricidad)
    'carrier_h2': '#CCFFCC',    # Verde claro (hidrógeno)
    'carrier_water': '#CCE5FF', # Azul muy claro (agua)
    'node': '#9b59b6',          # Morado (nodos - del notebook)
    'border': '#333333',        # Gris oscuro
    'text': '#2c3e50',          # Azul oscuro
}

def create_technology_box(ax, x, y, width, height, label, color, params=None, zorder=5):
    """Crea una caja para representar una tecnología"""
    # Sombra
    shadow = FancyBboxPatch(
        (x + 0.02, y - 0.02), width, height,
        boxstyle="round,pad=0.1", 
        facecolor='black', 
        edgecolor='none',
        alpha=0.3,
        zorder=zorder-1
    )
    ax.add_patch(shadow)
    
    # Caja principal
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.1", 
        facecolor=color, 
        edgecolor=COLORS['border'],
        linewidth=2.5,
        alpha=0.85,
        zorder=zorder
    )
    ax.add_patch(box)
    
    # Texto principal
    ax.text(x + width/2, y + height*0.65, label, 
            ha='center', va='center', 
            fontsize=12, fontweight='bold',
            color=COLORS['text'],
            zorder=zorder+1)
    
    # Parámetros técnicos
    if params:
        param_text = '\n'.join(params)
        ax.text(x + width/2, y + height*0.25, param_text, 
                ha='center', va='center', 
                fontsize=8,
                color=COLORS['text'],
                zorder=zorder+1)

def create_carrier_arrow(ax, x1, y1, x2, y2, label, color, style='solid', width=3):
    """Crea una flecha representando un carrier energético"""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='->,head_width=0.4,head_length=0.6',
        color=color,
        linewidth=width,
        linestyle=style,
        alpha=0.7,
        mutation_scale=25,
        zorder=3
    )
    ax.add_patch(arrow)
    
    # Etiqueta del carrier
    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
    
    # Ajustar posición del texto según dirección
    if abs(x2 - x1) > abs(y2 - y1):  # Horizontal
        offset_y = 0.15
        offset_x = 0
    else:  # Vertical
        offset_x = 0.25
        offset_y = 0
    
    ax.text(mid_x + offset_x, mid_y + offset_y, label,
            ha='center', va='center',
            fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', 
                     facecolor='white', 
                     edgecolor=color,
                     alpha=0.9,
                     linewidth=1.5),
            color=COLORS['text'],
            zorder=4)

def create_node_circle(ax, x, y, label, size=0.4):
    """Crea un círculo representando un nodo espacial"""
    circle = Circle((x, y), size, 
                   facecolor=COLORS['node'],
                   edgecolor=COLORS['border'],
                   linewidth=2,
                   alpha=0.7,
                   zorder=6)
    ax.add_patch(circle)
    
    ax.text(x, y, label,
            ha='center', va='center',
            fontsize=10, fontweight='bold',
            color='white',
            zorder=7)

# Crear figura con más espacio vertical
fig, ax = plt.subplots(figsize=(24, 16))
ax.set_xlim(0, 24)
ax.set_ylim(0, 16)
ax.axis('off')

# Título
ax.text(12, 15.2, 'Arquitectura Energética del Sistema H2V - Calliope v6',
        ha='center', va='top',
        fontsize=20, fontweight='bold',
        color=COLORS['text'])

ax.text(12, 14.5, 'Modelo Distribuido Punto por Punto (330 sistemas independientes)',
        ha='center', va='top',
        fontsize=13,
        color=COLORS['text'],
        style='italic')

# ====== NODO PV_SITE (izquierda) ======
node_pv_x, node_pv_y = 4, 9.5
create_node_circle(ax, node_pv_x, node_pv_y, 'PV_SITE\n(lat, lon)', size=0.6)

# Tecnología PV
pv_x, pv_y = 1, 6.8
create_technology_box(
    ax, pv_x, pv_y, 3.2, 2.3,
    'PV\nFotovoltaico',
    COLORS['pv'],
    params=[
        'Cap max: 5000 MW',
        'Lifetime: 25 años',
        'CAPEX: 700k $/MW'
    ]
)

# Input: CF solar (más arriba para no sobreponerse)
ax.text(2.6, 5.4, 'Input: CF mensual\n(CMIP6 SSP bias-corrected)',
        ha='center', va='top',
        fontsize=10,
        bbox=dict(boxstyle='round,pad=0.5',
                 facecolor='#FFF9E6',
                 edgecolor=COLORS['border'],
                 linewidth=1.5),
        color=COLORS['text'])

# Flecha: PV genera electricidad (hacia arriba al nodo)
create_carrier_arrow(ax, 2.6, 9.1, 3.4, 9.5, 
                    'electricity', COLORS['carrier_elec'])

# ====== TRANSMISIÓN ======
# AC Line
trans_x, trans_y = 7, 8.8
create_technology_box(
    ax, trans_x, trans_y, 4, 1.8,
    'AC Line\nTransmisión',
    COLORS['transmission'],
    params=[
        'Eficiencia: 95%',
        'Cap max: 5000 MW'
    ]
)

# Flecha: Del nodo PV a AC Line
create_carrier_arrow(ax, 4.6, 9.5, 7, 9.7,
                    'electricity\n(~5% pérdidas)', COLORS['carrier_elec'])

# ====== NODO VALPO (centro-derecha) ======
node_valpo_x, node_valpo_y = 14, 9.5
create_node_circle(ax, node_valpo_x, node_valpo_y, 'VALPO\n(-33.0, -71.6)', size=0.6)

# Flecha: AC Line a nodo VALPO
create_carrier_arrow(ax, 11, 9.7, 13.4, 9.5,
                    '', COLORS['carrier_elec'])

# ====== ELECTROLIZADOR (centro) ======
electrolyzer_x, electrolyzer_y = 12, 6
create_technology_box(
    ax, electrolyzer_x, electrolyzer_y, 4, 2.5,
    'Electrolyzer\nPEM',
    COLORS['electrolyzer'],
    params=[
        'Eff HHV: 68%',
        'Cap max: 10,000 MW_e',
        'CAPEX: 900k $/MW',
        'Consumo: 9 L/kg_H2'
    ]
)

# Flecha: Electricidad del nodo VALPO al electrolizador
create_carrier_arrow(ax, 14, 8.9, 14, 8.5,
                    'electricity', COLORS['carrier_elec'])

# ====== SUBSISTEMA DE AGUA (derecha) ======
# Seawater supply
water_supply_x, water_supply_y = 18.5, 1.8
create_technology_box(
    ax, water_supply_x, water_supply_y, 2.8, 1.6,
    'Seawater\nSupply',
    COLORS['water'],
    params=['Resource: ∞']
)

# Desalination
desal_x, desal_y = 18.5, 4.2
create_technology_box(
    ax, desal_x, desal_y, 2.8, 2.2,
    'Desalination\n(RO)',
    COLORS['desalination'],
    params=[
        'Eff: ~3.5 kWh/m³',
        'Cap max: 50 MW_e',
        'CAPEX: 800k $/MW'
    ]
)

# Flechas sistema agua
create_carrier_arrow(ax, 19.9, 3.4, 19.9, 4.2,
                    'seawater', COLORS['carrier_water'])

# Electricidad a desaladora
create_carrier_arrow(ax, 14, 8.9, 18.5, 5.4,
                    'electricity', COLORS['carrier_elec'], style='dashed', width=2)

# Agua a electrolizador
create_carrier_arrow(ax, 18.5, 5.4, 16, 7.2,
                    'water', COLORS['carrier_water'])

# ====== ALMACENAMIENTO H2 (superior derecha) ======
storage_x, storage_y = 18.5, 10
create_technology_box(
    ax, storage_x, storage_y, 3, 2.4,
    'H2 Storage',
    COLORS['storage'],
    params=[
        'Cap max: 500k MWh',
        'Pérdidas: 0.05%/h',
        'Lifetime: 20 años'
    ]
)

# Flecha: H2 del electrolizador al storage
create_carrier_arrow(ax, 16, 7.2, 18.5, 11.2,
                    'hydrogen', COLORS['carrier_h2'], width=4)

# ====== DEMANDA H2 (inferior derecha) ======
demand_x, demand_y = 18.5, 7
create_technology_box(
    ax, demand_x, demand_y, 3, 2.2,
    'Demand H2',
    COLORS['demand'],
    params=[
        'Temporal: mensual',
        'Distribuida: ÷330',
        'Input: CSV'
    ]
)

# Flecha: H2 del storage a la demanda
create_carrier_arrow(ax, 20, 10, 20, 9.2,
                    'hydrogen', COLORS['carrier_h2'], width=4)

# ====== INFORMACIÓN ADICIONAL (paneles laterales) ======

# Panel izquierdo inferior: Inputs
input_panel_x, input_panel_y = 0.5, 0.5
input_box = FancyBboxPatch(
    (input_panel_x, input_panel_y), 5, 4,
    boxstyle="round,pad=0.15",
    facecolor='#F8F9FA',
    edgecolor=COLORS['border'],
    linewidth=2,
    alpha=0.9,
    zorder=2
)
ax.add_patch(input_box)

ax.text(input_panel_x + 2.5, input_panel_y + 3.6, 'INPUTS DEL MODELO',
        ha='center', va='center',
        fontsize=12, fontweight='bold',
        color=COLORS['text'])

input_text = """• CF fotovoltaico: pv_cf_{ssp}.nc
  330 puntos × 1032 meses
  Fuente: CMIP6 rsds bias-corrected
  
• Demanda H2: demand_h2_monthly_MWh.csv
  Total valle / 330 puntos
  Periodo: 2015-2100
  
• Parámetros: techs.yml
  CAPEX, OPEX, lifetimes
  Eficiencias, capacidades máximas"""

ax.text(input_panel_x + 2.5, input_panel_y + 1.4, input_text,
        ha='center', va='center',
        fontsize=9.5,
        color=COLORS['text'],
        family='monospace')

# Panel derecho inferior: Outputs
output_panel_x, output_panel_y = 6.5, 0.5
output_box = FancyBboxPatch(
    (output_panel_x, output_panel_y), 5, 4,
    boxstyle="round,pad=0.15",
    facecolor='#F0FFF0',
    edgecolor=COLORS['border'],
    linewidth=2,
    alpha=0.9,
    zorder=2
)
ax.add_patch(output_box)

ax.text(output_panel_x + 2.5, output_panel_y + 3.6, 'OUTPUTS OPTIMIZADOS',
        ha='center', va='center',
        fontsize=12, fontweight='bold',
        color=COLORS['text'])

output_text = """• Capacidades óptimas
  PV, Electrolyzer, Storage
  
• Producción H2
  ton/año por punto espacial
  
• LCOH: $/kg_H2
  Costo nivelado completo
  
• Factor de capacidad
  Electrolyzer CF
  
• Consumos
  Electricidad (MWh), Agua (m³)"""

ax.text(output_panel_x + 2.5, output_panel_y + 1.4, output_text,
        ha='center', va='center',
        fontsize=9.5,
        color=COLORS['text'],
        family='monospace')

# Panel superior: Configuración espacial
spatial_panel_x, spatial_panel_y = 6, 12.8
spatial_box = FancyBboxPatch(
    (spatial_panel_x, spatial_panel_y), 12, 1.4,
    boxstyle="round,pad=0.15",
    facecolor='#F0F8FF',
    edgecolor=COLORS['border'],
    linewidth=2,
    alpha=0.9,
    zorder=2
)
ax.add_patch(spatial_box)

spatial_text = """330 sistemas independientes (sin transmisión entre puntos) | Valle de Aconcagua, Valparaíso
Cada punto: PV_SITE (lat, lon) → AC_line → VALPO (hub) → Electrolyzer + Storage + Demand
Escenarios: SSP2-4.5, SSP3-7.0, SSP5-8.5"""

ax.text(spatial_panel_x + 6, spatial_panel_y + 0.7, spatial_text,
        ha='center', va='center',
        fontsize=10,
        color=COLORS['text'])

# Panel optimización (inferior centro-derecha)
opt_panel_x, opt_panel_y = 12.5, 0.5
opt_box = FancyBboxPatch(
    (opt_panel_x, opt_panel_y), 6, 4,
    boxstyle="round,pad=0.15",
    facecolor='#FFF0F5',
    edgecolor=COLORS['border'],
    linewidth=2,
    alpha=0.9,
    zorder=2
)
ax.add_patch(opt_box)

ax.text(opt_panel_x + 3, opt_panel_y + 3.6, 'OPTIMIZACIÓN LP',
        ha='center', va='center',
        fontsize=12, fontweight='bold',
        color=COLORS['text'])

opt_text = """Solver: CBC (20 threads)

Objetivo: min Σ(CAPEX + OPEX)

Timestep: Mensual (2015-2100)

Restricciones:
• Balance energético por carrier
• Capacidades máximas
• Eficiencias tecnológicas

Variables decisión:
• Capacidades (MW, MWh)
• Despacho temporal (MWh/mes)"""

ax.text(opt_panel_x + 3, opt_panel_y + 1.4, opt_text,
        ha='center', va='center',
        fontsize=9.5,
        color=COLORS['text'],
        family='monospace')

# Leyenda de carriers (superior derecha)
legend_x, legend_y = 19.5, 12.8
legend_elements = [
    mpatches.Patch(facecolor=COLORS['carrier_elec'], edgecolor=COLORS['border'], 
                   label='Electricidad (AC)', linewidth=1.5, alpha=0.7),
    mpatches.Patch(facecolor=COLORS['carrier_h2'], edgecolor=COLORS['border'],
                   label='Hidrógeno (H2)', linewidth=1.5, alpha=0.7),
    mpatches.Patch(facecolor=COLORS['carrier_water'], edgecolor=COLORS['border'],
                   label='Agua (dulce/mar)', linewidth=1.5, alpha=0.7),
]
legend = ax.legend(handles=legend_elements, loc='upper right', 
                  bbox_to_anchor=(0.99, 0.99),
                  fontsize=10, framealpha=0.95,
                  edgecolor=COLORS['border'], fancybox=True)
legend.set_title('Carriers', prop={'size': 11, 'weight': 'bold'})

legend.set_title('Carriers', prop={'size': 11, 'weight': 'bold'})

plt.tight_layout()

# Guardar figura
OUTPUT_DIR = Path('/home/aninotna/magister/tesis/justh2_pipeline/plots/tesis_data_section')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
output_path = OUTPUT_DIR / 'calliope_arquitectura_energetica.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\nDiagrama guardado en: {output_path}")

# Versión PDF para LaTeX
output_path_pdf = OUTPUT_DIR / 'calliope_arquitectura_energetica.pdf'
plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
print(f"Versión PDF guardada en: {output_path_pdf}")

print(f"\nDimensiones finales: {fig.get_size_inches()[0]:.1f} × {fig.get_size_inches()[1]:.1f} pulgadas")

plt.show()
