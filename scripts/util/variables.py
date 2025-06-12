import pandas as pd

data = {
    "Categoría": [
        "Climática", "Climática", "Climática", "Climática", "Climática",
        "Climática", "Climática", "Recursos naturales", "Recursos naturales",
        "Recursos naturales", "Infraestructura energética", "Infraestructura energética",
        "Infraestructura energética", "Infraestructura energética", "Capacidad adaptativa",
        "Capacidad adaptativa", "Capacidad adaptativa", "Capacidad adaptativa",
        "Riesgos naturales", "Riesgos naturales", "Indicadores compuestos", "Indicadores compuestos"
    ],
    "Variable": [
        "Temperatura máxima (tmax)", "Precipitación total anual (PRCPTOT)",
        "Días secos consecutivos (CDD)", "Índice de sequía SPI12",
        "Radiación solar global", "Rx1day (precipitación extrema en un día)",
        "PCI (Índice de concentración de lluvia)", "Disponibilidad de agua (caudales, acuíferos)",
        "Elevación y pendiente", "Cobertura vegetal y uso de suelo",
        "Ubicación de plantas de H₂ y ERNC", "Capacidad de almacenamiento energético",
        "Red eléctrica y conexiones", "Flexibilidad en la demanda",
        "Inversión pública en energía", "Educación técnica en hidrógeno",
        "Presencia de políticas climáticas locales", "Índice de desarrollo humano",
        "Frecuencia de eventos extremos", "Historial de impacto en infraestructura",
        "Índice de vulnerabilidad climática", "Índice de diversificación energética"
    ],
    "Fuente sugerida": [
        "ERA5 / CR2MET", "CR2MET / CHIRPS", "CR2MET", "CHIRPS / CR2MET",
        "SolarGIS / ERA5", "CR2MET", "CR2MET", "DGA / AQUASTAT",
        "SRTM / DEMs", "MODIS / CONAF", "Ministerio de Energía / Mapa ERNC",
        "Proyectos / Empresa Nacional", "Ministerio de Energía / SEC",
        "Estudios técnicos", "Presupuesto nacional / BID / CEPAL",
        "SENCE / Mineduc", "MINENERGÍA / MMA", "PNUD / INE",
        "ONEMI / Senapred", "Bases de datos históricas", "Estudios previos / IPCC",
        "Ministerio de Energía / CEPAL"
    ],
    "Unidad o escala": [
        "°C", "mm/año", "días", "índice", "kWh/m²", "mm", "índice", "m³/s o nivel freático",
        "metros / %", "categoría/cobertura", "ubicación georreferenciada", "MWh o toneladas H₂",
        "mapa de red / nodos", "% de participación", "$/año o % PIB", "número de programas o egresados",
        "presencia/ausencia o puntuación", "valor entre 0 y 1", "conteo anual o decenal", "eventos reportados",
        "valor normalizado (0–1)", "% energía renovable vs total"
    ]
}

df_resiliencia = pd.DataFrame(data)
print(df_resiliencia)