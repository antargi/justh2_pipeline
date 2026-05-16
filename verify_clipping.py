"""
Verificar que la línea 520 de resilience_lib.py tenga el clipping correcto
"""

import os

filepath = "/home/aninotna/magister/tesis/justh2_pipeline/scripts/idroverdi_autoencoder_3/resilience_lib.py"

print("Buscando la línea donde se calcula IRCT...\n")

with open(filepath, 'r') as f:
    lines = f.readlines()

# Buscar líneas importantes
for i, line in enumerate(lines, 1):
    if 'IRCT = ' in line and '**' in line:
        print(f"Línea {i}: {line.rstrip()}")
        # Mostrar contexto
        for j in range(max(0, i-5), min(len(lines), i+5)):
            prefix = ">>> " if j+1 == i else "    "
            print(f"{prefix}{j+1}: {lines[j].rstrip()}")
        print()
    
    if 'S_E = np.clip' in line:
        print(f"Línea {i} (S_E clip): {line.rstrip()}")
        print()
    
    if 'IRCT = np.clip' in line:
        print(f"Línea {i} (IRCT clip): {line.rstrip()}")
        print()
    
    if 'A_future = np.clip' in line:
        print(f"Línea {i} (A_future clip): {line.rstrip()}")
        # Mostrar contexto
        for j in range(max(0, i-3), min(len(lines), i+10)):
            prefix = ">>> " if j+1 == i else "    "
            print(f"{prefix}{j+1}: {lines[j].rstrip()}")
        print()

print("\n✓ Búsqueda completada")
