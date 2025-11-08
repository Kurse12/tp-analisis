import pandas as pd
import numpy as np
import os

ruta = "microdatos_eph_2016_2025.csv"

# ==========================
# 1. Configuración
# ==========================
chunk_size = 500_000  # filas por bloque
encoding = 'latin1'   # evita problemas de Unicode
dfs = []

print("🔹 Leyendo archivo en chunks... Esto puede tardar varios minutos.")

# ==========================
# 2. Leer en chunks
# ==========================
for i, chunk in enumerate(pd.read_csv(ruta, encoding=encoding, chunksize=chunk_size, on_bad_lines="skip")):
    print(f"  • Procesando chunk {i+1}")
    
    # --- Seleccionar columnas relevantes ---
    cols = ['ANO4','TRIMESTRE','AGLOMERADO','CH04','CH06','ESTADO','CAT_OCUP','P21','PONDIIO']
    chunk = chunk[[c for c in cols if c in chunk.columns]]
    
    # --- Filtrar población >=14 años ---
    if 'CH06' in chunk.columns:
        chunk = chunk[chunk['CH06'] >= 14]
    
    # --- Limpiar ingresos ---
    if 'P21' in chunk.columns:
        chunk['P21'] = pd.to_numeric(chunk['P21'], errors='coerce')
        chunk['P21'] = chunk['P21'].replace({0: np.nan, 9999999: np.nan})
    
    # --- Mapear variables descriptivas ---
    if 'CH04' in chunk.columns:
        chunk['SEXO'] = chunk['CH04'].map({1: 'Varón', 2: 'Mujer'})
    
    if 'ESTADO' in chunk.columns:
        chunk['CONDICION'] = chunk['ESTADO'].map({
            1: 'Ocupado',
            2: 'Desocupado',
            3: 'Inactivo'
        })
    
    # --- Crear variable de período ---
    if 'ANO4' in chunk.columns and 'TRIMESTRE' in chunk.columns:
        chunk['PERIODO'] = chunk['ANO4'].astype(str) + 'T' + chunk['TRIMESTRE'].astype(str)
    
    dfs.append(chunk)

# ==========================
# 3. Concatenar todos los chunks
# ==========================
df = pd.concat(dfs, ignore_index=True)
print(f"\n✅ Archivo completo cargado con {len(df):,} filas y {len(df.columns)} columnas.")

# ==========================
# 4. Eliminar duplicados y resetear índice
# ==========================
df = df.drop_duplicates().reset_index(drop=True)

# ==========================
# 5. Guardar resultados
# ==========================
csv_salida = "microdatos_eph_limpio.csv"
parquet_salida = "microdatos_eph_limpio.parquet"

df.to_csv(csv_salida, index=False)
df.to_parquet(parquet_salida, index=False)

print(f"\n✅ Dataset limpio guardado en:")
print(f"   • {os.path.abspath(csv_salida)}")
print(f"   • {os.path.abspath(parquet_salida)}")

# ==========================
# 6. Resumen rápido
# ==========================
if 'CONDICION' in df.columns:
    print("\nDistribución de la variable CONDICION:")
    print(df['CONDICION'].value_counts(dropna=False))
else:
    print("⚠️ Columna CONDICION no encontrada.")
