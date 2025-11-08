import pandas as pd
import glob

# 🔹 Cambiá la ruta por la carpeta donde tenés todos los archivos individuales
ruta = "C:/Users/aethe/OneDrive/Escritorio/tp-analisis/microdatos/"  

# 🔹 Busca todos los archivos con t minúscula, sin importar si terminan en xls o xlsx
archivos = glob.glob(ruta + "usu_individual_t*.xls*")

print(f"Se encontraron {len(archivos)} archivos.")
if len(archivos) == 0:
    print("⚠️ No se encontraron archivos. Revisá la ruta o el nombre exacto de los archivos.")
else:
    dataframes = []
    for archivo in archivos:
        try:
            print("Leyendo:", archivo)
            df = pd.read_excel(archivo)
            df['origen'] = archivo.split('/')[-1]  # opcional, guarda el nombre
            dataframes.append(df)
        except Exception as e:
            print(f"Error leyendo {archivo}: {e}")

    # Concatenar todo
    df_total = pd.concat(dataframes, ignore_index=True)
    df_total.to_csv("microdatos_eph_2016_2025.csv", index=False)
    print("✅ Archivo combinado guardado como microdatos_eph_2016_2025.csv")
