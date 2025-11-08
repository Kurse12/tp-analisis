# ==========================
# TP EPH: Evolución de Indicadores Laborales (2016-2025)
# Comparación: BA vs CBA
# ==========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

sns.set_style("whitegrid")

# ==========================
# Cargar dataset limpio
# ==========================
ruta = "microdatos_eph_limpio.csv"
df = pd.read_csv(ruta, encoding='latin1')

# ==========================
# Mapear AGLOMERADO a nombres
# ==========================
aglo_map = {2: 'BA', 6: 'CBA'}  # Ajustar según códigos del CSV
df['AGLOMERADO'] = df['AGLOMERADO'].map(aglo_map)

# ==========================
# Crear/sobrescribir columna PERIODO
# ==========================
df['PERIODO'] = df['ANO4'].astype(str) + 'T' + df['TRIMESTRE'].astype(str)

# ==========================
# Crear rangos de edad
# ==========================
bins = [14, 24, 44, 64, 99]
labels = ['14-24', '25-44', '45-64', '65+']
df['edad_rango'] = pd.cut(df['CH06'], bins=bins, labels=labels, right=True)

# ==========================
# Filtrar por aglomerado
# ==========================
df_ba = df[df['AGLOMERADO'] == 'BA'].copy()
df_cba = df[df['AGLOMERADO'] == 'CBA'].copy()

# ==========================
# Función para indicadores univariado
# ==========================
def calcular_indicadores(df):
    df = df.copy()
    grouped = df.groupby('PERIODO', as_index=False)
    indicadores = grouped.apply(
        lambda x: pd.Series({
            'tasa_desocupacion': (x['CONDICION'] == 'Desocupado').sum() / (x['CONDICION'].isin(['Ocupado','Desocupado'])).sum() * 100,
            'tasa_empleo': (x['CONDICION'] == 'Ocupado').sum() / len(x) * 100,
            'tasa_actividad': (x['CONDICION'].isin(['Ocupado','Desocupado'])).sum() / len(x) * 100,
            'ingreso_promedio': x['P21'].mean(),
            'ingreso_mediana': x['P21'].median()
        })
    ).reset_index(drop=True)
    return indicadores

# ==========================
# Función para indicadores multivariado
# ==========================
def indicadores_multivariado(df, group_cols=['SEXO','edad_rango']):
    df = df.copy()
    grouped = df.groupby(['PERIODO'] + group_cols, as_index=False)
    indicadores = grouped.apply(
        lambda x: pd.Series({
            'tasa_desocupacion': (x['CONDICION'] == 'Desocupado').sum() / (x['CONDICION'].isin(['Ocupado','Desocupado'])).sum() * 100,
            'tasa_empleo': (x['CONDICION'] == 'Ocupado').sum() / len(x) * 100,
            'tasa_actividad': (x['CONDICION'].isin(['Ocupado','Desocupado'])).sum() / len(x) * 100,
            'ingreso_promedio': x['P21'].mean(),
            'ingreso_mediana': x['P21'].median()
        })
    ).reset_index(drop=True)
    return indicadores

# ==========================
# Calcular indicadores
# ==========================
# Univariado
ind_ba = calcular_indicadores(df_ba)
ind_ba['AGLOMERADO'] = 'BA'
ind_cba = calcular_indicadores(df_cba)
ind_cba['AGLOMERADO'] = 'CBA'
ind_total = pd.concat([ind_ba, ind_cba], ignore_index=True)

# Multivariado
ind_mult_ba = indicadores_multivariado(df_ba)
ind_mult_ba['AGLOMERADO'] = 'BA'
ind_mult_cba = indicadores_multivariado(df_cba)
ind_mult_cba['AGLOMERADO'] = 'CBA'
ind_mult_total = pd.concat([ind_mult_ba, ind_mult_cba], ignore_index=True)

# ==========================
# Exportar tablas
# ==========================
ind_total.to_csv("indicadores_univariado.csv", index=False)
ind_mult_total.to_csv("indicadores_multivariado.csv", index=False)

# ==========================
# Función para gráficos
# ==========================
def graficar_indicador(ind_df, columna, ylabel, title):
    plt.figure(figsize=(12,5))
    sns.lineplot(data=ind_df, x='PERIODO', y=columna, hue='AGLOMERADO', marker='o')
    plt.xticks(rotation=45)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

graficar_indicador(ind_total, 'tasa_desocupacion', 'Tasa de Desocupación (%)', 'Evolución TDes (BA vs CBA)')
graficar_indicador(ind_total, 'tasa_empleo', 'Tasa de Empleo (%)', 'Evolución TEmp (BA vs CBA)')
graficar_indicador(ind_total, 'tasa_actividad', 'Tasa de Actividad (%)', 'Evolución TAct (BA vs CBA)')
graficar_indicador(ind_total, 'ingreso_promedio', 'Ingreso Promedio', 'Evolución Ingreso Promedio (BA vs CBA)')

# ==========================
# Imputación de ingresos faltantes
# ==========================
# Filas con ingresos
df_model = df.dropna(subset=['P21']).copy()
df_model_dummy = pd.get_dummies(df_model, columns=['SEXO','edad_rango','CONDICION'], drop_first=True)
X = df_model_dummy.drop(columns=['P21','PERIODO','AGLOMERADO'])
y = df_model_dummy['P21']

# Entrenar modelo
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluación
y_pred = model.predict(X_test)
r2 = r2_score(y_test,y_pred)
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
print(f"R²: {r2:.3f}, RMSE: {rmse:.2f}")

# Filas faltantes
df_missing = df[df['P21'].isna()].copy()
df_missing_dummy = pd.get_dummies(df_missing, columns=['SEXO','edad_rango','CONDICION'], drop_first=True)

# Alinear columnas
for col in X.columns:
    if col not in df_missing_dummy.columns:
        df_missing_dummy[col] = 0
df_missing_dummy = df_missing_dummy[X.columns]

# Imputar
df.loc[df['P21'].isna(),'P21'] = model.predict(df_missing_dummy)
print("✅ Imputación completada. Filas sin P21:", df['P21'].isna().sum())
