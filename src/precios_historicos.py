import pandas as pd
import numpy as np


# Cargar y analizar precios históricos
precios_historicos_path = r"C:\Users\Joseph\Documents\MIAX\Modulo II\Taller-Renta-Fija\data\precios_historicos_universo.csv"
precios = pd.read_csv(precios_historicos_path, sep=';', low_memory=False)
precios = precios.replace({"#N/D": np.nan})
print(f"📈 ANÁLISIS DE PRECIOS HISTÓRICOS")
print(f"Dimensiones: {precios.shape}")
print(f"Período: {precios.columns[1]} a {precios.columns[-1]}")

# Convertir a formato adecuado
precios = precios.rename(columns={precios.columns[0]: "ISIN"})

date_cols = precios.columns[1:]
new_cols = ["ISIN"] + list(pd.to_datetime(date_cols, dayfirst=True))
precios.columns = new_cols

# Transponer para tener fechas como índice
precios_clean = precios.set_index("ISIN").T
precios_clean.index = pd.to_datetime(precios_clean.index)
precios_clean = precios_clean.sort_index()

# Limpiar ISINs
precios_clean.columns = precios_clean.columns.str.replace(' Corp', '', regex=False).str.strip()

# Conversión numérica
for col in precios_clean.columns:
    precios_clean[col] = pd.to_numeric(
        precios_clean[col].astype(str).str.replace(',', '.'), 
        errors='coerce'
    )

print(f"Precios procesados: {precios_clean.shape}")
print(f"Bonos con datos: {precios_clean.count(axis=0).sum()}")
print(f"✅ Datos de precios listos para análisis")