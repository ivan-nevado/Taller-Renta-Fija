import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
universo = r"C:\Users\Iván\Desktop\Ivan\MIAX\Taller-Renta-Fija\data\universo.csv"
    
datos = pd.read_csv(universo, sep=';')

print(datos["Ccy"].drop_duplicates())


for col in ['Coupon Type', 'Callable', 'Seniority', 'Description', 'Maturity']:
    datos[col] = datos[col].astype(str).str.strip()

fixed = datos["Coupon Type"].str.contains("FIXED").sum()
floating = datos["Coupon Type"].str.contains("VARIABLE").sum() 

perpetual = datos["Maturity"].isna().sum()

no_callable = datos["Callable"].str.contains("N").sum()
callable = datos["Callable"].str.contains("Y").sum()

prelacion = datos.groupby("Seniority").size()

sectors = datos.groupby("Industry Sector").size()

emisores = datos.groupby("Issuer").size()

rating = datos.groupby("Rating").size()

spread = abs((datos["Bid Price"]).astype(float)-(datos["Ask Price"]).astype(float))
mid_price = (datos["Bid Price"] + datos["Ask Price"]) / 2
rel_spread = spread/mid_price
spread_bps = rel_spread * 10000

# Convertir a float si no lo está
datos["Outstanding Amount"] = datos["Outstanding Amount"].astype(float)

# Histograma
plt.figure(figsize=(10,6))
plt.hist(datos["Outstanding Amount"], bins=30, color='skyblue', edgecolor='black')
plt.title("Histograma del Nominal Vivo")
plt.xlabel("Outstanding Amount (millones)")
plt.ylabel("Número de bonos")
plt.grid(True)
plt.show()

