import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
#universoIvan = r"C:\Users\Iván\Desktop\Ivan\MIAX\Taller-Renta-Fija\data\universo.csv"
universoJoseph =r"C:\Users\Joseph\Documents\MIAX\Modulo II\Taller-Renta-Fija\data\universo.csv"
    
datos = pd.read_csv(universoJoseph, sep=';')

print(datos["Ccy"].drop_duplicates())


for col in ['Coupon Type', 'Callable', 'Seniority', 'Description', 'Maturity']:
    datos[col] = datos[col].astype(str).str.strip()

fixed = datos["Coupon Type"].str.contains("FIXED").sum()
floating = datos["Coupon Type"].str.contains("VARIABLE").sum() 
print("Número de bonos fijos: ",fixed)
print("Número de bonos flotantes: ", floating)
print()

perpetual = datos["Maturity"].isna().sum()
print("Número de bonos perpetuos: ", perpetual)
print()

no_callable = datos["Callable"].str.contains("N").sum()
callable = datos["Callable"].str.contains("Y").sum()
print("Número de bonos Callable: ",callable)
print("Número de bonos No Callable: ", no_callable)
print()

prelacion = datos.groupby("Seniority").size()
print("Prelación: ")
print(prelacion)
print()

sectors = datos.groupby("Industry Sector").size()
print("Sectores: ")
print(sectors)
print()

emisores = datos.groupby("Issuer").size()
print("Emisores: ")
print(emisores.sort_values(ascending=False).head(10))
print()

rating = datos.groupby("Rating").size()
print(rating.sort_values(ascending=False))
print()

spread = abs((datos["Bid Price"]).astype(float)-(datos["Ask Price"]).astype(float))
mid_price = (datos["Bid Price"] + datos["Ask Price"]) / 2
rel_spread = spread/mid_price
spread_bps = rel_spread * 10000
print(spread_bps)

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



