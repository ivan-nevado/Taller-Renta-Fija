import numpy as np
import pandas as pd

# Cargar y analizar precios históricos varios
fichero_varios = r"C:\Users\Joseph\Documents\MIAX\Modulo II\Taller-Renta-Fija\data\precios_historicos_varios.csv"
precios_varios = pd.read_csv(fichero_varios, sep=';', index_col=0, parse_dates=True, dayfirst=True)
precios_varios = precios_varios.replace({'#N/D': np.nan})

print(f"📊 ANÁLISIS DE PRECIOS HISTÓRICOS VARIOS")
print(f"Dimensiones: {precios_varios.shape}")
print(f"Período: {precios_varios.index[0]} a {precios_varios.index[-1]}")
print(f"\nInstrumentos disponibles:")
for col in precios_varios.columns:
    non_null = precios_varios[col].count()
    print(f"  {col}: {non_null} observaciones")

# Convertir a numérico
for col in precios_varios.columns:
    precios_varios[col] = pd.to_numeric(precios_varios[col], errors='coerce')

print(f"\n📈 INSTRUMENTOS CLAVE:")
print(f"• ITRAXX Main: Índice CDS grado inversión (IG)")
print(f"• ITRAXX XOVER: Índice CDS alto rendimiento (HY)")
print(f"• DU1/OE1/RX1: Futuros alemanes (Schatz/BOBL/BUND)")
print(f"• RECMTREU: Benchmark índice Total Return")

# Cargar y analizar curva ESTR
fichero_curva = r"C:\Users\Joseph\Documents\MIAX\Modulo II\Taller-Renta-Fija\data\curvaESTR.csv"
curva_estr = pd.read_csv(fichero_curva, sep=';')
curva_estr['Date'] = pd.to_datetime(curva_estr['Date'], dayfirst=True)

print(f"\n📊 ANÁLISIS CURVA €STR")
print(f"Fecha de referencia: {curva_estr['Date'].iloc[0]}")
print(f"Puntos de la curva: {len(curva_estr)}")
print(f"Vencimiento más largo: {curva_estr['Date'].iloc[-1]}")

# Mostrar algunos puntos clave
print(f"\nPuntos clave de la curva:")
key_points = [1, 5, 10, 15, 20, -1]  # 1Y, 5Y, 10Y, 15Y, 20Y, último
for i in key_points:
    if i == -1:
        idx = len(curva_estr) - 1
    else:
        idx = min(i, len(curva_estr) - 1)
    
    date = curva_estr['Date'].iloc[idx]
    rate = curva_estr['Market Rate'].iloc[idx]
    zero_rate = curva_estr['Zero Rate'].iloc[idx]
    discount = curva_estr['Discount'].iloc[idx]
    
    years = (date - curva_estr['Date'].iloc[0]).days / 365.25
    print(f"  {years:.1f}Y: Market={rate:.3f}%, Zero={zero_rate:.3f}%, DF={discount:.6f}")

print(f"\n✅ Datos adicionales procesados correctamente")