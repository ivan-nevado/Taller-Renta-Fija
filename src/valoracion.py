import pandas as pd
import numpy as np
from datetime import datetime

# ------------------------------------------------------------
# Interpolación exponencial para factores de descuento
# ------------------------------------------------------------
def discount_interpolate(curve_df, target_date, valuation_date):
    """
    Interpola exponencialmente el factor de descuento para una fecha objetivo.
    curve_df: dataframe con columnas ["Date", "Discount"]
    """
    curve_df = curve_df.copy()  # para no modificar el original
    curve_df["Date"] = pd.to_datetime(curve_df["Date"], dayfirst=True)

    # Calcular t en años ACT/365
    t = (target_date - valuation_date).days / 365

    # Si la fecha está exactamente en la curva
    if target_date in curve_df["Date"].values:
        return float(curve_df.loc[curve_df["Date"] == target_date, "Discount"].values[0])

    # Buscar puntos adyacentes
    curve_df = curve_df.sort_values("Date")

    # Convertir fechas a t
    curve_df["t"] = (curve_df["Date"] - valuation_date).dt.days / 365

    # Puntos antes y después
    lower = curve_df[curve_df["t"] <= t].tail(1)
    upper = curve_df[curve_df["t"] >= t].head(1)

    t1, DF1 = float(lower["t"]), float(lower["Discount"])
    t2, DF2 = float(upper["t"]), float(upper["Discount"])

    # α para interpolación exponencial
    alpha = (t2 - t) / (t2 - t1)

    return DF1**alpha * DF2**(1 - alpha)

# ------------------------------------------------------------
# Función principal de valoración del bono
# ------------------------------------------------------------
def valorar_bono(bono_row, curve_df, valuation_date, spread_crediticio=0.0):
    """
    bono_row: una fila del dataframe de bonos (pandas Series)
    curve_df: curva ESTR con columnas [Date, Discount]
    valuation_date: datetime
    spread_crediticio: spread en términos continuos (decimal)
    """
    
    # 1) Determinar vencimiento según reglas del enunciado
    if bono_row["Callable"] == "Y":
        maturity = pd.to_datetime(bono_row["Next Call Date"])
    else:
        maturity = pd.to_datetime(bono_row["Maturity"])

    # 2) Construcción del calendario de flujos
    freq = bono_row["Coupon Frequency"]
    coupon_rate = bono_row["Coupon"] / 100
    nominal = 100

    # Generar fechas de cupón hacia atrás desde maturity
    dates = [maturity]
    while dates[-1] > valuation_date:
        days = 365 / freq
        dates.append(dates[-1] - pd.Timedelta(days=days))
    dates = sorted(dates)

    # 3) Cupón corrido
    # Fecha anterior y siguiente a la valoración
    prev_coupon = max(d for d in dates if d <= valuation_date)
    next_coupon = min(d for d in dates if d > valuation_date)

    ai_days = (valuation_date - prev_coupon).days
    period_days = (next_coupon - prev_coupon).days
    accrued_interest = coupon_rate * nominal * (ai_days / period_days)

    # 4) Valoración del bono (dirty price)
    dirty_price = 0

    for d in dates:
        if d <= valuation_date:
            continue

        # factor de descuento interpolado desde curva
        DF = discount_interpolate(curve_df, d, valuation_date)

        # aplicar spread
        t = (d - valuation_date).days / 365
        DF_adj = DF * np.exp(-spread_crediticio * t)

        # flujo
        if d == maturity:
            flujo = nominal + coupon_rate * nominal / freq
        else:
            flujo = coupon_rate * nominal / freq

        dirty_price += flujo * DF_adj

    # 5) Precio limpio
    clean_price = dirty_price - accrued_interest

    return clean_price, accrued_interest, dirty_price

direccionCurva = r"C:\Users\Joseph\Documents\MIAX\Modulo II\Taller-Renta-Fija\data\curvaESTR.csv"
direccionBonos = r"C:\Users\Joseph\Documents\MIAX\Modulo II\Taller-Renta-Fija\data\universo.csv"

curve = pd.read_csv(direccionCurva, sep=';', parse_dates=["Date"])
print(curve.columns)
bonos = pd.read_csv(direccionBonos, sep=';',parse_dates=["Maturity", "Next Call Date"])
print(bonos.columns)
valuation_date = datetime(2025, 10, 1)


""""
row = bonos.loc[0]   # primer bono

clean, ai, dirty = valorar_bono(row, curve, valuation_date, spread_crediticio=0)

print("Precio limpio:", clean)
print("Cupón corrido:", ai)
print("Precio sucio:", dirty)
"""

for i in range(bonos.size-1):
    row = bonos.loc[i]
    clean, ai, dirty = valorar_bono(row, curve, valuation_date, spread_crediticio=0)
    print("Bono:", i)
    print("Precio limpio:", clean)
    print("Cupón corrido:", ai)
    print("Precio sucio:", dirty)
    print()
    
