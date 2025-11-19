import pandas as pd
import numpy as np
from datetime import datetime


# ---------------------------------------------------------------------------
# 1. Funciones de Ayuda. Interpolación exponencial para factores de descuento
# ---------------------------------------------------------------------------

def discount_interpolate(curve_df, target_date, valuation_date):
    """Interpola exponencialmente el factor de descuento."""
    # Convertir a datetime si no lo es
    target_date = pd.to_datetime(target_date)
    
    # Si la fecha es anterior o igual a valoración, el DF es 1 (o manejado fuera)
    if target_date <= valuation_date:
        return 1.0

    # Buscar coincidencia exacta
    if target_date in curve_df["Date"].values:
        return float(curve_df.loc[curve_df["Date"] == target_date, "Discount"].values[0])

    # Calcular t para la fecha objetivo
    t = (target_date - valuation_date).days / 365.0

    # Preparar curva para búsqueda
    curve_sorted = curve_df.sort_values("Date").reset_index(drop=True)
    curve_sorted["t_curve"] = (curve_sorted["Date"] - valuation_date).dt.days / 365.0

    # Encontrar puntos lower (t1) y upper (t2)
    lower_df = curve_sorted[curve_sorted["t_curve"] <= t]
    upper_df = curve_sorted[curve_sorted["t_curve"] >= t]

    # Extrapolación si la fecha está fuera del rango de la curva (Safety check)
    if lower_df.empty:
        return curve_sorted.iloc[0]["Discount"]
    if upper_df.empty:
        return curve_sorted.iloc[-1]["Discount"]

    # Datos para interpolación
    t1 = float(lower_df.iloc[-1]["t_curve"])
    df1 = float(lower_df.iloc[-1]["Discount"])
    
    t2 = float(upper_df.iloc[0]["t_curve"])
    df2 = float(upper_df.iloc[0]["Discount"])

    # Evitar división por cero si t1 == t2
    if t2 == t1:
        return df1

    # Interpolación Exponencial
    alpha = (t2 - t) / (t2 - t1)
    return (df1 ** alpha) * (df2 ** (1 - alpha))

# ------------------------------------------------------------
# Función principal de valoración del bono
# ------------------------------------------------------------
def valorar_bono(bono_row, curve_df, valuation_date, spread_crediticio=0.0):
    # Determinar fecha de vencimiento (Call vs Maturity)
    if bono_row["Callable"] == "Y" and pd.notna(bono_row["Next Call Date"]):
        maturity = pd.to_datetime(bono_row["Next Call Date"], dayfirst=True)
    else:
        maturity = pd.to_datetime(bono_row["Maturity"], dayfirst=True)

    # Si el bono ya venció antes de la fecha de valoración
    if maturity <= valuation_date:
        return 0.0, 0.0, 0.0

    freq = int(bono_row["Coupon Frequency"])
    coupon_rate = float(bono_row["Coupon"]) / 100.0
    nominal = 100.0
    
    # Generación de fechas de cupón (hacia atrás desde maturity)
    # Usamos DateOffset para meses exactos en vez de dividir 365
    dates = [maturity]
    months_step = int(12 / freq)
    
    current_date = maturity
    while current_date > valuation_date:
        current_date = current_date - pd.DateOffset(months=months_step)
        dates.append(current_date)
    
    dates = sorted(dates)
    
    # Filtrar fechas futuras
    future_dates = [d for d in dates if d > valuation_date]
    
    # Cálculo de Cupón Corrido (ACT/365)
    # Buscamos la fecha de inicio del cupón actual
    prev_coupon_date = [d for d in dates if d <= valuation_date]
    if prev_coupon_date:
        prev_coupon = prev_coupon_date[-1]
    else:
        # Si no hay fecha previa generada, asumimos periodo completo hacia atrás
        prev_coupon = future_dates[0] - pd.DateOffset(months=months_step)

    days_accrued = (valuation_date - prev_coupon).days
    accrued_interest = (days_accrued / 365.0) * coupon_rate * nominal

    # Valoración de flujos (Precio Sucio)
    dirty_price = 0.0
    
    for d in future_dates:
        # Factor de descuento interpolado
        DF = discount_interpolate(curve_df, d, valuation_date)
        
        # Ajuste por spread de crédito: DF_risky = DF_riskfree * exp(-spread * t)
        t_flow = (d - valuation_date).days / 365.0
        DF_adj = DF * np.exp(-spread_crediticio * t_flow)
        
        # Flujo de caja
        if d == maturity:
            cashflow = nominal + (coupon_rate * nominal / freq)
        else:
            cashflow = (coupon_rate * nominal / freq)
            
        dirty_price += cashflow * DF_adj

        clean_price = dirty_price - accrued_interest
    return clean_price, accrued_interest, dirty_price


# ------------------------------------------------------------
# Ejecución Principal Ejercicio 2
# ------------------------------------------------------------

direccionCurva = r"C:\Users\Joseph\Documents\MIAX\Modulo II\Taller-Renta-Fija\data\curvaESTR.csv"
direccionBonos = r"C:\Users\Joseph\Documents\MIAX\Modulo II\Taller-Renta-Fija\data\universo.csv"

df_curve = pd.read_csv(direccionCurva, sep=';', parse_dates=["Date"])
df_bonos = pd.read_csv(direccionBonos, sep=';', parse_dates=["Maturity", "Next Call Date"])

# Convertir fechas de la curva
df_curve["Date"] = pd.to_datetime(df_curve["Date"], dayfirst=True)

# Configuración
val_date = pd.Timestamp(2025, 10, 1)

results = []

print(f"{'ISIN':<15} | {'Market Price':<12} | {'Clean Calc':<12} | {'Accrued':<10} | {'Dirty':<12} | {'Diff':<10}")
print("-" * 80)

for idx, row in df_bonos.iterrows():
    clean, ai, dirty = valorar_bono(row, df_curve, val_date, spread_crediticio=0.0)
    
    # Precio limpio de mercado
    market_price = float(row["Price"]) if pd.notna(row["Price"]) else 0.0
    diff = clean - market_price
    
    # Print ordenado
    #print(f"{row['ISIN']:<15} | {market_price:<12.3f} | {clean:<12.3f} | {ai:<10.3f} | {dirty:<12.3f} | {diff:<10.3f}")
    
    results.append({
        "ISIN": row["ISIN"],
        "Issuer": row["Issuer"],
        "Market_Price": market_price,
        "Calculated_Clean": clean,
        "Accrued_Interest": ai,
        "Calculated_Dirty": dirty,
        "Difference": diff,
        "Maturity_Used": row["Next Call Date"] if row["Callable"] == "Y" else row["Maturity"]
    })


# Convertir a DataFrame para análisis posterior si se desea
df_results = pd.DataFrame(results)


# ------------------------------------------------------------
# Ejercicio 3. Calculo de Spreads
# ------------------------------------------------------------

def find_parallel_spread_for_bond(bono_row, curve_df, valuation_date, market_clean_price,
                                  valorar_bono, s_low=-0.10, s_high=1.00,
                                  tol_price=1e-6, tol_s=1e-8, max_iter=80):
    """
    Encuentra spread paralelo s (en unidades decimales, e.g. 0.01 = 100 bp)
    tal que valorar_func(..., spread_crediticio=s)[0] ~= market_clean_price.

    Parametros:
      bono_row, curve_df, valuation_date : datos
      market_clean_price : float (precio limpio de mercado)
      valorar_func : función que devuelve (clean, accrued, dirty) dado spread
      s_low, s_high : bounds iniciales para bisección (decimales)
      tol_price : tolerancia en precio (cuando parar)
      tol_s : tolerancia en spread
      max_iter : iteraciones máximas
    Retorna:
      spread implicado (float) o None si falla (no converge o no cambia signo)
    """

    def f(s):
        clean, ai, dirty = valorar_bono(bono_row, curve_df, valuation_date, spread_crediticio=s)
        return clean - market_clean_price

    # Valores en los extremos
    f_low = f(s_low)
    f_high = f(s_high)

    # Si ya están dentro de tolerancia en extremos
    if abs(f_low) <= tol_price:
        return s_low
    if abs(f_high) <= tol_price:
        return s_high

    # Si los signos no son opuestos, expandimos los límites de búsqueda gradualmente
    if f_low * f_high > 0:
        # intentamos expandir simétricamente hasta cierto número de intentos
        expand_factor = 2.0
        tries = 0
        while f_low * f_high > 0 and tries < 20:
            s_low *= expand_factor
            s_high *= expand_factor
            f_low = f(s_low)
            f_high = f(s_high)
            tries += 1

    if f_low * f_high > 0:
        # No se encontró intervalo con signos opuestos: root finding por bisección no aplicable
        return None

    # Bisección
    a, b = s_low, s_high
    fa, fb = f_low, f_high

    for _ in range(max_iter):
        m = 0.5 * (a + b)
        fm = f(m)

        if abs(fm) <= tol_price or (b - a) / 2.0 < tol_s:
            return m

        # Decide subintervalo
        if fa * fm <= 0:
            b, fb = m, fm
        else:
            a, fa = m, fm

    # No convergió en max_iter
    return None

# ------------------------------------------------------------
# Ejecución Principal Ejercicio 3
# ------------------------------------------------------------

print(f"{'ISIN':<15} | {'Market Price':<12} | {'Implied Spread (bps)':<20}")
print("-" * 60)

spread_results = []

for idx, row in df_bonos.iterrows():
    # Precio limpio de mercado
    market_price = float(row["Price"]) if pd.notna(row["Price"]) else None

    if market_price is None or market_price == 0:
        implied_spread = None
    else:
        implied_spread = find_parallel_spread_for_bond(
            row, df_curve, val_date,
            market_price,
            valorar_bono,
            s_low=-0.05,   # -5% = -500 bps
            s_high=1.00    # +100% = +10 000 bps
        )

    # Convertimos a bps para que sea legible
    spread_bps = implied_spread * 10000 if implied_spread is not None else None

    #print(f"{row['ISIN']:<15} | {market_price:<12.3f} | {spread_bps if spread_bps is not None else 'N/A':<20}")

    spread_results.append({
        "ISIN": row["ISIN"],
        "Issuer": row["Issuer"],
        "Market_Price": market_price,
        "Implied_Spread_bps": spread_bps,
        "Raw_Spread_decimal": implied_spread
    })

df_spreads = pd.DataFrame(spread_results)

