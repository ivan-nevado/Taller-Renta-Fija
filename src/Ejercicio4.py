import pandas as pd
import numpy as np
from datetime import datetime


# ------------------------------------------------------------
# Ejercicio 4. 
# ------------------------------------------------------------


# --- Helper: construir calendario de flujos (fechas y montos) usando misma lógica que valorar_bono ---
def build_cashflows(bono_row, valuation_date):
    """
    Devuelve una lista de (date, cashflow) desde valuation_date (solo flujos futuros)
    Usa la misma regla: si Callable == 'Y' usar Next Call Date como maturity.
    """
    # Determinar maturity a usar
    if bono_row["Callable"] == "Y" and pd.notna(bono_row["Next Call Date"]):
        maturity = pd.to_datetime(bono_row["Next Call Date"], dayfirst=True)
    else:
        maturity = pd.to_datetime(bono_row["Maturity"], dayfirst=True)

    freq = int(bono_row["Coupon Frequency"])
    coupon_rate = float(bono_row["Coupon"]) / 100.0
    nominal = 100.0

    # Construir fechas hacia atrás (como en el código previo)
    dates = [maturity]
    months_step = int(12 / freq)
    current_date = maturity
    while current_date > valuation_date:
        current_date = current_date - pd.DateOffset(months=months_step)
        dates.append(current_date)
    dates = sorted(dates)

    # Flujos futuros
    flows = []
    for d in dates:
        if d <= valuation_date:
            continue
        if d == maturity:
            cf = nominal + coupon_rate * nominal / freq
        else:
            cf = coupon_rate * nominal / freq
        flows.append((d, cf))
    return flows

# --- Helper: precio dado un yield anual nominal (compuesto m veces) ---
def price_from_yield(bono_row, valuation_date, y_annual):
    """
    Calcula precio limpio (PV) dado yield anual nominal y_annual.
    y_annual en decimal (ej. 0.03 = 3%)
    Compounding: periódico, m = coupon frequency.
    Utiliza ACT/365 para tiempos en años.
    """
    flows = build_cashflows(bono_row, valuation_date)
    if len(flows) == 0:
        return 0.0
    m = int(bono_row["Coupon Frequency"])
    y_period = y_annual / m

    pv = 0.0
    for (d, cf) in flows:
        # número de periodos hasta d (puede ser no entero si fechas imperfectas, así usamos tiempo en años * m)
        t_years = (d - valuation_date).days / 365.0
        n_periods = t_years * m
        # descontar con (1 + y_period)^{n_periods}
        pv += cf / ((1 + y_period) ** n_periods)
    return pv

# --- Encontrar yield (TIR) dado precio limpio de mercado (bisección) ---
def find_yield_from_price(bono_row, valuation_date, market_clean_price,
                          y_low= -0.5, y_high= 5.0, tol_price=1e-8, tol_y=1e-8, max_iter=200):
    """
    Busca yield anual nominal (decimal) tal que price_from_yield(...) ~= market_clean_price.
    Retorna yield anual o None si no converge.
    """
    def f(y):
        return price_from_yield(bono_row, valuation_date, y) - market_clean_price

    f_low = f(y_low)
    f_high = f(y_high)

    # Si signos iguales, intentar expandir (limitado)
    if f_low * f_high > 0:
        # trata algunos intentos de expansión
        factor = 2.0
        tries = 0
        while f_low * f_high > 0 and tries < 30:
            y_low *= factor
            y_high *= factor
            f_low = f(y_low)
            f_high = f(y_high)
            tries += 1

    if f_low * f_high > 0:
        return None

    a, b = y_low, y_high
    fa, fb = f_low, f_high
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        fm = f(m)
        if abs(fm) < tol_price or (b - a) / 2 < tol_y:
            return m
        if fa * fm <= 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return None

# --- Duración y convexidad por derivadas numéricas ---
def duration_convexity_numeric(bono_row, valuation_date, y_annual, dy=1e-4):
    """
    Calcula Modified Duration y Convexity numéricamente usando un bump dy en yield (decimal).
    Devuelve (Macaulay, Modified, Convexity)
    Convexity devuelta en unidades por (yield decimal)^2 (adimensional).
    """
    P = price_from_yield(bono_row, valuation_date, y_annual)
    P_up = price_from_yield(bono_row, valuation_date, y_annual + dy)
    P_down = price_from_yield(bono_row, valuation_date, y_annual - dy)
    # Modified Duration
    mod_dur = - (P_up - P_down) / (2 * P * dy)
    # Convexity
    conv = (P_up + P_down - 2 * P) / (P * (dy ** 2))
    # Macaulay: aproximación usando ModDur: Mac = ModDur * (1 + y_period)
    m = int(bono_row["Coupon Frequency"])
    y_period = y_annual / m
    macaulay = mod_dur * (1 + y_period)
    return macaulay, mod_dur, conv

# --- Ejemplo de ejecución para toda la cartera ---

direccionCurva = r"C:\Users\Joseph\Documents\MIAX\Modulo II\Taller-Renta-Fija\data\curvaESTR.csv"
direccionBonos = r"C:\Users\Joseph\Documents\MIAX\Modulo II\Taller-Renta-Fija\data\universo.csv"

df_curve = pd.read_csv(direccionCurva, sep=';', parse_dates=["Date"])
df_bonos = pd.read_csv(direccionBonos, sep=';', parse_dates=["Maturity", "Next Call Date"])

# Convertir fechas de la curva
df_curve["Date"] = pd.to_datetime(df_curve["Date"], dayfirst=True)

# Configuración
val_date = pd.Timestamp(2025, 10, 1)

results = []

for idx, row in df_bonos.iterrows():
    market_price = float(row["Price"]) if pd.notna(row["Price"]) else None
    if market_price is None or market_price == 0:
        results.append({
            "ISIN": row["ISIN"],
            "Yield": None,
            "Macaulay": None,
            "Modified": None,
            "Convexity": None
        })
        continue

    y = find_yield_from_price(row, val_date, market_price, y_low=-0.5, y_high=5.0)
    if y is None:
        # no converge, marcamos
        results.append({
            "ISIN": row["ISIN"],
            "Yield": None,
            "Macaulay": None,
            "Modified": None,
            "Convexity": None
        })
        continue

    mac, mod, conv = duration_convexity_numeric(row, val_date, y, dy=1e-4)
    results.append({
        "ISIN": row["ISIN"],
        "Yield": y,
        "Macaulay": mac,
        "Modified": mod,
        "Convexity": conv
    })

df_dur_conv = pd.DataFrame(results)
print(df_dur_conv)
