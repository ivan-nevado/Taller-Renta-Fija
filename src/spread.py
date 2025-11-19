import pandas as pd
import numpy as np
from datetime import datetime
import valoracion

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
        clean, ai, dirty = valoracion.valorar_bono(bono_row, curve_df, valuation_date, spread_crediticio=s)
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

direccionCurva = r"C:\Users\Joseph\Documents\MIAX\Modulo II\Taller-Renta-Fija\data\curvaESTR.csv"
direccionBonos = r"C:\Users\Joseph\Documents\MIAX\Modulo II\Taller-Renta-Fija\data\universo.csv"

df_curve = pd.read_csv(direccionCurva, sep=';', parse_dates=["Date"])
df_bonos = pd.read_csv(direccionBonos, sep=';', parse_dates=["Maturity", "Next Call Date"])

# Convertir fechas de la curva
df_curve["Date"] = pd.to_datetime(df_curve["Date"], dayfirst=True)

# Configuración
val_date = pd.Timestamp(2025, 10, 1)

results = []

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
            valoracion.valorar_bono,
            s_low=-0.05,   # -5% = -500 bps
            s_high=1.00    # +100% = +10 000 bps
        )

    # Convertimos a bps para que sea legible
    spread_bps = implied_spread * 10000 if implied_spread is not None else None

    print(f"{row['ISIN']:<15} | {market_price:<12.3f} | {spread_bps if spread_bps is not None else 'N/A':<20}")

    spread_results.append({
        "ISIN": row["ISIN"],
        "Issuer": row["Issuer"],
        "Market_Price": market_price,
        "Implied_Spread_bps": spread_bps,
        "Raw_Spread_decimal": implied_spread
    })

df_spreads = pd.DataFrame(spread_results)
