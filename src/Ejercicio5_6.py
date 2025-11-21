import pandas as pd
import numpy as np
from datetime import datetime

# ---------- Parámetros ---------------
MAX_BONDS = 20
DURATION_LIMIT = 3.0
HY_LIMIT = 0.10
MAX_PER_ISSUE = 0.10
MAX_PER_ISSUER = 0.15
MIN_OUTSTANDING = 500000000.0
REBALANCE_FREQ = "M"
VAL_DATE = pd.Timestamp(2025, 10, 1)

# ---------- Funciones financieras -----------

def build_cashflows(bono_row, valuation_date=VAL_DATE):
    if bono_row["Callable"] == "Y" and pd.notna(bono_row.get("Next Call Date")):
        maturity = bono_row["Next Call Date"]
    else:
        maturity = bono_row["Maturity"]

    if maturity <= valuation_date:
        return []

    freq = int(bono_row["Coupon Frequency"])
    coupon_rate = float(bono_row["Coupon"]) / 100.0
    nominal = 100.0

    dates = [maturity]
    step = int(12 / freq)
    d = maturity
    while d > valuation_date:
        d = d - pd.DateOffset(months=step)
        dates.append(d)

    dates = sorted(dates)

    flows = []
    for d in dates:
        if d <= valuation_date:
            continue
        if d == maturity:
            cf = nominal + nominal * coupon_rate / freq
        else:
            cf = nominal * coupon_rate / freq
        flows.append((d, cf))
    return flows

"""
def price_from_yield(bono_row, valuation_date, y_annual):
    flows = build_cashflows(bono_row, valuation_date)
    if len(flows) == 0:
        return 0.0

    freq = int(bono_row["Coupon Frequency"])
    y_period = y_annual / freq
    pv = 0.0

    for (d, cf) in flows:
        t = (d - valuation_date).days / 365.0
        n = t * freq
        pv += cf / ((1 + y_period) ** n)

    return pv
 """


def price_from_yield(bono_row, valuation_date, y_annual):
    flows = build_cashflows(bono_row, valuation_date)
    if len(flows) == 0:
        return 0.0

    freq = int(bono_row["Coupon Frequency"])
    y_period = y_annual / freq
    pv = 0.0

    for (d, cf) in flows:
        t = (d - valuation_date).days / 365.0
        n = t * freq
        try:
            pv += cf / ((1 + y_period) ** n)
        except:
            return np.nan  # si da error, devolver nan

    # asegurar que sea real
    if isinstance(pv, complex):
        return np.real(pv)

    return pv




def find_yield_from_price(bono_row, valuation_date, price, y_low=-0.5, y_high=5.0):
    def f(y):
        return price_from_yield(bono_row, valuation_date, y) - price

    fa = f(y_low); fb = f(y_high)
    # si son complejos, usar solo la parte real
    fa = np.real(fa) if isinstance(fa, complex) else fa
    fb = np.real(fb) if isinstance(fb, complex) else fb


    if fa * fb > 0:
        for _ in range(30):
            y_low *= 1.5
            y_high *= 1.5
            fa = f(y_low); fb = f(y_high)
            if fa * fb <= 0:
                break
        if fa * fb > 0:
            return None

    for _ in range(100):
        m = 0.5 * (y_low + y_high)
        fm = f(m)
        if abs(fm) < 1e-8:
            return m
        if fa * fm <= 0:
            y_high = m; fb = fm
        else:
            y_low = m; fa = fm

    return None

def duration_numeric(bono_row, valuation_date, y, dy=1e-4):
    P = price_from_yield(bono_row, valuation_date, y)
    P_up = price_from_yield(bono_row, valuation_date, y + dy)
    P_dn = price_from_yield(bono_row, valuation_date, y - dy)

    if P == 0:
        return None, None, None

    mod_dur = -(P_up - P_dn) / (2 * P * dy)
    convex = (P_up + P_dn - 2 * P) / (P * dy * dy)

    freq = int(bono_row["Coupon Frequency"])
    macaulay = mod_dur * (1 + y/freq)

    return macaulay, mod_dur, convex


# ------------------ Cargar datos -----------------------

universo_path = r"C:\Users\Joseph\Documents\MIAX\Modulo II\Taller-Renta-Fija\data\universo.csv"
precios_path  = r"C:\Users\Joseph\Documents\MIAX\Modulo II\Taller-Renta-Fija\data\precios_historicos_universo.csv"
bench_path    = r"C:\Users\Joseph\Documents\MIAX\Modulo II\Taller-Renta-Fija\data\precios_historicos_varios.csv"

# ------------------------------------------------------------------
# CARGAMOS UNIVERSO
# ------------------------------------------------------------------

df_meta = pd.read_csv(
    universo_path,
    sep=';',
    parse_dates=["Maturity", "Next Call Date"],
    dayfirst=True
)
df_meta["ISIN"] = df_meta["ISIN"].str.strip()


# ------------------------------------------------------------------
# CARGAMOS PRECIOS HISTORICOS UNIVERSO
# ------------------------------------------------------------------

raw = pd.read_csv(precios_path, sep=';', low_memory=False)

# Reemplazar strings N/D
raw = raw.replace({"#N/D": np.nan})

# Primera columna = ISIN
raw = raw.rename(columns={raw.columns[0]: "ISIN"})

# Convertir encabezados de fecha a datetime
date_cols = raw.columns[1:]
new_cols = ["ISIN"] + list(pd.to_datetime(date_cols, dayfirst=True))
raw.columns = new_cols

# Establecer ISIN como índice
prices = raw.set_index("ISIN")

# Transponer: queremos fechas como índice
prices = prices.T
prices.index = pd.to_datetime(prices.index)
prices = prices.sort_index()

# Limpiar ISINs (quitar ' Corp' del nombre de columna)
prices.columns = prices.columns.str.replace(r'\s+Corp$', '', regex=True)

# ------------------------------------------------------------------
# 🔥🔥🔥 CONVERSIÓN A NUMÉRICO (EL FIX QUE TE ELIMINA EL ERROR)
# ------------------------------------------------------------------
prices = prices.apply(
    lambda col: (
        col.astype(str)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
)

# Convertir finalmente a FLOAT
prices = prices.apply(pd.to_numeric, errors="coerce")


# ===============================
# CARGAR BENCHMARK (RECMTREU)
# ===============================

bench = pd.read_csv(
    bench_path,
    sep=';',
    index_col=0,            # Usamos la primera columna como fecha
    dayfirst=True,
    parse_dates=True
)

bench.index.name = "Date"
bench = bench.sort_index()

print("Total bonos iniciales:", len(df_meta))

# -------------- Limpieza metadatos ---------------------

df_meta = df_meta[df_meta["Seniority"].str.lower() != "Subordinated"]
df_meta = df_meta[df_meta["Outstanding Amount"] > MIN_OUTSTANDING]

print("Después de limpiar subordinadas y tamaño:", len(df_meta))

# Mantener sólo ISINs presentes en precios
df_meta = df_meta[df_meta["ISIN"].isin(prices.columns)].reset_index(drop=True)
print("AAAAAAA", len(df_meta))

# ---------------- Calcular retornos esperados ----------------

monthly_prices = prices.resample("M").last()
rets = monthly_prices.pct_change()

exp_returns = {}
for isin in df_meta["ISIN"]:
    r = rets[isin].dropna()
    if len(r) < 6:
        exp_returns[isin] = np.nan
    else:
        exp_returns[isin] = (np.prod(1+r))**(12/len(r)) - 1

df_meta["Exp_Return"] = df_meta["ISIN"].map(exp_returns).fillna(0.01)

# ---------- Yield, Duración, Convexidad ---------------------

price_on_val = monthly_prices.loc[:VAL_DATE].iloc[-1]

yields = {}
mac = {}
mod = {}
conv = {}

for _, row in df_meta.iterrows():
    isin = row["ISIN"]
    try:
        p = price_on_val[isin]
        if np.isnan(p):
            raise Exception()
    except:
        yields[isin] = np.nan; mac[isin]=np.nan; mod[isin]=np.nan; conv[isin]=np.nan
        continue

    y = find_yield_from_price(row, VAL_DATE, p)
    yields[isin] = y

    if y is None:
        mac[isin]=mod[isin]=conv[isin]=np.nan
    else:
        m,md,cv = duration_numeric(row, VAL_DATE, y)
        mac[isin] = m
        mod[isin] = md
        conv[isin] = cv

df_meta["Yield"] = df_meta["ISIN"].map(yields)
df_meta["Macaulay"] = df_meta["ISIN"].map(mac)
df_meta["Modified"] = df_meta["ISIN"].map(mod)
df_meta["Convexity"] = df_meta["ISIN"].map(conv)

# ---------- Flags HY -------------------

def is_hy(r):
    if not isinstance(r,str): return False
    r = r.upper()
    return r.startswith("BB") or r.startswith("B") or r.startswith("CCC") or r.startswith("CC") or r.startswith("C") or r.startswith("D")

df_meta["Is_HY"] = df_meta["Rating"].apply(is_hy)

df_candidates = df_meta.dropna(subset=["Exp_Return", "Yield"])

# ---------- Selección de cartera --------------------

df_candidates = df_candidates.sort_values("Exp_Return", ascending=False)

selected = []

issuer_weight = {}
hy_weight = 0

for _, row in df_candidates.iterrows():
    if len(selected) >= MAX_BONDS:
        break

    isin = row["ISIN"]
    issuer = row["Issuer"]
    hy = row["Is_HY"]

    if issuer_weight.get(issuer,0) >= MAX_PER_ISSUER:
        continue
    if hy and hy_weight >= HY_LIMIT:
        continue

    selected.append(isin)
    issuer_weight[issuer] = issuer_weight.get(issuer,0) + 1e-6
    if hy:
        hy_weight += 1e-6


sel_df = df_meta[df_meta["ISIN"].isin(selected)].copy()

# Pesos
shift = max(0, -sel_df["Exp_Return"].min())
sel_df["base"] = sel_df["Exp_Return"] + shift
sel_df["weight"] = sel_df["base"] / sel_df["base"].sum()
sel_df["weight"] = sel_df["weight"].clip(upper=MAX_PER_ISSUE)
sel_df["weight"] /= sel_df["weight"].sum()

# ---------- Resultado final --------------------

portfolio = sel_df[["ISIN","Issuer","Exp_Return","Yield","Modified","Is_HY","weight"]]
portfolio = portfolio.sort_values("weight", ascending=False)
portfolio.to_csv("cartera_resultado.csv", index=False)

print("\nCartera final:")
print(portfolio)

# ---------- Backtest --------------------------

monthly_pf = monthly_prices[portfolio["ISIN"]]
nav = pd.Series(index=monthly_pf.index, dtype=float)
nav.iloc[0] = 1.0

weights = portfolio.set_index("ISIN")["weight"]

for i in range(1, len(monthly_pf)):
    prev = monthly_pf.index[i-1]
    cur = monthly_pf.index[i]

    ret = (monthly_pf.loc[cur] / monthly_pf.loc[prev]) - 1
    ret = ret.fillna(0)

    nav.iloc[i] = nav.iloc[i-1] * (1 + (weights * ret).sum())

bench_month = bench.resample("M").last()
bench_nav = bench_month["RECMTREU Index"]
bench_nav = bench_nav / bench_nav.iloc[0]

pd.DataFrame({"Portfolio": nav}).to_csv("evolucion_cartera.csv")
bench_nav.to_csv("bench_nav.csv")

print("\nSimulación guardada.")
print("Total bonos iniciales:", len(df_meta))
print("Después de limpiar subordinadas y tamaño:", len(df_meta))
print("Después de filtrar por precios disponibles:", len(df_meta))
print("Candidatos con Exp_Return y Yield:", len(df_candidates))

