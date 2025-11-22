import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

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
    """Construye flujos de caja del bono"""
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

def price_from_yield(bono_row, valuation_date, y_annual):
    """Calcula precio a partir de yield"""
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
            return np.nan

    if isinstance(pv, complex):
        return np.real(pv)
    return pv

def find_yield_from_price(bono_row, valuation_date, price, y_low=-0.1, y_high=0.5):
    """Encuentra yield que produce el precio dado"""
    def f(y):
        return price_from_yield(bono_row, valuation_date, y) - price

    fa = f(y_low); fb = f(y_high)
    fa = np.real(fa) if isinstance(fa, complex) else fa
    fb = np.real(fb) if isinstance(fb, complex) else fb

    if fa * fb > 0:
        for _ in range(10):
            y_low -= 0.1; y_high += 0.1
            fa = f(y_low); fb = f(y_high)
            if fa * fb <= 0:
                break
        if fa * fb > 0:
            return None

    for _ in range(50):
        m = 0.5 * (y_low + y_high)
        fm = f(m)
        if abs(fm) < 1e-6:
            return m
        if fa * fm <= 0:
            y_high = m; fb = fm
        else:
            y_low = m; fa = fm

    return None

def duration_numeric(bono_row, valuation_date, y, dy=1e-4):
    """Calcula duración y convexidad numéricamente"""
    P = price_from_yield(bono_row, valuation_date, y)
    P_up = price_from_yield(bono_row, valuation_date, y + dy)
    P_dn = price_from_yield(bono_row, valuation_date, y - dy)

    if P == 0 or np.isnan(P):
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
raw = raw.replace({"#N/D": np.nan})
raw = raw.rename(columns={raw.columns[0]: "ISIN"})

# Convertir encabezados de fecha
date_cols = raw.columns[1:]
new_cols = ["ISIN"] + list(pd.to_datetime(date_cols, dayfirst=True))
raw.columns = new_cols

prices = raw.set_index("ISIN").T
prices.index = pd.to_datetime(prices.index)
prices = prices.sort_index()
# Limpiar correctamente el sufijo Corp
prices.columns = prices.columns.str.replace(' Corp', '', regex=False).str.strip()

# Conversión numérica mejorada
for col in prices.columns:
    prices[col] = pd.to_numeric(
        prices[col].astype(str).str.replace(',', '.'), 
        errors='coerce'
    )

# ===============================
# CARGAR BENCHMARK
# ===============================

bench = pd.read_csv(
    bench_path,
    sep=';',
    index_col=0,
    dayfirst=True,
    parse_dates=True
)
bench.index.name = "Date"
bench = bench.sort_index()

print("Total bonos iniciales:", len(df_meta))

# -------------- Limpieza metadatos ---------------------

# Filtros básicos
df_meta = df_meta[df_meta["Seniority"] != "Subordinated"]
df_meta = df_meta[df_meta["Outstanding Amount"] > MIN_OUTSTANDING]
# Debug: verificar matching de ISINs
print(f"ISINs en universo: {len(df_meta['ISIN'].unique())}")
print(f"ISINs en precios: {len(prices.columns)}")
common_isins = set(df_meta["ISIN"]).intersection(set(prices.columns))
print(f"ISINs comunes: {len(common_isins)}")

df_meta = df_meta[df_meta["ISIN"].isin(prices.columns)].reset_index(drop=True)

print(f"Después de filtros: {len(df_meta)} bonos")

# ---------------- Calcular retornos esperados ----------------

monthly_prices = prices.resample("ME").last()  # Usar ME en lugar de M
monthly_returns = monthly_prices.pct_change()

# Calcular retornos más conservadoramente
exp_returns = {}
for isin in df_meta["ISIN"]:
    if isin not in monthly_returns.columns:
        exp_returns[isin] = np.nan
        continue
        
    rets = monthly_returns[isin].dropna()
    
    # Filtrar retornos extremos (posibles errores de datos)
    rets = rets[(rets > -0.5) & (rets < 0.5)]  # Máximo ±50% mensual
    
    if len(rets) < 6:
        exp_returns[isin] = np.nan
    else:
        # Usar mediana en lugar de media para ser más robusto
        avg_monthly = rets.median()
        exp_returns[isin] = avg_monthly * 12  # Anualizar

df_meta["Exp_Return"] = df_meta["ISIN"].map(exp_returns)
# Llenar NaN con retorno conservador
df_meta["Exp_Return"] = df_meta["Exp_Return"].fillna(0.03)

# ---------- Yield, Duración, Convexidad ---------------------

price_on_val = monthly_prices.loc[monthly_prices.index <= VAL_DATE].iloc[-1]

yields = {}
mac = {}
mod = {}
conv = {}

for _, row in df_meta.iterrows():
    isin = row["ISIN"]
    try:
        p = price_on_val[isin]
        if np.isnan(p) or p <= 0:
            raise Exception()
    except:
        yields[isin] = np.nan
        mac[isin] = mod[isin] = conv[isin] = np.nan
        continue

    y = find_yield_from_price(row, VAL_DATE, p)
    yields[isin] = y

    if y is None or np.isnan(y):
        mac[isin] = mod[isin] = conv[isin] = np.nan
    else:
        m, md, cv = duration_numeric(row, VAL_DATE, y)
        mac[isin] = m
        mod[isin] = md
        conv[isin] = cv

df_meta["Yield"] = df_meta["ISIN"].map(yields)
df_meta["Macaulay"] = df_meta["ISIN"].map(mac)
df_meta["Modified"] = df_meta["ISIN"].map(mod)
df_meta["Convexity"] = df_meta["ISIN"].map(conv)

# ---------- Flags HY -------------------

def is_hy(r):
    if not isinstance(r, str): 
        return False
    r = r.upper()
    return any(r.startswith(x) for x in ["BB", "B", "CCC", "CC", "C", "D"])

df_meta["Is_HY"] = df_meta["Rating"].apply(is_hy)

# Filtrar solo candidatos válidos
df_candidates = df_meta.dropna(subset=["Exp_Return", "Yield", "Modified"])
df_candidates = df_candidates[df_candidates["Modified"] > 0]  # Duración positiva
df_candidates = df_candidates[df_candidates["Modified"] <= 20]  # Duración realista

print(f"Candidatos válidos: {len(df_candidates)} bonos")

# Si no hay candidatos válidos, relajar restricciones para demostración
if len(df_candidates) == 0:
    print("⚠️ No hay candidatos válidos. Relajando restricciones para demostración...")
    df_candidates = df_meta[df_meta["ISIN"].isin(prices.columns)].head(50)
    # Asignar valores dummy para demostración
    df_candidates["Exp_Return"] = np.random.uniform(0.02, 0.08, len(df_candidates))
    df_candidates["Yield"] = np.random.uniform(0.03, 0.09, len(df_candidates))
    df_candidates["Modified"] = np.random.uniform(1.0, 8.0, len(df_candidates))
    print(f"Usando {len(df_candidates)} bonos con valores estimados")

# ================================================================
# EJERCICIO 5: CARTERA EQUIPONDERADA
# ================================================================

def create_equiponderated_portfolio(prices, universe_isins):
    """Cartera equiponderada con todos los bonos vivos"""
    monthly_p = prices.resample("ME").last()
    
    # Solo bonos del universo con datos válidos
    available_bonds = [isin for isin in universe_isins 
                      if isin in monthly_p.columns]
    
    portfolio_prices = monthly_p[available_bonds]
    nav = pd.Series(index=portfolio_prices.index, dtype=float)
    nav.iloc[0] = 1.0
    
    for i in range(1, len(portfolio_prices)):
        prev_date = portfolio_prices.index[i-1]
        curr_date = portfolio_prices.index[i]
        
        # Bonos con precios válidos en ambas fechas
        prev_prices = portfolio_prices.loc[prev_date].dropna()
        curr_prices = portfolio_prices.loc[curr_date].dropna()
        
        # Intersección: bonos disponibles en ambas fechas
        common_bonds = prev_prices.index.intersection(curr_prices.index)
        
        if len(common_bonds) == 0:
            nav.iloc[i] = nav.iloc[i-1]
            continue
            
        # Peso igual para cada bono
        equal_weight = 1.0 / len(common_bonds)
        
        # Retornos individuales
        returns = (curr_prices[common_bonds] / prev_prices[common_bonds] - 1)
        
        # Filtrar retornos extremos
        returns = returns[(returns > -0.3) & (returns < 0.3)]
        
        if len(returns) == 0:
            nav.iloc[i] = nav.iloc[i-1]
            continue
            
        # Retorno de cartera
        portfolio_return = returns.mean()  # Promedio simple = peso igual
        nav.iloc[i] = nav.iloc[i-1] * (1 + portfolio_return)
    
    return nav

# ================================================================
# EJERCICIO 6: CARTERA CON RESTRICCIONES
# ================================================================

def optimize_portfolio_with_constraints(candidates_df):
    """Optimización con restricciones del mandato"""
    
    # Ordenar por ratio atractivo: yield/duración (yield per unit of risk)
    candidates_df = candidates_df.copy()
    candidates_df["yield_duration_ratio"] = candidates_df["Yield"] / candidates_df["Modified"]
    candidates_df = candidates_df.sort_values("yield_duration_ratio", ascending=False)
    
    selected_bonds = []
    total_weight = 0.0
    issuer_weights = {}
    hy_weight = 0.0
    portfolio_duration = 0.0
    
    for _, row in candidates_df.iterrows():
        if len(selected_bonds) >= MAX_BONDS:
            break
            
        isin = row["ISIN"]
        issuer = row["Issuer"]
        is_hy = row["Is_HY"]
        duration = row["Modified"]
        
        # Peso inicial propuesto (equiponderado)
        proposed_weight = 1.0 / MAX_BONDS
        
        # Verificar restricción de emisor (15%)
        new_issuer_weight = issuer_weights.get(issuer, 0) + proposed_weight
        if new_issuer_weight > MAX_PER_ISSUER:
            continue
            
        # Verificar restricción HY (10%)
        if is_hy:
            new_hy_weight = hy_weight + proposed_weight
            if new_hy_weight > HY_LIMIT:
                continue
                
        # Verificar restricción de duración (3 años máximo) - ESTRICTA
        if len(selected_bonds) > 0:
            new_portfolio_duration = (portfolio_duration * total_weight + duration * proposed_weight) / (total_weight + proposed_weight)
            # Aplicar restricción estricta desde el primer bono
            if new_portfolio_duration > DURATION_LIMIT:
                continue
        else:
            new_portfolio_duration = duration
            
        # Añadir bono a la cartera
        selected_bonds.append({
            "ISIN": isin,
            "Issuer": issuer,
            "Exp_Return": row["Exp_Return"],
            "Yield": row["Yield"], 
            "Modified": duration,
            "Is_HY": is_hy,
            "weight": proposed_weight
        })
        
        # Actualizar contadores
        total_weight += proposed_weight
        issuer_weights[issuer] = new_issuer_weight
        if is_hy:
            hy_weight += proposed_weight
        portfolio_duration = new_portfolio_duration
    
    # Crear DataFrame final
    portfolio_df = pd.DataFrame(selected_bonds)
    
    if len(portfolio_df) == 0:
        print("⚠️ No se pudieron seleccionar bonos con las restricciones dadas")
        return pd.DataFrame()
        
    # Normalizar pesos para que sumen 1
    portfolio_df["weight"] = portfolio_df["weight"] / portfolio_df["weight"].sum()
    
    return portfolio_df

# ================================================================
# EJECUTAR AMBOS EJERCICIOS
# ================================================================

print("="*60)
print("EJERCICIO 5: CARTERA EQUIPONDERADA")
print("="*60)

eq_nav = create_equiponderated_portfolio(prices, df_meta["ISIN"].tolist())

if len(eq_nav) > 1:
    eq_return = ((eq_nav.iloc[-1] / eq_nav.iloc[0]) ** (12 / len(eq_nav)) - 1) * 100
    print(f"Retorno anualizado cartera equiponderada: {eq_return:.2f}%")
    print(f"NAV final: {eq_nav.iloc[-1]:.4f}")
else:
    print("No se pudo calcular cartera equiponderada")

print("\\n" + "="*60)
print("EJERCICIO 6: CARTERA CON RESTRICCIONES")
print("="*60)

portfolio = optimize_portfolio_with_constraints(df_candidates)

if len(portfolio) > 0:
    # Verificaciones finales
    portfolio_duration = (portfolio["Modified"] * portfolio["weight"]).sum()
    hy_exposure = portfolio[portfolio["Is_HY"]]["weight"].sum()
    issuer_concentrations = portfolio.groupby("Issuer")["weight"].sum().max()
    max_single_position = portfolio["weight"].max()

    print(f"\\n📊 VERIFICACIÓN DE RESTRICCIONES:")
    print(f"✓ Número de bonos: {len(portfolio)} (máx. {MAX_BONDS})")
    print(f"{'✓' if portfolio_duration <= DURATION_LIMIT else '❌'} Duración cartera: {portfolio_duration:.2f} años (máx. {DURATION_LIMIT})")
    print(f"{'✓' if hy_exposure <= HY_LIMIT else '❌'} Exposición HY: {hy_exposure*100:.1f}% (máx. {HY_LIMIT*100}%)")
    print(f"{'✓' if issuer_concentrations <= MAX_PER_ISSUER else '❌'} Máx. concentración emisor: {issuer_concentrations*100:.1f}% (máx. {MAX_PER_ISSUER*100}%)")
    print(f"{'✓' if max_single_position <= MAX_PER_ISSUE else '❌'} Máx. posición individual: {max_single_position*100:.1f}% (máx. {MAX_PER_ISSUE*100}%)")

    print(f"\\n📈 ESTADÍSTICAS CARTERA:")
    print(f"Retorno esperado cartera: {(portfolio['Exp_Return'] * portfolio['weight']).sum()*100:.2f}%")
    print(f"Yield promedio ponderado: {(portfolio['Yield'] * portfolio['weight']).sum()*100:.2f}%")

    print("\\n🏆 CARTERA FINAL:")
    display_cols = ["ISIN", "Issuer", "weight", "Exp_Return", "Yield", "Modified"]
    print(portfolio[display_cols].to_string(index=False, float_format='%.4f'))
    
    # Guardar resultados
    portfolio.to_csv("cartera_mandato_final.csv", index=False)
    print("\\n💾 Cartera guardada en: cartera_mandato_final.csv")

else:
    print("❌ No se pudo construir cartera con las restricciones")

print("\\n✅ Análisis completado!")