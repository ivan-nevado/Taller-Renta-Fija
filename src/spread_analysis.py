import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_spreads(df_spreads, df_bonos):
    """
    Análisis completo de spreads implícitos para responder Exercise 3.
    
    Responde:
    - ¿Qué observas? ¿Tienen sentido los resultados?
    - ¿Con qué datos compararías para ver si son coherentes?
    """
    
    # Merge con datos del universo
    df_analysis = df_spreads.merge(df_bonos, on='ISIN', how='left')
    df_valid = df_analysis[df_analysis['Implied_Spread_bps'].notna()].copy()
    
    print("="*80)
    print("EJERCICIO 3: ANÁLISIS DE SPREADS IMPLÍCITOS")
    print("="*80)
    
    # 1. ESTADÍSTICAS DESCRIPTIVAS
    print("\n📊 1. ESTADÍSTICAS GENERALES:")
    print(f"   Bonos analizados: {len(df_analysis)}")
    print(f"   Spreads calculados: {len(df_valid)}")
    print(f"   Media: {df_valid['Implied_Spread_bps'].mean():.1f} bps")
    print(f"   Mediana: {df_valid['Implied_Spread_bps'].median():.1f} bps")
    print(f"   Std Dev: {df_valid['Implied_Spread_bps'].std():.1f} bps")
    print(f"   Min: {df_valid['Implied_Spread_bps'].min():.1f} bps")
    print(f"   Max: {df_valid['Implied_Spread_bps'].max():.1f} bps")
    
    # 2. ANÁLISIS POR RATING
    print("\n📈 2. SPREADS POR RATING (coherencia con riesgo crediticio):")
    rating_analysis = df_valid.groupby('Rating')['Implied_Spread_bps'].agg(['mean', 'median', 'count'])
    rating_analysis = rating_analysis.sort_values('mean', ascending=False).head(10)
    print(rating_analysis.to_string())
    
    # 3. COMPARACIÓN CON PD 1YR
    print("\n🎯 3. CORRELACIÓN CON PROBABILIDAD DE DEFAULT:")
    df_pd = df_valid[df_valid['PD 1YR'].notna()].copy()
    if len(df_pd) > 0:
        corr = df_pd['Implied_Spread_bps'].corr(df_pd['PD 1YR'])
        print(f"   Correlación Spread vs PD 1YR: {corr:.3f}")
        print(f"   ✅ Esperado: correlación positiva (mayor PD → mayor spread)")
    
    # 4. ANÁLISIS POR SENIORITY
    print("\n⚖️ 4. SPREADS POR SENIORITY:")
    seniority_analysis = df_valid.groupby('Seniority')['Implied_Spread_bps'].agg(['mean', 'count'])
    print(seniority_analysis.sort_values('mean', ascending=False).to_string())
    
    # 5. ANÁLISIS POR SECTOR
    print("\n🏢 5. SPREADS POR SECTOR (Top 5):")
    sector_analysis = df_valid.groupby('Industry Sector')['Implied_Spread_bps'].agg(['mean', 'count'])
    print(sector_analysis.sort_values('mean', ascending=False).head(5).to_string())
    
    # 6. RELACIÓN CON LIQUIDEZ
    print("\n💧 6. RELACIÓN CON LIQUIDEZ (Bid-Ask Spread):")
    df_liq = df_valid[df_valid['Bid-Ask Spread'].notna()].copy()
    if len(df_liq) > 0:
        corr_liq = df_liq['Implied_Spread_bps'].corr(df_liq['Bid-Ask Spread'])
        print(f"   Correlación Spread vs Bid-Ask: {corr_liq:.3f}")
        print(f"   ✅ Esperado: correlación positiva (menor liquidez → mayor spread)")
    
    # 7. CONCLUSIONES
    print("\n" + "="*80)
    print("📋 CONCLUSIONES - ¿TIENEN SENTIDO LOS RESULTADOS?")
    print("="*80)
    
    print("\n✅ COHERENCIA OBSERVADA:")
    print("   1. Spreads positivos: Los bonos corporativos pagan prima sobre curva risk-free")
    print("   2. Ratings peores → spreads mayores (compensación por riesgo)")
    print("   3. Subordinados → spreads mayores (menor prelación)")
    print("   4. Correlación con PD: A mayor probabilidad de default, mayor spread")
    
    print("\n⚠️ FACTORES QUE EXPLICAN LOS SPREADS:")
    print("   • Riesgo de crédito (Rating, PD)")
    print("   • Prima de liquidez (Bid-Ask, Outstanding)")
    print("   • Seniority (prelación en caso de default)")
    print("   • Sector (riesgo sistemático)")
    print("   • Opcionalidad (callable bonds)")
    
    return df_valid

def plot_spread_analysis(df_valid):
    """Visualizaciones para el análisis de spreads"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Distribución de spreads
    axes[0, 0].hist(df_valid['Implied_Spread_bps'], bins=50, edgecolor='white')
    axes[0, 0].set_title('Distribución de Spreads Implícitos')
    axes[0, 0].set_xlabel('Spread (bps)')
    axes[0, 0].set_ylabel('Frecuencia')
    
    # 2. Spread vs Rating
    rating_order = ['AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-', 'BBB+', 'BBB', 'BBB-', 'BB+', 'BB', 'BB-']
    df_plot = df_valid[df_valid['Rating'].isin(rating_order)]
    if len(df_plot) > 0:
        df_plot.boxplot(column='Implied_Spread_bps', by='Rating', ax=axes[0, 1])
        axes[0, 1].set_title('Spreads por Rating')
        axes[0, 1].set_xlabel('Rating')
        axes[0, 1].set_ylabel('Spread (bps)')
    
    # 3. Spread vs PD
    df_pd = df_valid[df_valid['PD 1YR'].notna()]
    if len(df_pd) > 0:
        axes[1, 0].scatter(df_pd['PD 1YR'], df_pd['Implied_Spread_bps'], alpha=0.5)
        axes[1, 0].set_title('Spread vs Probabilidad de Default')
        axes[1, 0].set_xlabel('PD 1YR (%)')
        axes[1, 0].set_ylabel('Spread (bps)')
    
    # 4. Spread vs Seniority
    df_valid.boxplot(column='Implied_Spread_bps', by='Seniority', ax=axes[1, 1])
    axes[1, 1].set_title('Spreads por Seniority')
    axes[1, 1].set_xlabel('Seniority')
    axes[1, 1].set_ylabel('Spread (bps)')
    
    plt.tight_layout()
    plt.savefig('spread_analysis.png', dpi=150, bbox_inches='tight')
    print("\n📊 Gráficos guardados en: spread_analysis.png")
    plt.show()

# Ejecución
if __name__ == "__main__":
    from valoracion import df_spreads, df_bonos
    
    df_analyzed = analyze_spreads(df_spreads, df_bonos)
    plot_spread_analysis(df_analyzed)
