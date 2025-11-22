# hedging_analysis.py
# EJERCICIO 7: COBERTURA DE TIPOS DE INTERÉS
# Módulo profesional para análisis de cobertura con futuros alemanes

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

class InterestRateHedging:
    """
    Clase para análisis y implementación de cobertura de tipos de interés
    usando futuros alemanes (Schatz, BOBL, BUND)
    """
    
    def __init__(self, portfolio_duration: float, portfolio_value: float):
        """
        Inicializar análisis de cobertura
        
        Args:
            portfolio_duration: Duración de la cartera en años
            portfolio_value: Valor de la cartera en EUR
        """
        self.portfolio_duration = portfolio_duration
        self.portfolio_value = portfolio_value
        self.duration_risk = portfolio_duration * portfolio_value
        
        # Especificaciones de futuros alemanes
        self.instruments = {
            'Schatz (DU1)': {
                'duration': 1.92,
                'maturity': '2 años',
                'contract_size': 100_000,
                'underlying': 'Bono alemán 2Y',
                'liquidity': 'Muy alta',
                'ticker': 'DU1 Comdty'
            },
            'BOBL (OE1)': {
                'duration': 5.44,
                'maturity': '5 años', 
                'contract_size': 100_000,
                'underlying': 'Bono alemán 5Y',
                'liquidity': 'Alta',
                'ticker': 'OE1 Comdty'
            },
            'BUND (RX1)': {
                'duration': 10.0,
                'maturity': '10 años',
                'contract_size': 100_000,
                'underlying': 'Bono alemán 10Y',
                'liquidity': 'Muy alta',
                'ticker': 'RX1 Comdty'
            }
        }
    
    def load_futures_data(self, file_path: str) -> pd.DataFrame:
        """
        Cargar datos históricos de futuros alemanes
        
        Args:
            file_path: Ruta al archivo CSV con datos históricos
            
        Returns:
            DataFrame con precios históricos limpiados
        """
        try:
            # Cargar datos
            data = pd.read_csv(file_path, sep=';', index_col=0)
            data.index = pd.to_datetime(data.index, dayfirst=True)
            data = data.sort_index()
            
            # Convertir a numérico
            for ticker in ['DU1 Comdty', 'OE1 Comdty', 'RX1 Comdty']:
                if ticker in data.columns:
                    data[ticker] = pd.to_numeric(data[ticker], errors='coerce')
            
            print(f"✅ Datos cargados: {data.shape[0]} observaciones")
            return data
            
        except Exception as e:
            print(f"❌ Error cargando datos: {e}")
            return pd.DataFrame()
    
    def calculate_hedge_ratios(self) -> Dict[str, float]:
        """
        Calcular hedge ratios óptimos para cada instrumento
        
        Returns:
            Diccionario con contratos necesarios por instrumento
        """
        hedge_ratios = {}
        
        print("🎯 CÁLCULO DE HEDGE RATIOS:")
        print("Fórmula: Hedge Ratio = (Portfolio Duration × Portfolio Value) / (Future Duration × Contract Size)")
        print()
        
        for name, specs in self.instruments.items():
            ratio = self.duration_risk / (specs['duration'] * specs['contract_size'])
            hedge_ratios[name] = ratio
            
            print(f"{name}:")
            print(f"  Contratos necesarios: {ratio:.1f} contratos")
            print(f"  Exposición: {ratio * specs['contract_size'] * specs['duration']:,.0f} EUR·años")
            print(f"  Eficiencia: {(ratio * specs['contract_size'] * specs['duration'] / self.duration_risk * 100):.1f}%")
            print()
            
        return hedge_ratios
    
    def analyze_instruments(self) -> Dict[str, Dict]:
        """
        Análisis comparativo de instrumentos de cobertura
        
        Returns:
            Análisis detallado de ventajas/desventajas
        """
        analysis = {
            'Schatz (DU1)': {
                'pros': [
                    f'Duración más cercana a cartera ({self.instruments["Schatz (DU1)"]["duration"]} vs {self.portfolio_duration})',
                    'Máxima liquidez',
                    'Menor basis risk',
                    'Mejor matching temporal'
                ],
                'cons': [
                    'Mayor número de contratos requeridos',
                    'Mayor coste operativo de transacciones',
                    'Más ajustes de rebalanceo'
                ],
                'score': 5
            },
            'BOBL (OE1)': {
                'pros': [
                    'Número moderado de contratos',
                    'Buena liquidez',
                    'Coste operativo equilibrado'
                ],
                'cons': [
                    f'Duración muy diferente ({self.instruments["BOBL (OE1)"]["duration"]} vs {self.portfolio_duration})',
                    'Mayor basis risk',
                    'Sobrecubierto significativo',
                    'Sensibilidad excesiva a largo plazo'
                ],
                'score': 3
            },
            'BUND (RX1)': {
                'pros': [
                    'Mínimo número de contratos',
                    'Máxima liquidez',
                    'Menor coste operativo'
                ],
                'cons': [
                    f'Duración muy diferente ({self.instruments["BUND (RX1)"]["duration"]} vs {self.portfolio_duration})',
                    'Máximo basis risk',
                    'Muy sobrecubierto',
                    'Riesgo de convexidad negativa'
                ],
                'score': 2
            }
        }
        
        print("📋 ANÁLISIS COMPARATIVO:")
        for name, data in analysis.items():
            stars = "⭐" * data['score']
            print(f"\n{name} {stars}:")
            print("  ✅ Ventajas:")
            for pro in data['pros']:
                print(f"    • {pro}")
            print("  ❌ Desventajas:")
            for con in data['cons']:
                print(f"    • {con}")
                
        return analysis
    
    def scenario_analysis(self, contracts: int = 100) -> Dict[str, Dict]:
        """
        Análisis de escenario con número fijo de contratos
        
        Args:
            contracts: Número de contratos a analizar
            
        Returns:
            Análisis de P&L por escenarios
        """
        scenarios = [-1.0, -0.5, 0, 0.5, 1.0]  # Cambios en yield (%)
        results = {}
        
        print(f"\n📊 ANÁLISIS DE ESCENARIO: {contracts} CONTRATOS")
        print("="*60)
        
        for name, specs in self.instruments.items():
            total_exposure = contracts * specs['contract_size'] * specs['duration']
            ratio_vs_portfolio = total_exposure / self.duration_risk
            
            # P&L por escenario
            pnl_scenarios = {}
            for scenario in scenarios:
                pnl = -total_exposure * scenario / 100
                pnl_scenarios[f"{scenario:+.1f}%"] = pnl
            
            results[name] = {
                'total_exposure': total_exposure,
                'ratio_vs_portfolio': ratio_vs_portfolio,
                'pnl_scenarios': pnl_scenarios
            }
            
            print(f"📈 {name}:")
            print(f"  Exposición total: {total_exposure:,.0f} EUR·años")
            print(f"  Ratio vs cartera: {ratio_vs_portfolio:.1f}x")
            print("  P&L por cambio de yield:")
            for scenario_label, pnl in pnl_scenarios.items():
                print(f"    {scenario_label}: {pnl:+,.0f} EUR")
            print()
        
        # P&L de cartera sin cobertura
        print("📊 P&L CARTERA (sin cobertura):")
        print("  P&L por cambio de yield:")
        for scenario in scenarios:
            pnl_portfolio = -self.duration_risk * scenario / 100
            print(f"    {scenario:+.1f}%: {pnl_portfolio:+,.0f} EUR")
            
        return results
    
    def get_recommendation(self) -> Dict[str, any]:
        """
        Obtener recomendación final de cobertura
        
        Returns:
            Recomendación completa con justificación
        """
        hedge_ratios = self.calculate_hedge_ratios()
        recommended_instrument = 'Schatz (DU1)'
        contracts_needed = hedge_ratios[recommended_instrument]
        contracts_rounded = round(contracts_needed)
        
        specs = self.instruments[recommended_instrument]
        final_exposure = contracts_rounded * specs['contract_size'] * specs['duration']
        efficiency = (final_exposure / self.duration_risk) * 100
        
        recommendation = {
            'instrument': recommended_instrument,
            'contracts': contracts_rounded,
            'direction': 'VENDER',  # Para neutralizar riesgo de subida de tipos
            'exposure': final_exposure,
            'efficiency': efficiency,
            'reasoning': [
                f"Mejor matching de duración ({specs['duration']} vs {self.portfolio_duration} años)",
                "Minimiza basis risk entre corporativos EUR y soberanos alemanes",
                "Máxima liquidez para ajustes dinámicos",
                "Precisión de cobertura compensa mayor número de contratos"
            ],
            'implementation': {
                'operation': f"VENDER {contracts_rounded} contratos {recommended_instrument}",
                'objective': "Neutralizar sensibilidad a tipos de interés",
                'monitoring': "Rebalancear mensualmente según duración de cartera",
                'stop_loss': "Si basis risk > 50bps durante 5 días consecutivos"
            },
            'expected_results': {
                'hedge_coverage': f"{efficiency:.0f}%",
                'transaction_cost': "5-10 bps (spread + comisiones)",
                'benefit': "Cartera neutralizada ante movimientos de tipos",
                'residual_risk': "Basis risk corporativo vs soberano ≈ 10-20 bps"
            }
        }
        
        return recommendation
    
    def get_alternative_instruments(self) -> Dict[str, Dict]:
        """
        Instrumentos alternativos para cobertura de tipos
        
        Returns:
            Análisis de instrumentos alternativos
        """
        alternatives = {
            "Interest Rate Swaps (IRS)": {
                "description": "Swap fixed vs floating EUR",
                "pros": ["Duración exacta personalizable", "Sin costes iniciales", "Máxima flexibilidad"],
                "cons": ["Requiere línea de crédito", "Riesgo de contraparte", "Menos líquido para montos pequeños"],
                "best_for": "Carteras >50M EUR",
                "complexity": "Alto"
            },
            "ETFs de Bonos Alemanes": {
                "description": "iShares Core € Govt Bond, Xtrackers Bund ETF",
                "pros": ["Fácil ejecución", "Sin derivados", "Transparente", "Acceso retail"],
                "cons": ["Comisiones de gestión", "Tracking error", "Menos eficiente en capital"],
                "best_for": "Inversores retail",
                "complexity": "Bajo"
            },
            "Futuros de otros países": {
                "description": "OAT (Francia), BTP (Italia), Bonos España",
                "pros": ["Diversificación geográfica", "Menor basis con corporate EUR"],
                "cons": ["Basis risk país", "Menor liquidez", "Riesgo spread soberanos"],
                "best_for": "Estrategias multi-país",
                "complexity": "Medio"
            },
            "Options on Futures": {
                "description": "Opciones sobre BUND/BOBL/SCHATZ",
                "pros": ["Asimetría de riesgo", "Prima definida", "Flexibilidad estratégica"],
                "cons": ["Coste de prima", "Decay temporal", "Más complejo"],
                "best_for": "Cobertura direccional con límite de pérdida",
                "complexity": "Alto"
            }
        }
        
        return alternatives
    
    def print_summary_report(self):
        """
        Imprimir reporte resumen completo del análisis
        """
        print("🎯 EJERCICIO 7: COBERTURA DE TIPOS DE INTERÉS")
        print("="*60)
        
        print("\n📊 DATOS DE LA CARTERA A CUBRIR:")
        print(f"Valor de la cartera: {self.portfolio_value:,} EUR")
        print(f"Duración de la cartera: {self.portfolio_duration:.2f} años")
        print(f"Riesgo de duración: {self.duration_risk:,.0f} EUR·años")
        
        # Cálculo de hedge ratios
        hedge_ratios = self.calculate_hedge_ratios()
        
        # Análisis comparativo
        self.analyze_instruments()
        
        # Escenario 100 contratos
        self.scenario_analysis(100)
        
        # Recomendación final
        recommendation = self.get_recommendation()
        
        print("\n🏆 RECOMENDACIÓN FINAL:")
        print("="*40)
        print(f"Instrumento: {recommendation['instrument']}")
        print(f"Operación: {recommendation['direction']} {recommendation['contracts']} contratos")
        print(f"Cobertura: {recommendation['efficiency']:.0f}%")
        print(f"Exposición final: {recommendation['exposure']:,.0f} EUR·años")
        
        print("\n📋 JUSTIFICACIÓN:")
        for reason in recommendation['reasoning']:
            print(f"• {reason}")
            
        print(f"\n📈 IMPLEMENTACIÓN:")
        for key, value in recommendation['implementation'].items():
            print(f"• {key.replace('_', ' ').title()}: {value}")
            
        print("\n✅ ANÁLISIS COMPLETADO")


def run_hedging_analysis():
    """
    Función principal para ejecutar el análisis completo de cobertura
    """
    # Datos de la cartera del Ejercicio 6 (confirmados)
    PORTFOLIO_DURATION = 0.95  # años
    PORTFOLIO_VALUE = 10_000_000  # EUR
    
    # Crear instancia del analizador
    hedging = InterestRateHedging(PORTFOLIO_DURATION, PORTFOLIO_VALUE)
    
    # Cargar datos históricos
    futures_data = hedging.load_futures_data('../data/precios_historicos_varios.csv')
    
    if not futures_data.empty:
        print(f"📈 Datos históricos cargados:")
        for ticker in ['DU1 Comdty', 'OE1 Comdty', 'RX1 Comdty']:
            if ticker in futures_data.columns:
                series = futures_data[ticker].dropna()
                if len(series) > 0:
                    print(f"  {ticker}: {len(series)} obs., rango {series.min():.2f} - {series.max():.2f}")
    
    # Ejecutar análisis completo
    hedging.print_summary_report()
    
    # Mostrar instrumentos alternativos
    alternatives = hedging.get_alternative_instruments()
    print("\n🛠️ INSTRUMENTOS ALTERNATIVOS:")
    print("="*50)
    for name, details in alternatives.items():
        print(f"\n📋 {name} (Complejidad: {details['complexity']}):")
        print(f"   {details['description']}")
        print(f"   ✅ Ventajas: {', '.join(details['pros'])}")
        print(f"   ❌ Desventajas: {', '.join(details['cons'])}")
        print(f"   🎯 Ideal para: {details['best_for']}")
    
    return hedging


if __name__ == "__main__":
    # Ejecutar análisis si se ejecuta directamente
    analyzer = run_hedging_analysis()