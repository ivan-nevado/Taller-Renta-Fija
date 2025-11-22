"""
custom_strategy.py

Estrategia propia de renta fija: Valor relativo IG/HY y estructura barbell de duración.

Autor: Ivan Nevado
Fecha: 2025-11-22
"""
import pandas as pd
import numpy as np

class CustomFixedIncomeStrategy:
    """
    Estrategia profesional de renta fija combinando valor relativo IG/HY y barbell de duración.
    """
    def __init__(self, universe_path, portfolio_constraints):
        self.universe_path = universe_path
        self.universe = None
        self.portfolio_constraints = portfolio_constraints
        self.portfolio = None
        self.results = None

    def load_universe(self):
        """Carga el universo de bonos."""
        self.universe = pd.read_csv(self.universe_path)

    def construct_portfolio(self):
        """Construye la cartera siguiendo la estrategia barbell y valor relativo IG/HY."""
        # Filtrar por duración y rating
        short_bonds = self.universe[(self.universe['Duración'] <= 1.5) & (self.universe['Rating'].str.contains('IG'))]
        long_bonds = self.universe[(self.universe['Duración'] > 1.5) & (self.universe['Duración'] <= self.portfolio_constraints['max_duration'])]
        hy_bonds = self.universe[(self.universe['Rating'].str.contains('HY')) & (self.universe['Duración'] <= self.portfolio_constraints['max_duration'])]
        # Selección barbell: mitad en cortos IG, mitad en largos IG, HY solo si cabe en restricción
        n_short = min(len(short_bonds), self.portfolio_constraints['max_bonds']//2)
        n_long = min(len(long_bonds), self.portfolio_constraints['max_bonds']//2)
        n_hy = min(len(hy_bonds), int(self.portfolio_constraints['max_bonds']*self.portfolio_constraints['max_hy_pct']))
        selected = pd.concat([
            short_bonds.head(n_short),
            long_bonds.head(n_long),
            hy_bonds.head(n_hy)
        ])
        # Normalizar pesos
        selected['Peso'] = 1/len(selected)
        self.portfolio = selected.reset_index(drop=True)

    def simulate_scenarios(self, rate_shock=0.01, spread_shock=0.005):
        """Simula escenarios de subida de tipos y ampliación de spreads."""
        # Supone que el universo tiene columnas 'Duración' y 'Spread'
        base_value = (self.portfolio['Valor nominal'] * (1 + self.portfolio['Spread'])).sum()
        # Escenario de subida de tipos
        rate_impact = -self.portfolio['Duración'] * rate_shock * self.portfolio['Valor nominal']
        # Escenario de ampliación de spread
        spread_impact = -self.portfolio['Spread'] * spread_shock * self.portfolio['Valor nominal']
        total_impact = rate_impact + spread_impact
        self.results = {
            'Base Value': base_value,
            'Rate Impact (€)': rate_impact.sum(),
            'Spread Impact (€)': spread_impact.sum(),
            'Total Impact (€)': total_impact.sum()
        }
        return self.results

    def summary(self):
        """Resumen de la estrategia y resultados."""
        return {
            'Portfolio': self.portfolio,
            'Results': self.results
        }

if __name__ == "__main__":
    # Ejemplo de uso
    constraints = {
        'max_bonds': 20,
        'max_duration': 3.0,
        'max_hy_pct': 0.10
    }
    strategy = CustomFixedIncomeStrategy(
        universe_path="universo.csv",
        portfolio_constraints=constraints
    )
    strategy.load_universe()
    strategy.construct_portfolio()
    print("Cartera construida:\n", strategy.portfolio)
    results = strategy.simulate_scenarios(rate_shock=0.01, spread_shock=0.005)
    print("Resultados de escenarios:\n", results)
