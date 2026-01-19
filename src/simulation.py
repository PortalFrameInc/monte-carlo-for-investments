"""
Ce module contient les fonctions d'orchestration pour les simulations Monte-Carlo
et la construction de frontières efficientes.

Ces fonctions de haut niveau encapsulent les workflows complets et seront
utilisées par le CLI.
"""

from typing import Optional

from .models import Portfolio, Equity, LeveragedEquity
from .config import RF, CONF_LEVEL

def create_securities_from_config(securities_config: list, rf: float = RF, conf_level: float = CONF_LEVEL) -> list:
    """
    Crée des objets Security à partir d'une configuration.
    
    Args:
        securities_config: liste de dictionnaires décrivant les titres
        rf: taux sans risque
        conf_level: niveau de confiance
        
    Returns:
        Liste d'objets Security (Equity, LeveragedEquity, etc.)
        
    Example:
        securities_config = [
            {"name": "S&P 500", "identifier": "SPY", "mu": 0.10, "sigma": 0.15, "type": "equity"},
            {"name": "QQQ 2x", "identifier": "QLD", "base": "QQQ", "leverage": 2, "type": "leveraged"}
        ]
    """
    securities = []
    securities_by_id = {}  # Pour les références aux sous-jacents
    
    # Première passe : créer les titres de base (non-leveraged)
    for config in securities_config:
        sec_type = config.get("type", "equity")
        
        if sec_type == "equity":
            sec = Equity(
                name=config["name"],
                identifier=config["identifier"],
                mu=config["mu"],
                sigma=config["sigma"],
                n=config.get("n", 100),
                rf=rf,
                conf_level=conf_level
            )
            securities.append(sec)
            securities_by_id[config["identifier"]] = sec
    
    # Deuxième passe : créer les titres à effet de levier
    for config in securities_config:
        sec_type = config.get("type", "equity")
        
        if sec_type == "leveraged":
            base_id = config["base"]
            if base_id not in securities_by_id:
                raise ValueError(
                    f"❌ Titre sous-jacent '{base_id}' non trouvé pour {config['name']}.\n"
                    f"Assurez-vous que le titre de base est défini avant le titre à levier."
                )
            
            sec = LeveragedEquity(
                name=config["name"],
                identifier=config["identifier"],
                base_equity=securities_by_id[base_id],
                leverage_factor=config["leverage"],
                rf=rf,
                conf_level=conf_level
            )
            securities.append(sec)
            securities_by_id[config["identifier"]] = sec
    
    return securities

def run_monte_carlo_simulation(
    portfolio: Portfolio,
    simulations: int = 500,
    years: int = 10,
    frequency: str = "monthly",
    rebalancing: bool = False,
    show_plots: bool = False,
    verbose: bool = True
) -> dict:
    """
    Exécute une simulation Monte-Carlo sur un portefeuille.
    
    Cette fonction encapsule le workflow complet :
    1. Exécuter la simulation
    2. Afficher les graphiques (optionnel)
    3. Retourner les résultats
    
    Args:
        portfolio: le portefeuille à simuler
        simulations: nombre de simulations Monte-Carlo
        years: durée de la simulation en années
        frequency: fréquence ('daily', 'monthly', 'quarterly', 'annual')
        rebalancing: si True, rééquilibre à chaque période
        show_plots: si True, affiche les graphiques
        verbose: si True, affiche les informations de progression
        
    Returns:
        Dictionnaire contenant les résultats de la simulation :
        - 'portfolio': l'objet Portfolio avec les résultats
        - 'cagr_mean': rendement annualisé moyen
        - 'cagr_median': rendement annualisé médian
        - 'volatility': volatilité annualisée
        - 'sharpe_ratio': ratio de Sharpe géométrique
        - 'sortino_ratio': ratio de Sortino
        - 'cvar': CVaR au niveau de confiance
        - 'dd95': drawdown au 5ème percentile
    """
    if verbose:
        print(f"🎲 Simulation Monte-Carlo")
        print(f"   Portefeuille: {portfolio.name}")
        print(f"   Simulations: {simulations:,}")
        print(f"   Durée: {years} ans")
        print(f"   Fréquence: {frequency}")
        print(f"   Rééquilibrage: {'Oui' if rebalancing else 'Non'}")
        print("=" * 60)
    
    # Exécuter la simulation
    portfolio.run_simulation(
        simulations=simulations,
        years=years,
        frequency=frequency,
        rebalancing=rebalancing
    )
    
    # Afficher les graphiques
    if show_plots:
        portfolio.plot_portfolio_simulations()
        portfolio.plot_boxplots()
        portfolio.plot_expected_values()
    
    # Afficher le résumé
    if verbose:
        print(portfolio)
    
    # Retourner les résultats
    return {
        'portfolio': portfolio,
        'cagr_mean': portfolio.cagr_mean,
        'cagr_median': portfolio.cagr_median,
        'cagr_q1': portfolio.cagr_q1,
        'cagr_q3': portfolio.cagr_q3,
        'volatility': portfolio.mean_volatility,
        'downside_volatility': portfolio.downside_volatility,
        'sharpe_ratio': portfolio.sharpe_ratio,
        'sortino_ratio': portfolio.sortino_ratio,
        'cvar_ratio': portfolio.cvar_ratio,
        'cvar': portfolio.cvar,
        'dd95': portfolio.DD95,
        'dd75': portfolio.DD75,
        'simulation_results': portfolio.simulation_results,
        'expected_values': portfolio.expected_values
    }

def build_efficient_frontier(
    portfolio: Portfolio,
    total_weight: int = 100,
    min_weight: int = 0,
    max_weight: int = 100,
    weight_increment: int = 5,
    num_sims: int = 100,
    years: int = 10,
    frequency: str = "daily",
    rebalancing: bool = False,
    show_plot: bool = True,
    top_n: int = 5,
    verbose: int = 1000,
    seed: Optional[int] = None
) -> dict:
    """
    Construit la frontière efficiente d'un portefeuille.
    
    Cette fonction encapsule le workflow complet :
    1. Générer toutes les combinaisons de poids
    2. Exécuter des simulations pour chaque combinaison
    3. Identifier les portefeuilles optimaux
    4. Afficher le graphique (optionnel)
    
    Args:
        portfolio: le portefeuille de base (avec les titres)
        total_weight: poids total du portefeuille (100 = long only)
        min_weight: poids minimum par titre (0 = peut être exclu)
        max_weight: poids maximum par titre
        weight_increment: incrément des poids (5 = 5%, 10%, 15%...)
        num_sims: nombre de simulations par combinaison
        years: durée de simulation
        frequency: fréquence de simulation
        rebalancing: si True, rééquilibre à chaque période
        show_plot: si True, affiche le graphique de la frontière
        top_n: nombre de meilleurs portefeuilles à retourner
        verbose: afficher la progression tous les N portefeuilles
        seed: graine aléatoire pour la reproductibilité
        
    Returns:
        Dictionnaire contenant :
        - 'portfolio': l'objet Portfolio avec la frontière
        - 'optimal_weights': les poids optimaux selon Sharpe
        - 'top_portfolios_sharpe': DataFrame des meilleurs par Sharpe
        - 'top_portfolios_sortino': DataFrame des meilleurs par Sortino
        - 'top_portfolios_cvar': DataFrame des meilleurs par CVaR
        - 'num_combinations': nombre de combinaisons testées
        - 'efficient_frontier': données brutes de la frontière
    """
    # Calculer le nombre de combinaisons
    combinations = portfolio.weight_combinations(
        total_portfolio_wt=total_weight,
        min_security_wt=min_weight,
        max_security_wt=max_weight,
        weight_increment=weight_increment
    )
    num_combinations = len(combinations)
    
    print(f"📈 Construction de la Frontière Efficiente")
    print(f"   Portefeuille: {portfolio.name}")
    print(f"   Titres: {len(portfolio.securities)}")
    print(f"   Combinaisons de poids: {num_combinations:,}")
    print(f"   Simulations par combinaison: {num_sims}")
    print(f"   Total de simulations: {num_combinations * num_sims:,}")
    print(f"   Durée: {years} ans")
    print(f"   Fréquence: {frequency}")
    print("=" * 60)
    
    if num_combinations == 0:
        print("❌ Aucune combinaison de poids valide. Ajustez les paramètres.")
        return None
    
    # Construire la frontière
    portfolio.build_efficient_frontier(
        total_portfolio_wt=total_weight,
        min_security_wt=min_weight,
        max_security_wt=max_weight,
        weight_increment=weight_increment,
        num_sims=num_sims,
        years=years,
        frequency=frequency,
        rebalancing=rebalancing,
        verbose=verbose,
        seed=seed
    )
    
    # Afficher le graphique
    if show_plot:
        portfolio.plot_efficient_frontier()
    
    # Récupérer les meilleurs portefeuilles
    top_sharpe = portfolio.get_optimal_portfolios_by_sharpe_ratio(top_n)
    top_sortino = portfolio.get_optimal_portfolios_by_sortino_ratio(top_n)
    top_cvar = portfolio.get_optimal_portfolios_by_cvar_ratio(top_n)
    top_cagr = portfolio.get_optimal_portfolios_by_cagr_median(top_n)
    
    # Afficher les résultats
    print("\n" + "=" * 60)
    print(f"🏆 Top {top_n} Portefeuilles par Sharpe Ratio Géométrique")
    print("=" * 60)
    print(top_sharpe[['weights', 'cagr_mean', 'expected-risk', 'sharpe-ratio']].to_string())
    
    return {
        'portfolio': portfolio,
        'optimal_weights': portfolio.target_weights,
        'top_portfolios_sharpe': top_sharpe,
        'top_portfolios_sortino': top_sortino,
        'top_portfolios_cvar': top_cvar,
        'top_portfolios_cagr': top_cagr,
        'num_combinations': num_combinations,
        'efficient_frontier': portfolio.efficient_frontier
    }
