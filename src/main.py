"""
Fonctions métier pour Monte Carlo Portfolio Simulations.

Ce module expose des fonctions réutilisables dans le CLI ou un notebook.
"""

from typing import List, Optional

from src import (
    RF,
    CONF_LEVEL,
    Security,
    Portfolio,
    create_securities_from_config,
    run_monte_carlo_simulation,
    build_efficient_frontier,
)


def create_portfolio(
    securities: List[Security],
    portfolio_name: str = "Mon Portefeuille",
    portfolio_value: float = 100000,
    weights: Optional[List[float]] = None,
    rf: float = RF,
    conf_level: float = CONF_LEVEL,
    price_start_year: int = 2013,
    verbose: bool = True
) -> Portfolio:
    """
    Crée un Portfolio à partir d'une liste de titres.
    
    Args:
        securities: Liste d'objets Security (Equity, LeveragedEquity, etc.)
        portfolio_name: Nom du portefeuille
        portfolio_value: Valeur initiale du portefeuille
        weights: Poids des titres (équipondéré si None)
        rf: Taux sans risque
        conf_level: Niveau de confiance
        price_start_year: Année de début pour les prix historiques
        verbose: Afficher les messages de progression
    
    Returns:
        Portfolio: Instance du portefeuille configuré
    """
    if not securities:
        raise ValueError("Aucun titre défini dans la configuration.")
    
    # Poids par défaut: équipondéré
    if weights is None:
        weights = [1 / len(securities)] * len(securities)
    
    # Créer le portefeuille
    if verbose:
        print(f"Création du portefeuille: {portfolio_name}")
    portfolio = Portfolio(
        name=portfolio_name,
        securities=securities,
        target_weights=weights,
        portfolio_value=portfolio_value,
        rf=rf,
        conf_level=conf_level
    )
    
    # Récupérer les prix historiques
    if verbose:
        print(f"Récupération des prix depuis Alpha Vantage (depuis {price_start_year})...")
    portfolio.get_security_prices(
        yr=price_start_year,
        plot=False,
        requests_per_min=5
    )
    
    # Calculer la covariance
    if verbose:
        print("Calcul de la matrice de covariance...")
    portfolio.calc_covariance()
    
    return portfolio


def run_simulate(
    portfolio: Portfolio,
    simulations: int = 500,
    years: int = 10,
    frequency: str = "daily",
    rebalancing: bool = False,
    show_plots: bool = True,
    verbose: bool = True
):
    """
    Exécute une simulation Monte-Carlo sur un portefeuille.
    
    Args:
        portfolio: Instance du portefeuille
        simulations: Nombre de simulations
        years: Durée de la simulation en années
        frequency: Fréquence de simulation ("daily", "monthly", "quarterly", "annual")
        rebalancing: Activer le rééquilibrage périodique
        show_plots: Afficher les graphiques
        verbose: Afficher les messages de progression
    
    Returns:
        Résultats de la simulation
    """
    results = run_monte_carlo_simulation(
        portfolio=portfolio,
        simulations=simulations,
        years=years,
        frequency=frequency,
        rebalancing=rebalancing,
        show_plots=show_plots,
        verbose=verbose
    )
    
    return results


def run_frontier(
    portfolio: Portfolio,
    min_weight: int = 0,
    max_weight: int = 100,
    weight_increment: int = 5,
    num_sims: int = 100,
    years: int = 10,
    frequency: str = "daily",
    rebalancing: bool = False,
    show_plot: bool = True,
    top_n: int = 5,
    verbose: int = 1000
):
    """
    Construit la frontière efficiente.
    
    Args:
        portfolio: Instance du portefeuille
        min_weight: Poids minimum par titre (0 = peut être exclu)
        max_weight: Poids maximum par titre
        weight_increment: Incrément des poids
        num_sims: Nombre de simulations par combinaison
        years: Durée de simulation en années
        frequency: Fréquence de simulation
        rebalancing: Activer le rééquilibrage périodique
        show_plot: Afficher le graphique
        top_n: Nombre de meilleurs portefeuilles à retourner
        verbose: Niveau de verbosité (0 = silencieux)
    
    Returns:
        Résultats de la frontière efficiente
    """
    results = build_efficient_frontier(
        portfolio=portfolio,
        min_weight=min_weight,
        max_weight=max_weight,
        weight_increment=weight_increment,
        num_sims=num_sims,
        years=years,
        frequency=frequency,
        rebalancing=rebalancing,
        show_plot=show_plot,
        top_n=top_n,
        verbose=verbose
    )
    
    return results


# =============================================================================
# Fonctions utilitaires pour le CLI (chargement depuis config)
# =============================================================================

def create_portfolio_from_config(config: dict, verbose: bool = True) -> Portfolio:
    """
    Crée un Portfolio à partir d'une configuration fusionnée (pour le CLI).
    
    Args:
        config: Configuration fusionnée (portfolio + simulation)
        verbose: Afficher les messages de progression
    
    Returns:
        Portfolio: Instance du portefeuille configuré
    """
    general = config.get('general', {})
    portfolio_config = config.get('portfolio', {})
    rf = general.get('rf', RF)
    conf_level = general.get('conf_level', CONF_LEVEL)
    
    # Créer les titres depuis la configuration
    if verbose:
        print("Création des titres...")
    securities = create_securities_from_config(
        config.get('securities', []),
        rf,
        conf_level
    )
    
    return create_portfolio(
        securities=securities,
        portfolio_name=portfolio_config.get('name', 'Mon Portefeuille'),
        portfolio_value=portfolio_config.get('value', 100000),
        weights=portfolio_config.get('weights'),
        rf=rf,
        conf_level=conf_level,
        price_start_year=general.get('price_start_year', 2013),
        verbose=verbose
    )


def cmd_simulate(args, config_data: dict):
    """
    Commande CLI: Exécuter une simulation Monte-Carlo.
    """
    portfolio = create_portfolio_from_config(config_data, verbose=not args.quiet)
    
    sim_config = config_data.get('simulation', {})
    
    return run_simulate(
        portfolio=portfolio,
        simulations=sim_config.get('simulations', 500),
        years=sim_config.get('years', 10),
        frequency=sim_config.get('frequency', 'daily'),
        rebalancing=config_data.get('rebalancing', False),
        show_plots=not args.no_plots,
        verbose=not args.quiet
    )


def cmd_frontier(args, config_data: dict):
    """
    Commande CLI: Construire la frontière efficiente.
    """
    portfolio = create_portfolio_from_config(config_data, verbose=not args.quiet)
    
    frontier_config = config_data.get('frontier', {})
    
    return run_frontier(
        portfolio=portfolio,
        min_weight=frontier_config.get('min_weight', 0),
        max_weight=frontier_config.get('max_weight', 100),
        weight_increment=frontier_config.get('weight_increment', 5),
        num_sims=frontier_config.get('num_sims', 100),
        years=frontier_config.get('years', 10),
        frequency=frontier_config.get('frequency', 'daily'),
        rebalancing=config_data.get('rebalancing', False),
        show_plot=not args.no_plots,
        top_n=args.top_n,
        verbose=0 if args.quiet else 1000
    )
