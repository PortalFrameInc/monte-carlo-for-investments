"""
CLI pour Monte Carlo Portfolio Simulations.

Usage :
  python -m src.cli simulate --portfolio config/portfolios/example.yaml --config config/base_conf.yaml
  python -m src.cli frontier --portfolio config/portfolios/example.yaml --config config/base_conf.yaml
  python -m src.cli --help
"""

import typer
from pathlib import Path
import yaml
from pydantic import ValidationError

from src.main import cmd_simulate, cmd_frontier
from src.schemas import validate_portfolio, validate_config, format_validation_error

app = typer.Typer(help="Monte Carlo Portfolio Simulations CLI")


class ConfigError(Exception):
    """Erreur de configuration."""
    pass


def load_yaml(file_path: str, name: str) -> dict:
    """
    Charge un fichier YAML.
    
    Args:
        file_path: Chemin vers le fichier YAML
        name: Nom descriptif pour les messages d'erreur
    
    Returns:
        dict: Contenu du fichier YAML
    """
    path = Path(file_path)
    if not path.exists():
        raise ConfigError(f"Fichier {name} non trouvé: {file_path}")
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    if data is None:
        raise ConfigError(f"Fichier {name} vide: {file_path}")
    return data


def load_and_validate_configs(portfolio_path: str, config_path: str) -> dict:
    """
    Charge, valide et fusionne les configurations portfolio et simulation.
    
    Args:
        portfolio_path: Chemin vers le fichier portfolio YAML
        config_path: Chemin vers le fichier de configuration YAML
    
    Returns:
        dict: Configuration fusionnée et validée
    
    Raises:
        ConfigError: Si un fichier est invalide
    """
    # Charger les fichiers
    portfolio_data = load_yaml(portfolio_path, "portfolio")
    config_data = load_yaml(config_path, "configuration")
    
    # Valider le fichier portfolio
    try:
        validate_portfolio(portfolio_data)
    except ValidationError as e:
        raise ConfigError(
            f"Erreur de validation dans {portfolio_path}:\n{format_validation_error(e)}"
        )
    
    # Valider le fichier config
    try:
        validate_config(config_data)
    except ValidationError as e:
        raise ConfigError(
            f"Erreur de validation dans {config_path}:\n{format_validation_error(e)}"
        )
    
    # Fusion: portfolio_data contient securities, portfolio, rebalancing
    # config_data contient general, simulation, frontier
    merged = {
        **config_data,
        "securities": portfolio_data.get("securities", []),
        "portfolio": portfolio_data.get("portfolio", {}),
        "rebalancing": portfolio_data.get("rebalancing", False),
    }
    
    return merged


@app.command()
def simulate(
    portfolio: str = typer.Option(..., '--portfolio', '-p', help='Chemin vers le fichier portfolio YAML'),
    config: str = typer.Option(..., '--config', '-c', help='Chemin vers le fichier de configuration YAML'),
    no_plots: bool = typer.Option(False, '--no-plots', help='Ne pas afficher les graphiques'),
    quiet: bool = typer.Option(False, '--quiet', '-q', help='Mode silencieux (pas de sortie console)')
):
    """Exécuter une simulation Monte-Carlo sur un portefeuille."""
    try:
        config_data = load_and_validate_configs(portfolio, config)
        
        class Args:
            pass
        
        args = Args()
        args.no_plots = no_plots
        args.quiet = quiet
        
        cmd_simulate(args, config_data)
        
        if not quiet:
            typer.echo("Simulation terminée.")
            
    except ConfigError as e:
        typer.echo(f"Erreur de configuration:\n{e}", err=True)
        raise typer.Exit(code=2)
    except Exception as e:
        typer.echo(f"Erreur: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def frontier(
    portfolio: str = typer.Option(..., '--portfolio', '-p', help='Chemin vers le fichier portfolio YAML'),
    config: str = typer.Option(..., '--config', '-c', help='Chemin vers le fichier de configuration YAML'),
    top_n: int = typer.Option(5, '--top-n', help='Nombre de meilleurs portefeuilles à afficher'),
    no_plots: bool = typer.Option(False, '--no-plots', help='Ne pas afficher les graphiques'),
    quiet: bool = typer.Option(False, '--quiet', '-q', help='Mode silencieux (pas de progression)')
):
    """Construire la frontière efficiente."""
    try:
        config_data = load_and_validate_configs(portfolio, config)
        
        class Args:
            pass
        
        args = Args()
        args.top_n = top_n
        args.no_plots = no_plots
        args.quiet = quiet
        
        cmd_frontier(args, config_data)
        
        if not quiet:
            typer.echo("Frontière efficiente construite.")
            
    except ConfigError as e:
        typer.echo(f"Erreur de configuration:\n{e}", err=True)
        raise typer.Exit(code=2)
    except Exception as e:
        typer.echo(f"Erreur: {e}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
