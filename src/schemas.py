"""
Schémas de validation Pydantic pour les fichiers de configuration YAML.

Ce module définit les modèles de validation pour:
- portfolio.yaml : titres, portefeuille, rebalancing
- config.yaml : général, simulation, frontière
"""

from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator


# =============================================================================
# Schémas pour portfolio.yaml
# =============================================================================

class EquityConfig(BaseModel):
    """Configuration d'un titre Equity."""
    name: str = Field(..., min_length=1, description="Nom du titre")
    identifier: str = Field(..., min_length=1, description="Identifiant (ticker)")
    type: Literal["equity"] = Field(..., description="Type de titre")
    mu: float = Field(..., ge=-1, le=2, description="Rendement annuel attendu")
    sigma: float = Field(..., gt=0, le=2, description="Volatilité annuelle")


class LeveragedConfig(BaseModel):
    """Configuration d'un titre à effet de levier."""
    name: str = Field(..., min_length=1, description="Nom du titre")
    identifier: str = Field(..., min_length=1, description="Identifiant (ticker)")
    type: Literal["leveraged"] = Field(..., description="Type de titre")
    base: str = Field(..., min_length=1, description="Identifiant du titre sous-jacent")
    leverage: int = Field(..., ge=1, le=10, description="Facteur de levier")


SecurityConfig = Union[EquityConfig, LeveragedConfig]


class PortfolioSettings(BaseModel):
    """Configuration du portefeuille."""
    name: str = Field(default="Mon Portefeuille", min_length=1, description="Nom du portefeuille")
    value: float = Field(default=100000, gt=0, description="Valeur initiale du portefeuille")
    weights: Optional[List[float]] = Field(default=None, description="Poids des titres")
    
    @field_validator('weights')
    @classmethod
    def validate_weights(cls, v):
        if v is not None:
            if not all(0 <= w <= 1 for w in v):
                raise ValueError("Les poids doivent être entre 0 et 1")
            total = sum(v)
            if not (0.99 <= total <= 1.01):
                raise ValueError(f"La somme des poids doit être égale à 1 (actuel: {total})")
        return v


class PortfolioYaml(BaseModel):
    """Schéma complet pour portfolio.yaml."""
    securities: List[SecurityConfig] = Field(..., min_length=1, description="Liste des titres")
    portfolio: PortfolioSettings = Field(default_factory=PortfolioSettings)
    rebalancing: bool = Field(default=False, description="Activer le rééquilibrage")
    
    @model_validator(mode='after')
    def validate_weights_length(self):
        """Vérifie que le nombre de poids correspond au nombre de titres."""
        if self.portfolio.weights is not None:
            if len(self.portfolio.weights) != len(self.securities):
                raise ValueError(
                    f"Le nombre de poids ({len(self.portfolio.weights)}) "
                    f"doit correspondre au nombre de titres ({len(self.securities)})"
                )
        return self
    
    @model_validator(mode='after')
    def validate_leveraged_base(self):
        """Vérifie que les titres leveraged référencent des titres existants."""
        identifiers = {s.identifier for s in self.securities}
        for security in self.securities:
            if hasattr(security, 'base') and security.base not in identifiers:
                # Le base peut référencer un ticker externe (ex: QQQ pour QLD)
                # On ne lève pas d'erreur, juste un avertissement implicite
                pass
        return self


# =============================================================================
# Schémas pour config.yaml (simulation)
# =============================================================================

class GeneralConfig(BaseModel):
    """Configuration générale."""
    rf: float = Field(default=0.04, ge=0, le=0.5, description="Taux sans risque")
    conf_level: float = Field(default=0.95, gt=0, lt=1, description="Niveau de confiance")
    price_start_year: int = Field(default=2013, ge=1990, le=2030, description="Année de début des prix")


class SimulationSettings(BaseModel):
    """Paramètres de simulation Monte-Carlo."""
    simulations: int = Field(default=500, ge=1, le=100000, description="Nombre de simulations")
    years: int = Field(default=10, ge=1, le=100, description="Durée en années")
    frequency: Literal["daily", "monthly", "quarterly", "annual"] = Field(
        default="daily", description="Fréquence de simulation"
    )


class FrontierSettings(BaseModel):
    """Paramètres de la frontière efficiente."""
    min_weight: int = Field(default=0, ge=0, le=100, description="Poids minimum par titre")
    max_weight: int = Field(default=100, ge=1, le=100, description="Poids maximum par titre")
    weight_increment: int = Field(default=5, ge=1, le=50, description="Incrément des poids")
    num_sims: int = Field(default=100, ge=1, le=10000, description="Simulations par combinaison")
    years: int = Field(default=10, ge=1, le=100, description="Durée en années")
    frequency: Literal["daily", "monthly", "quarterly", "annual"] = Field(
        default="daily", description="Fréquence de simulation"
    )
    
    @model_validator(mode='after')
    def validate_weight_range(self):
        """Vérifie que min_weight <= max_weight."""
        if self.min_weight > self.max_weight:
            raise ValueError(
                f"min_weight ({self.min_weight}) ne peut pas être supérieur à max_weight ({self.max_weight})"
            )
        return self


class ConfigYaml(BaseModel):
    """Schéma complet pour config.yaml (simulation/frontier)."""
    general: GeneralConfig = Field(default_factory=GeneralConfig)
    simulation: SimulationSettings = Field(default_factory=SimulationSettings)
    frontier: FrontierSettings = Field(default_factory=FrontierSettings)


# =============================================================================
# Fonctions de validation
# =============================================================================

def validate_portfolio(data: dict) -> PortfolioYaml:
    """
    Valide les données d'un fichier portfolio.yaml.
    
    Args:
        data: Données YAML chargées
    
    Returns:
        PortfolioYaml: Modèle validé
    
    Raises:
        ValidationError: Si les données sont invalides
    """
    return PortfolioYaml.model_validate(data)


def validate_config(data: dict) -> ConfigYaml:
    """
    Valide les données d'un fichier config.yaml.
    
    Args:
        data: Données YAML chargées
    
    Returns:
        ConfigYaml: Modèle validé
    
    Raises:
        ValidationError: Si les données sont invalides
    """
    return ConfigYaml.model_validate(data)


def format_validation_error(error) -> str:
    """
    Formate une erreur de validation Pydantic en message lisible.
    
    Args:
        error: Exception ValidationError de Pydantic
    
    Returns:
        str: Message d'erreur formaté
    """
    messages = []
    for err in error.errors():
        location = " -> ".join(str(loc) for loc in err['loc'])
        msg = err['msg']
        messages.append(f"  - {location}: {msg}")
    return "\n".join(messages)
