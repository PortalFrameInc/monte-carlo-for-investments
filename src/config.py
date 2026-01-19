"""
Ce module contient toutes les constantes globales et fonctions de configuration
utilisées dans les simulations Monte-Carlo.
"""

import os


# =============================================================================
# CONFIGURATION PARALLÉLISATION
# =============================================================================

def get_optimal_n_jobs(n_tasks=None):
    """
    Détermine le nombre optimal de workers basé sur les ressources CPU.
    
    Args:
        n_tasks: nombre de tâches à exécuter (optionnel)
        
    Returns:
        int: nombre de jobs parallèles optimal
    """
    # Nombre de cœurs logiques
    n_cores = os.cpu_count() or 4
    
    # Garder 1 cœur libre pour le système
    max_workers = max(1, n_cores - 1)
    
    # Si peu de tâches, ne pas utiliser plus de workers que de tâches
    if n_tasks is not None:
        max_workers = min(max_workers, n_tasks)
    
    return max_workers


# =============================================================================
# CONSTANTES GLOBALES
# =============================================================================

# Paramètres financiers
RF = 0.04                # Taux sans risque (Risk-Free Rate)
CONF_LEVEL = 0.95        # Niveau de confiance (utilisé pour calculer la CVaR)

# Jours/périodes de trading
TRADING_DAYS_PER_YEAR = 252
TRADING_DAYS_PER_MONTH = 21
MONTHS_PER_YEAR = 12
QUARTERS_PER_YEAR = 4

# Dictionnaire de fréquences (périodes par an)
FREQUENCY_MAP = {
    'annual': 1,
    'quarterly': QUARTERS_PER_YEAR,
    'monthly': MONTHS_PER_YEAR,
    'daily': TRADING_DAYS_PER_YEAR
}

# Facteurs de conversion de covariance (mensuel vers autre fréquence)
COV_CONVERSION_FACTORS = {
    'monthly': 1,
    'annual': 12,
    'quarterly': 3,
    'daily': 1 / TRADING_DAYS_PER_MONTH
}

# Degrés de liberté par défaut pour la distribution Student-t
# ν ≈ 4-6 est typique pour les actions (fat tails)
# ν → ∞ converge vers la distribution normale
DEFAULT_STUDENT_T_DF = 5

# Labels pour les graphiques
FREQUENCY_LABELS = {
    'annual': 'Années',
    'monthly': 'Mois',
    'quarterly': 'Trimestres',
    'daily': 'Jours'
}


# =============================================================================
# PALETTE DE COULEURS
# =============================================================================

class Colors:
    """Palette de couleurs pour les graphiques Plotly."""
    PRIMARY = 'rgb(83,128,141)'
    PRIMARY_ALPHA = 'rgba(83,128,141,0.5)'
    PRIMARY_LIGHT = 'rgba(83,128,141,0.4)'
    PRIMARY_DARK = 'rgba(83,128,141,0.8)'
    PRIMARY_LIGHTER = 'rgba(83,128,141,0.3)'
    PRIMARY_LIGHTEST = 'rgba(83,128,141,0.1)'
    ACCENT = '#FC4C01'
