# Monte Carlo for Investments - Core Module
"""
Ce module contient les classes et fonctions principales pour la simulation
Monte-Carlo de portefeuilles d'investissement.
"""

from .config import (
    RF,
    CONF_LEVEL,
    TRADING_DAYS_PER_YEAR,
    TRADING_DAYS_PER_MONTH,
    MONTHS_PER_YEAR,
    QUARTERS_PER_YEAR,
    FREQUENCY_MAP,
    COV_CONVERSION_FACTORS,
    DEFAULT_STUDENT_T_DF,
    FREQUENCY_LABELS,
    Colors,
    get_optimal_n_jobs,
)

from .models import (
    Security,
    Equity,
    LeveragedEquity,
    Fixedincome,
    Option,
    Future,
    Portfolio,
)

__all__ = [
    # Config
    'RF',
    'CONF_LEVEL',
    'TRADING_DAYS_PER_YEAR',
    'TRADING_DAYS_PER_MONTH',
    'MONTHS_PER_YEAR',
    'QUARTERS_PER_YEAR',
    'FREQUENCY_MAP',
    'COV_CONVERSION_FACTORS',
    'DEFAULT_STUDENT_T_DF',
    'FREQUENCY_LABELS',
    'Colors',
    'get_optimal_n_jobs',
    # Models
    'Security',
    'Equity',
    'LeveragedEquity',
    'Fixedincome',
    'Option',
    'Future',
    'Portfolio',
]
