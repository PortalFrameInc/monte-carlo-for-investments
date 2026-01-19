"""
Ce module contient les classes pour modéliser les titres financiers
(Security, Equity, LeveragedEquity, etc.) et les portefeuilles (Portfolio).
"""

import numpy as np
import pandas as pd
import requests
import itertools
from time import sleep
from math import ceil
from joblib import Parallel, delayed

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
)

from .utils import get_optimal_n_jobs, get_api_key

from .visualization import (
    plot_security_prices,
    plot_portfolio_boxplots,
    plot_portfolio_histograms,
    plot_portfolio_simulations,
    plot_portfolio_expected_values,
    plot_efficient_frontier,
)


# =============================================================================
# CLASSE DE BASE: SECURITY
# =============================================================================

class Security:
    """Classe de base pour tous les types de titres financiers."""
    
    def __init__(self, name, identifier, mu, sigma, n, rf, conf_level, seed=None):
        self.name = name
        self.identifier = identifier
        self.mu = mu
        self.sigma = sigma
        self.n = n
        self.rf = rf
        self.conf_level = conf_level
        self.seed = seed
        self.prices = None
        self.generate_returns(self.mu, self.sigma, self.n)
        self.calc_max_drawdown()
        self.calc_cvar(conf_level)
        self.calc_geometric_sharpe_ratio(rf)
    
    def __str__(self):
        return f"{self.name}\n{'='*50}\nIdentifier:{self.identifier}\nRendements:{len(self.returns)}\nMoyenne:{self.mu*100:.2f}%\nÉcart-type:{self.sigma*100:.2f}%\nSharpe Ratio (Géo):{self.sharpe_ratio:.2f}\nCVaR:{self.cvar*100:.2f}%\nDrawdown Max:{self.MDD:.2f}%"
    
    def calc_geometric_sharpe_ratio(self, rf):
        """
        Calcule le Sharpe Ratio géométrique.
        
        Formule: (mu - sigma²/2 - rf) / sigma
        
        Le terme -sigma²/2 est la "pénalité de variance" (volatility drag)
        qui capture l'érosion des rendements composés due à la volatilité.
        
        Contrairement au Sharpe arithmétique standard, le Sharpe géométrique
        pénalise correctement la volatilité pour les investissements long terme.
        """
        # Pénalité de variance (volatility drag)
        variance_penalty = (self.sigma ** 2) / 2
        
        # Rendement géométrique attendu
        geometric_return = self.mu - variance_penalty
        
        # Sharpe géométrique
        sr = round((geometric_return - rf) / self.sigma, 2)
        self.sharpe_ratio = sr
        return sr

    def calc_cvar(self, conf_level):
        """
        Valeur Conditionnelle à Risque (CVaR) pour un intervalle de confiance.
        Calcule la moyenne des pires rendements (en dessous du niveau de confiance).
        """
        # Nombre de rendements
        n = len(self.returns)
        
        # Éviter division par zéro si pas assez de données
        if n == 0:
            self.cvar = 0
            return 0

        # Trie les rendements
        r = np.sort(self.returns)

        # Nombre de rendements en dessous du niveau de confiance
        # Utiliser max(1, ...) pour éviter une slice vide qui retournerait nan
        num_below_conf = max(1, int((1 - conf_level) * n))

        # Moyenne des pires rendements (en dessous du niveau de confiance)
        cvar = np.mean(r[0:num_below_conf])
        
        # Update de l'attribut
        self.cvar = cvar
        return cvar

    def calc_max_drawdown(self):
        """
        Drawdown Max (MDD).
        Combien perdriez-vous si vous aviez acheté au plus haut et vendu au plus bas ?
        """
        # Rendements cumulés
        cumulative_returns = np.cumprod(1 + np.array(self.returns))

        # Obtenir le rendement cumulé max à chaque point de la série temporelle
        max_cumulative_return = np.maximum.accumulate(cumulative_returns)

        # Calculer le drawdown à chaque point de la série
        drawdown = cumulative_returns / max_cumulative_return - 1

        # Trouver le pire drawdown
        max_drawdown = np.min(drawdown)
        
        # Update de l'attribut
        self.MDD = max_drawdown
        return max_drawdown

    def generate_returns(self, mu, sigma, n):
        """Génère n rendements aléatoires suivant une distribution normale."""
        rng = np.random.default_rng(self.seed)
        self.returns = rng.normal(mu, sigma, n)

    def get_price_data(self, from_year=None):
        """
        Récupère les données mensuelles pour un actif (à partir de son symbol) et en utilisant l'api alphavantage
        symbols: GLD (or), SPY (S&P500), LQD (iShares Corp Bond)...
        """
        key = get_api_key()
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY&symbol={self.identifier}&apikey={key}"
        r = requests.get(url)
        d = r.json()

        try:
            # Vérifier si des données ont été retournées
            df = pd.DataFrame(d['Monthly Time Series']).T
        except (KeyError, TypeError):
            print(f"Erreur pour {self.identifier}: {d}")
            self.prices = None
            return

        # Extraire les données dans un DataFrame
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        df['symbol'] = self.identifier

        # Changer les types de données
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # Convertir en float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype('float')

        # Calculer les rendements mensuels
        df['returns'] = df['close'].pct_change()

        # Filtrer par année si fournie
        if from_year is not None:
            df = df.query('index>=@from_year')

        # Calcul de l'espérance et de l'écart-type (annualisés)
        m = (1 + np.mean(df['returns']))**MONTHS_PER_YEAR - 1  # Annualisation géométrique
        s = np.std(df['returns']) * np.sqrt(MONTHS_PER_YEAR)
        
        self.prices = df
        self.returns = df['returns']
        self.mu = m
        self.sigma = s
        self.sharpe_ratio = self.calc_geometric_sharpe_ratio(rf=self.rf)
        self.MDD = self.calc_max_drawdown()
        self.cvar = self.calc_cvar(conf_level=self.conf_level)
        
    def plot_prices(self):
        """Affiche le graphique des prix de clôture."""
        plot_security_prices(self)


# =============================================================================
# EQUITY (ACTIONS/ETF)
# =============================================================================

class Equity(Security):
    """Un type de titre représentant des actions/ETFs."""
    
    def __init__(self, name, identifier, mu, sigma, n, rf, conf_level, seed=None):
        Security.__init__(self, name, identifier, mu, sigma, n, rf, conf_level, seed)
        self.type = 'equity'


# =============================================================================
# LEVERAGED EQUITY (ETF À LEVIER)
# =============================================================================

class LeveragedEquity(Equity):
    """
    Un type de titre à effet de levier basé sur un titre (Equity) sous-jacent.
    Les rendements sont calculés comme leverage_factor * rendements du sous-jacent.
    ⚠️ IMPORTANT: Nécessite frequency='daily' pour capturer correctement la composition quotidienne.
    
    Note sur le calcul des rendements:
    - Cette classe surcharge generate_returns() pour éviter de générer des rendements inutiles.
    - Les vrais rendements sont calculés dans Portfolio.run_simulation() en appliquant
      le levier aux rendements du sous-jacent.
    """
    
    def __init__(self, name, identifier, base_equity, leverage_factor, rf, conf_level, seed=None):
        """
        Args:
            name: nom de l'actif
            identifier: symbole/ticker
            base_equity: l'Equity sous-jacent (ex: QQQ pour NASDAQ-100)
            leverage_factor: facteur de levier (ex: 2 pour 2x, 3 pour 3x)
            rf: taux sans risque
            conf_level: niveau de confiance pour CVaR
            seed: graine aléatoire pour la reproductibilité (optionnel)
        """
        self.base_equity = base_equity
        self.leverage_factor = leverage_factor
        
        # Utiliser les mêmes paramètres que le sous-jacent multiplié par le levier
        leveraged_mu = base_equity.mu * leverage_factor
        leveraged_sigma = base_equity.sigma * leverage_factor
        
        # Initialiser la classe parente
        Equity.__init__(self, name, identifier, leveraged_mu, leveraged_sigma, base_equity.n, rf, conf_level, seed)
        self.type = 'leveraged equity'
    
    def generate_returns(self, mu, sigma, n):
        """Ne pas générer de rendements - ils seront calculés dans run_simulation."""
        self.returns = np.array([])
    
    def __str__(self):
        base_str = Equity.__str__(self)
        return (f"{base_str}\n"
                f"Actif sous-jacent:{self.base_equity.name}\n"
                f"Facteur de levier:{self.leverage_factor}x")
    
    def get_price_data(self, from_year=None):
        """
        Pour un LeveragedEquity, on copie les données du sous-jacent 
        car le levier est appliqué pendant la simulation.
        """
        # S'assurer que le base_equity a ses données
        if self.base_equity.prices is None:
            self.base_equity.get_price_data(from_year)
        
        # Copier les données du sous-jacent (le levier sera appliqué dans run_simulation)
        if self.base_equity.prices is not None:
            self.prices = self.base_equity.prices.copy()
            self.returns = self.base_equity.returns.copy()
            # Mettre à jour mu et sigma avec le levier
            self.mu = self.base_equity.mu * self.leverage_factor
            self.sigma = self.base_equity.sigma * self.leverage_factor
            self.sharpe_ratio = self.calc_geometric_sharpe_ratio(rf=self.rf)
            self.MDD = self.calc_max_drawdown()
            self.cvar = self.calc_cvar(conf_level=self.conf_level)


# =============================================================================
# FIXED INCOME (OBLIGATIONS)
# =============================================================================

class Fixedincome(Security):
    """
    Obligation ou ETF obligataire avec rendement fixe (yield) et faible volatilité.
    Modélise: Rendement total = Yield fixe + Variation de prix (liée aux taux)
    """
    
    def __init__(self, name, identifier, mu, sigma, n, rf, conf_level, yld, duration=5.0, seed=None):
        """
        yld: rendement du coupon annuel (ex: 0.03 pour 3%)
        duration: duration de l'obligation en années (sensibilité aux taux)
        seed: graine aléatoire pour la reproductibilité (optionnel)
        """
        self.yld = yld
        self.duration = duration
        
        # Pour les obligations, sigma devrait être plus faible que pour les actions
        bond_sigma = min(sigma, 0.08)  # Limiter la volatilité max à 8%
        
        Security.__init__(self, name, identifier, mu, bond_sigma, n, rf, conf_level, seed)
        self.type = 'fixed income'
    
    def generate_returns(self, mu, sigma, n):
        """
        Génère des rendements pour une obligation avec composante fixe (yield) 
        et composante variable (variation de prix liée aux taux).
        """
        _ = mu  # Explicitement ignorer mu (requis par signature parente)
        # Composante fixe: le yield (converti en fréquence appropriée)
        fixed_component = np.full(n, self.yld)
        
        # Composante variable: variation du prix (beaucoup plus faible que le yield)
        rng = np.random.default_rng(self.seed)
        price_volatility = sigma * 0.5
        price_variation = rng.normal(0, price_volatility, n)
        
        # Rendement total = Yield + Variation de prix
        min_return = self.yld * 0.5
        self.returns = np.maximum(fixed_component + price_variation, min_return)

    def __str__(self):
        base_str = Security.__str__(self)
        return f"{base_str}\nYield:{self.yld*100:.2f}%\nDuration:{self.duration} ans\nVolatilité des prix: Réduite"


# =============================================================================
# OPTION (NON TERMINÉ)
# =============================================================================

class Option(Security):
    """Option financière (implémentation non terminée)."""
    
    def __init__(self, name, identifier, mu, sigma, n, rf, conf_level, option_type, strike_price, expiration_date, seed=None):
        Security.__init__(self, name, identifier, mu, sigma, n, rf, conf_level, seed)
        self.option_type = option_type  # put/call
        self.strike_price = strike_price
        self.expiration_date = expiration_date
        
    def __str__(self):
        base_str = Security.__str__(self)
        return f"{base_str}\nType d'option:{self.option_type}\nPrix d'exercice:{self.strike_price}\nDate d'expiration:{self.expiration_date}"


# =============================================================================
# FUTURE (NON TERMINÉ)
# =============================================================================

class Future(Security):
    """Contrat à terme (implémentation non terminée)."""
    
    def __init__(self, name, identifier, mu, sigma, n, rf, conf_level, future_type, direction, settlement_price, expiration_date, notional_amount, seed=None):
        Security.__init__(self, name, identifier, mu, sigma, n, rf, conf_level, seed)
        self.future_type = future_type  # futures, forward
        self.direction = direction  # long, short
        self.settlement_price = settlement_price
        self.expiration_date = expiration_date
        self.notional_amount = notional_amount
    
    def __str__(self):
        base_str = Security.__str__(self)
        return f"{base_str}\nType de contrat:{self.future_type}\nDirection:{self.direction}\nPrix de règlement:{self.settlement_price}\nDate d'expiration:{self.expiration_date}\nMontant notionnel:{self.notional_amount}"


# =============================================================================
# PORTFOLIO
# =============================================================================

class Portfolio:
    """
    Classe représentant un portefeuille de titres financiers.
    
    Permet de:
    - Calculer la covariance entre les titres
    - Exécuter des simulations Monte-Carlo
    - Construire la frontière efficiente
    - Visualiser les résultats
    """
    
    def __init__(self, name, securities, target_weights, portfolio_value, rf, conf_level):
        # Validation des poids
        if len(securities) != len(target_weights):
            raise ValueError(f"❌ Le nombre de titres ({len(securities)}) ne correspond pas au nombre de poids ({len(target_weights)})")
        
        weight_sum = sum(target_weights)
        if not np.isclose(weight_sum, 1.0, atol=1e-6):
            raise ValueError(f"❌ La somme des poids doit être égale à 1.0, actuellement: {weight_sum:.6f}")
        
        if any(w < 0 for w in target_weights):
            raise ValueError("❌ Les poids négatifs ne sont pas autorisés (pas de vente à découvert)")
        
        self.name = name
        self.securities = securities
        self.target_weights = target_weights
        self.portfolio_value = portfolio_value
        self.rf = rf
        self.conf_level = conf_level
        self.cov = None  # Covariance mensuelle (source des données)
        self.simulation_results = None
        self.mean_volatility = None
        self.sharpe_ratio = None
        self.sortino_ratio = None
        self.cvar_ratio = None
        self.expected_values = None
        self.cvar = None
        self.DD95 = None
        self.DD75 = None
        self.cagr_mean = None
        self.cagr_q1 = None
        self.cagr_median = None
        self.cagr_q3 = None
        self.efficient_frontier = None
    
    def __str__(self):
        msg = ""
        msg += f"{self.name}\n"
        msg += "=" * 75 + "\n"
        msg += f"Capital investi: ${self.portfolio_value:,.0f}\n"
        
        if self.cagr_mean is not None:
            msg += f"CAGR Moyen: {self.cagr_mean*100:.2f}%\n"
        if self.cagr_q1 is not None and self.cagr_median is not None and self.cagr_q3 is not None:
            msg += f"CAGR Q1: {self.cagr_q1*100:.2f}% | Median: {self.cagr_median*100:.2f}% | Q3: {self.cagr_q3*100:.2f}%\n"
        if self.mean_volatility is not None:
            msg += f"Volatilité moyenne: {self.mean_volatility*100:.2f}% (annualisée)\n"
        if self.sharpe_ratio is not None:
            msg += f"Sharpe Ratio (Géo): {self.sharpe_ratio}\n"
        if self.sortino_ratio is not None:
            msg += f"Sortino Ratio: {self.sortino_ratio}\n"
        if self.cvar_ratio is not None:
            msg += f"CVaR Ratio: {self.cvar_ratio}\n"
        if self.cvar is not None:
            msg += f"CVaR ({int(self.conf_level*100)}%): {self.cvar*100:.02f}% (sur CAGR)\n"
        if self.DD95 is not None:
            msg += f"DD95 (5% pires scénarios): {self.DD95*100:.02f}%\n"
        if self.DD75 is not None:
            msg += f"DD75 (25% pires scénarios): {self.DD75*100:.02f}%\n"
        
        # Obtenir les positions et pondérations
        msg += "Composition du portefeuille:\n"
        for i, sec in enumerate(self.securities):
            if self.target_weights[i] > 0:
                msg += f"  => {sec.name}: {self.target_weights[i]*100:.1f}%\n"
        
        return msg
    
    def calc_geometric_sharpe_ratio(self):
        """
        Calcule le Sharpe Ratio géométrique du portefeuille.
        
        Formule: (CAGR - sigma²/2 - rf) / sigma
        """
        if self.cagr_mean is None or self.mean_volatility is None or self.mean_volatility == 0:
            return 0
        
        variance_penalty = (self.mean_volatility ** 2) / 2
        adjusted_return = self.cagr_mean - variance_penalty
        
        return round((adjusted_return - self.rf) / self.mean_volatility, 2)
    
    def calc_downside_deviation(self, returns, target=0):
        """
        Calcule la downside deviation (semi-écart-type des rendements négatifs).
        """
        downside_returns = returns[returns < target]
        
        if len(downside_returns) == 0:
            return 0
        
        semi_variance = np.mean((downside_returns - target) ** 2)
        return np.sqrt(semi_variance)
    
    def calc_sortino_ratio(self, downside_std):
        """
        Calcule le Sortino Ratio du portefeuille.
        
        Formule: (CAGR - rf) / downside_deviation
        """
        if self.cagr_mean is None or downside_std is None or downside_std == 0:
            return 0
        
        return round((self.cagr_mean - self.rf) / downside_std, 2)
    
    def calc_cvar_ratio(self):
        """
        Calcule le CVaR Ratio du portefeuille.
        
        Formule: (CAGR - rf) / |CVaR|
        """
        if self.cagr_mean is None or self.cvar is None or self.cvar == 0:
            return 0
        
        return round((self.cagr_mean - self.rf) / abs(self.cvar), 2)
    
    def calc_cvar(self, returns, conf_level):
        """
        Valeur Conditionnelle à Risque (CVaR) pour un intervalle de confiance.
        """
        n = len(returns)
        if n == 0:
            return 0

        r = np.sort(returns)
        num_below_conf = max(1, int((1 - conf_level) * n))
        cvar = np.mean(r[0:num_below_conf])
        
        return cvar

    def calc_max_drawdown(self, returns):
        """
        Drawdown Maximum (MDD).
        """
        if len(returns) == 0:
            self.MDD = 0
            return 0
        
        cumulative_returns = np.cumprod(1 + np.array(returns))
        max_cumulative_return = np.maximum.accumulate(cumulative_returns)
        drawdown = cumulative_returns / max_cumulative_return - 1
        max_drawdown = np.min(drawdown)
        
        self.MDD = max_drawdown
        return max_drawdown
    
    def calc_covariance(self):
        """
        Calcule la matrice de covariance MENSUELLE entre les rendements des titres.
        """
        dfs = []
        for sec in self.securities:
            if sec.prices is None:
                print(f"⚠️ Pas de données de prix pour {sec.name}. Exécutez get_security_prices() d'abord.")
                return
            r = pd.DataFrame(sec.prices['returns'])
            r.columns = [f'returns_{sec.identifier}']
            dfs.append(r)
        
        df_r = dfs[0]
        for df in dfs[1:]:
            df_r = df_r.join(df, how='inner')
        
        df_r = df_r.dropna()
        self.cov = df_r.cov()
        self._cov_frequency = 'monthly'
        
    def _convert_covariance(self, target_frequency):
        """
        Convertit la matrice de covariance de mensuelle vers la fréquence cible.
        """
        if self.cov is None:
            raise ValueError("Covariance non calculée. Exécutez calc_covariance() d'abord.")
        
        factor = COV_CONVERSION_FACTORS.get(target_frequency)
        if factor is None:
            raise ValueError(f"Fréquence inconnue: {target_frequency}")
        
        return np.array(self.cov) * factor
    
    def _multivariate_student_t(self, rng, mu, cov, df, size):
        """
        Génère des échantillons d'une distribution Student-t multivariée.
        """
        n_assets = len(mu)
        mu = np.array(mu)
        
        normal_samples = rng.multivariate_normal(
            mean=np.zeros(n_assets), 
            cov=cov, 
            size=size
        )
        
        chi2_samples = rng.chisquare(df=df, size=size)
        scaling_factor = np.sqrt(df / chi2_samples)
        student_t_samples = mu + normal_samples * scaling_factor[:, np.newaxis]
        
        return student_t_samples.T

    def display_covariance_matrix(self):
        """Affiche la matrice de covariance du portefeuille."""
        if self.cov is None:
            print("❌ Aucune matrice de covariance calculée. Exécutez d'abord calc_covariance().")
            return
        
        print("=" * 80)
        print("MATRICE DE COVARIANCE (Mensuelle)")
        print("=" * 80)
        print(self.cov)
        print("\n")
        
    def display_correlation_matrix(self):
        """
        Affiche la matrice de corrélation du portefeuille.
        """
        if self.cov is None:
            print("❌ Aucune matrice de covariance calculée. Exécutez d'abord calc_covariance().")
            return
        
        cov_array = np.array(self.cov)
        std_devs = np.sqrt(np.diag(cov_array))
        corr_matrix = cov_array / np.outer(std_devs, std_devs)
        
        identifiers = [sec.identifier for sec in self.securities]
        corr_df = pd.DataFrame(corr_matrix, index=identifiers, columns=identifiers)
        
        print("=" * 80)
        print("MATRICE DE CORRÉLATION")
        print("=" * 80)
        print("Légende : -1 = corrélation négative parfaite, 0 = pas de corrélation, +1 = corrélation positive parfaite")
        print("-" * 80)
        print(corr_df.round(3))
        print("\n")
        
        print("Paires d'actifs les plus corrélées:")
        print("-" * 80)
        correlations = []
        for i in range(len(identifiers)):
            for j in range(i + 1, len(identifiers)):
                correlations.append((identifiers[i], identifiers[j], corr_matrix[i, j]))
        
        correlations.sort(key=lambda x: abs(x[2]), reverse=True)
        for asset1, asset2, corr in correlations[:5]:
            emoji = "🔴" if corr > 0.7 else "🟡" if corr > 0.4 else "🟢"
            print(f"{emoji} {asset1} ↔ {asset2}: {corr:+.3f}")
        print("\n")

    def convert_from_annual(self, metric, freq, measure):
        """Convertit une métrique annuelle vers une autre fréquence."""
        if measure == 'mean':
            return (1 + metric) ** (1 / FREQUENCY_MAP[freq]) - 1
        elif measure == 'std':
            return metric / np.sqrt(FREQUENCY_MAP[freq])
        return None

    def convert_to_annual(self, metric, freq, years, measure):
        """Convertit une métrique vers une base annuelle."""
        if measure == 'mean':
            return (1 + metric) ** (1 / years) - 1
        elif measure == 'std':
            return metric * np.sqrt(FREQUENCY_MAP[freq])
        return None
        
    def get_security_prices(self, yr, plot=False, requests_per_min=5):
        """Récupère les prix de chaque titre du portefeuille depuis alphavantage."""
        n_sec = len(self.securities)
        if n_sec == 0:
            return None
        
        total_mins = ceil(n_sec / requests_per_min)
        throttle = (total_mins / n_sec) * 60

        for sec in self.securities:
            sec.get_price_data(yr)

            if plot:
                sec.plot_prices()

            sleep(throttle)

    def get_expected_values(self):
        """Calcule les valeurs attendues avec intervalles de confiance."""
        if self.simulation_results is None:
            return None
            
        vals = []
        for _, v in self.simulation_results.items():
            vals.append(np.array(v['portfolio-values']))
        vals = np.array(vals).T

        p = {
            'mean': [],
            'sig1-upper': [],
            'sig1-lower': [],
            'sig2-upper': [],
            'sig2-lower': [],
            'sig3-upper': [],
            'sig3-lower': []
        }
        for i in range(len(vals)):
            sig, m = np.std(vals[i]), np.mean(vals[i])
            p['mean'].append(m)
            p['sig1-upper'].append(m + sig)
            p['sig1-lower'].append(m - sig)
            p['sig2-upper'].append(m + sig * 2)
            p['sig2-lower'].append(m - sig * 2)
            p['sig3-upper'].append(m + sig * 3)
            p['sig3-lower'].append(m - sig * 3)

        return pd.DataFrame(p)
    
    def plot_boxplots(self):
        """Affiche les distributions de rendement/risque en boxplots."""
        plot_portfolio_boxplots(self)

    def plot_histograms(self):
        """Affiche les histogrammes de rendement/risque."""
        plot_portfolio_histograms(self)
    
    def plot_portfolio_simulations(self):
        """Affiche les chemins de simulation du portefeuille."""
        plot_portfolio_simulations(self)
        
    def plot_expected_values(self):
        """Affiche les valeurs attendues avec bandes de confiance."""
        plot_portfolio_expected_values(self)

    def run_single_trajectory(self, sim_id, periods, years, frequency, cov_converted, rebalancing, seed):
        """
        Exécute une seule trajectoire de simulation Monte-Carlo.
        Cette méthode est conçue pour être appelée en parallèle.
        """
        returns = []
        s_values = []
        csr = []
        s_std = []
        result = {
            'security-returns': [],
            'security-values': [],
            'security-total-return': [],
            'security-std': None,
            'portfolio-returns': [],
            'portfolio-values': [],
            'portfolio-total-return': None,
            'portfolio-std': None,
            'portfolio-downside-std': None,
            'portfolio-cvar': None,
            'portfolio-mdd': None,
            'frequency': frequency
        }
        
        portfolio = np.array([w * self.portfolio_value for w in self.target_weights])

        sim_seed = None if seed is None else seed + sim_id
        rng = np.random.default_rng(sim_seed)
        mus = [self.convert_from_annual(m.mu, frequency, 'mean') for m in self.securities]
        rets = self._multivariate_student_t(rng, mus, cov_converted, DEFAULT_STUDENT_T_DF, periods)
        
        # Appliquer le levier pour LeveragedEquity
        for i, sec in enumerate(self.securities):
            if isinstance(sec, LeveragedEquity):
                base_idx = None
                for j, s in enumerate(self.securities):
                    if s is sec.base_equity:
                        base_idx = j
                        break
                
                if base_idx is not None:
                    rets[i] = rets[base_idx] * sec.leverage_factor
                else:
                    base_mu = self.convert_from_annual(sec.base_equity.mu, frequency, 'mean')
                    base_sigma = self.convert_from_annual(sec.base_equity.sigma, frequency, 'std')
                    base_rets = rng.normal(base_mu, base_sigma, periods)
                    rets[i] = base_rets * sec.leverage_factor
        
        for i, sec in enumerate(self.securities):
            returns.append(rets[i])
            s_std.append(np.std(rets[i]))
            s_returns = 1 + rets[i]
            csr.append(s_returns.prod() - 1)
            s_values.append(np.cumprod(s_returns) * portfolio[i])
        
        returns = np.array(returns).T
        
        if frequency == 'daily':
            rebalancing_period = TRADING_DAYS_PER_MONTH
        elif frequency == 'monthly':
            rebalancing_period = 1
        elif frequency == 'quarterly':
            rebalancing_period = 1
        else:
            rebalancing_period = 1
        
        if rebalancing:
            target_weights = np.array(self.target_weights)
            position_values = target_weights * self.portfolio_value
            p_values = []
            p_returns = []
            
            for t in range(periods):
                period_returns = returns[t]
                position_values = position_values * (1 + period_returns)
                total_value = np.sum(position_values)
                p_values.append(total_value)
                
                if t == 0:
                    p_ret = total_value / self.portfolio_value - 1
                else:
                    p_ret = total_value / p_values[t-1] - 1
                p_returns.append(p_ret)
                
                if (t + 1) % rebalancing_period == 0:
                    position_values = target_weights * total_value
            
            p_values = np.array(p_values)
            p_returns = np.array(p_returns)
        else:
            p_returns = returns.dot(self.target_weights)
            p_values = np.cumprod(p_returns + 1) * self.portfolio_value
        
        cpr = p_values[-1] / self.portfolio_value - 1
        p_std = np.std(p_returns)
        p_downside_std = self.calc_downside_deviation(p_returns, target=0)
        cvar_val = self.calc_cvar(p_returns, self.conf_level)
        mdd_val = self.calc_max_drawdown(returns=p_returns)
        
        r = self.convert_to_annual(cpr, frequency, years, 'mean')
        v = self.convert_to_annual(p_std, frequency, years, 'std')
        v_downside = self.convert_to_annual(p_downside_std, frequency, years, 'std')
        
        result['security-returns'] = returns
        result['security-values'] = s_values
        result['security-total-return'] = csr
        result['security-std'] = s_std
        result['portfolio-returns'] = p_returns
        result['portfolio-total-return'] = r
        result['portfolio-values'] = np.insert(p_values, 0, self.portfolio_value)
        result['portfolio-std'] = v
        result['portfolio-downside-std'] = v_downside
        result['portfolio-cvar'] = cvar_val
        result['portfolio-mdd'] = mdd_val
        
        return result

    def run_simulation(self, simulations, years, frequency, rebalancing=False, robust=False, seed=None, n_jobs=None):
        """
        Exécute une simulation de Monte-Carlo avec distribution Student-t multivariée.
        
        Args:
            simulations: nombre de simulations
            years: nombre d'années dans la simulation
            frequency: 'annual', 'quarterly', 'monthly', 'daily'
            rebalancing: si True, rééquilibre le portefeuille à chaque période
            robust: afficher des informations de débogage
            seed: graine aléatoire pour la reproductibilité (optionnel)
            n_jobs: nombre de workers parallèles (None = auto-détection, 1 = séquentiel)
        """
        if self.cov is None:
            raise ValueError("❌ Covariance non calculée. Exécutez calc_covariance() d'abord.")
        
        has_leveraged = any(isinstance(sec, LeveragedEquity) for sec in self.securities)
        if has_leveraged and frequency != 'daily':
            raise ValueError(
                f"❌ ERREUR: Le portefeuille contient des actifs à levier (LeveragedEquity).\n"
                f"Ces actifs nécessitent frequency='daily' pour capturer correctement la composition quotidienne.\n"
                f"Fréquence actuelle: '{frequency}'\n"
                f"Solution: Changez frequency='daily' dans l'appel à run_simulation() ou build_efficient_frontier()"
            )
        
        periods = years * FREQUENCY_MAP[frequency]
        cov_converted = self._convert_covariance(frequency)
        
        if n_jobs is None:
            n_jobs = get_optimal_n_jobs(simulations)
        
        if robust:
            print(f"Simulation Monte-Carlo (Student-t multivariée) - {n_jobs} workers")
            print("=" * 100)
        
        results_list = Parallel(n_jobs=n_jobs, verbose=10 if robust else 0)(
            delayed(self.run_single_trajectory)(
                sim_id=sim,
                periods=periods,
                years=years,
                frequency=frequency,
                cov_converted=cov_converted,
                rebalancing=rebalancing,
                seed=seed
            )
            for sim in range(simulations)
        )
        
        results = {i: r for i, r in enumerate(results_list)}

        all_cagrs = [v['portfolio-total-return'] for v in results.values()]
        cagr_mean = round(np.mean(all_cagrs), 4)
        cagr_q1 = round(np.percentile(all_cagrs, 25), 4)
        cagr_median = round(np.percentile(all_cagrs, 50), 4)
        cagr_q3 = round(np.percentile(all_cagrs, 75), 4)

        pv = round(np.mean([v['portfolio-std'] for v in results.values()]), 4)
        downside_std = round(np.mean([v['portfolio-downside-std'] for v in results.values()]), 4)
        cvar = round(self.calc_cvar(np.array(all_cagrs), self.conf_level), 4)
        
        all_mdds = np.array([v['portfolio-mdd'] for v in results.values()])
        dd95 = round(np.percentile(all_mdds, 5), 4)
        dd75 = round(np.percentile(all_mdds, 25), 4)
        
        self.simulation_results = results
        self.mean_volatility = pv
        self.downside_volatility = downside_std
        self.expected_values = self.get_expected_values()
        self.cvar = cvar
        self.DD95 = dd95
        self.DD75 = dd75
        self.cagr_mean = cagr_mean
        self.cagr_q1 = cagr_q1
        self.cagr_median = cagr_median
        self.cagr_q3 = cagr_q3
        
        self.sharpe_ratio = self.calc_geometric_sharpe_ratio()
        self.sortino_ratio = self.calc_sortino_ratio(downside_std)
        self.cvar_ratio = self.calc_cvar_ratio()

    def weight_combinations(self, total_portfolio_wt, min_security_wt, max_security_wt, weight_increment):
        """Génère toutes les combinaisons de poids possibles pour le portefeuille."""
        n_securities = len(self.securities)
        check = n_securities * max_security_wt >= total_portfolio_wt

        if check:
            weights = list(range(min_security_wt, max_security_wt + weight_increment, weight_increment))
            possible_weights = itertools.product(weights, repeat=n_securities - 1)

            sim_weights = []
            for perm in possible_weights:
                final = total_portfolio_wt - sum(perm)
                if final in weights:
                    wts = perm + (final,)
                    sim_weights.append([w / 100 for w in wts])
        else:
            sim_weights = []
            print("Le poids max par titre est trop faible pour le nombre de titres. Augmentez le poids max par titre.")
        return sim_weights
    
    def build_efficient_frontier(self, total_portfolio_wt, min_security_wt, max_security_wt, weight_increment, num_sims, years, frequency, rebalancing=False, verbose=0, seed=None, n_jobs=None):
        """
        Construit la frontière efficiente en testant toutes les combinaisons de poids.
        
        Args:
            total_portfolio_wt: somme totale des pondérations (100 pour long only)
            min_security_wt: pondération minimale par titre
            max_security_wt: pondération maximale par titre
            weight_increment: incrément entre les pondérations
            num_sims: nombre de simulations Monte-Carlo par combinaison
            years: nombre d'années de simulation
            frequency: fréquence de simulation ('daily', 'monthly', etc.)
            rebalancing: si True, rééquilibre le portefeuille à chaque période
            verbose: afficher la progression tous les N portefeuilles (0 = désactivé)
            seed: graine aléatoire pour la reproductibilité (optionnel)
            n_jobs: nombre de workers parallèles pour les simulations (None = auto-détection)
        """
        weights = self.weight_combinations(total_portfolio_wt, min_security_wt, max_security_wt, weight_increment)

        num_eff_frontier_sims = len(weights)
        if num_eff_frontier_sims == 0:
            print("=> Aucune simulation ne sera exécutée")
            return None
        
        results = {
            'simulation': [],
            'cagr_mean': [],
            'expected-risk': [],
            'sharpe-ratio': [],
            'sortino-ratio': [],
            'cvar-ratio': [],
            'weights': [],
            'cvar': [],
            'dd95': [],
            'dd75': [],
            'cagr-q1': [],
            'cagr-median': [],
            'cagr-q3': []
        }
        
        if verbose > 0:
            print(f"Simulations Frontière Efficiente: {len(weights):,.0f} Portefeuilles Possibles")
            n_workers = n_jobs if n_jobs is not None else get_optimal_n_jobs(num_sims)
            print(f"Parallélisation: {n_workers} workers pour les simulations Monte-Carlo")
            print("=" * 100)
            
        for sim, weight in enumerate(weights):
            if verbose > 0 and sim % verbose == 0:
                progress = round(sim / num_eff_frontier_sims, 2)
                print(f"=>{progress * 100:,.0f}% terminé")
                
            self.target_weights = weight
            
            frontier_seed = None if seed is None else seed + sim
            self.run_simulation(simulations=num_sims, years=years, frequency=frequency, rebalancing=rebalancing, seed=frontier_seed, n_jobs=n_jobs)
            results['simulation'].append(sim)
            results['expected-risk'].append(self.mean_volatility)
            results['sharpe-ratio'].append(self.sharpe_ratio)
            results['sortino-ratio'].append(self.sortino_ratio)
            results['cvar-ratio'].append(self.cvar_ratio)
            results['weights'].append(weight)
            results['cvar'].append(self.cvar)
            results['dd95'].append(self.DD95)
            results['dd75'].append(self.DD75)
            results['cagr_mean'].append(self.cagr_mean)
            results['cagr-q1'].append(self.cagr_q1)
            results['cagr-median'].append(self.cagr_median)
            results['cagr-q3'].append(self.cagr_q3)
            
        self.efficient_frontier = results
        self.target_weights = self.get_optimal_portfolios_by_sharpe_ratio(1)['weights'].item()
    
    def get_optimal_portfolios_by_sharpe_ratio(self, top_n):
        """Retourne les n meilleurs portefeuilles selon le Sharpe Ratio (géométrique)"""
        if self.efficient_frontier is None:
            print("❌ Pas de frontière efficiente. Exécutez build_efficient_frontier() d'abord.")
            return None
            
        frontier_df = pd.DataFrame(self.efficient_frontier)
        frontier_df = frontier_df.sort_values(by='sharpe-ratio', ascending=False)
        return frontier_df.iloc[0:top_n]
    
    def get_optimal_portfolios_by_sortino_ratio(self, top_n):
        """
        Retourne les n meilleurs portefeuilles selon le Sortino Ratio.
        """
        if self.efficient_frontier is None:
            print("❌ Pas de frontière efficiente. Exécutez build_efficient_frontier() d'abord.")
            return None
            
        frontier_df = pd.DataFrame(self.efficient_frontier)
        frontier_df = frontier_df.sort_values(by='sortino-ratio', ascending=False)
        return frontier_df.iloc[0:top_n]
    
    def get_optimal_portfolios_by_cvar_ratio(self, top_n):
        """
        Retourne les n meilleurs portefeuilles selon le CVaR Ratio.
        """
        if self.efficient_frontier is None:
            print("❌ Pas de frontière efficiente. Exécutez build_efficient_frontier() d'abord.")
            return None
            
        frontier_df = pd.DataFrame(self.efficient_frontier)
        frontier_df = frontier_df.sort_values(by='cvar-ratio', ascending=False)
        return frontier_df.iloc[0:top_n]
    
    def get_optimal_portfolios_by_cagr_median(self, top_n):
        """
        Retourne les n meilleurs portefeuilles selon le CAGR médian.
        """
        if self.efficient_frontier is None:
            print("❌ Pas de frontière efficiente. Exécutez build_efficient_frontier() d'abord.")
            return None
            
        frontier_df = pd.DataFrame(self.efficient_frontier)
        frontier_df = frontier_df.sort_values(by='cagr-median', ascending=False)
        return frontier_df.iloc[0:top_n]
    
    def plot_efficient_frontier(self):
        """Affiche le graphique de la frontière efficiente."""
        plot_efficient_frontier(self)
