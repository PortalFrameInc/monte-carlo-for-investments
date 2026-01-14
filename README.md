# monte-carlo-for-investments

Fork du projet [monte-carlo-for-investments](https://github.com/kconstable/monte-carlo-for-investments) de kconstable, dont l'objectif initial était de lancer des simulations sur d'autres actifs, cependant j'ai fait quelques ajouts pour mieux correspondre à mes besoins.

## Description

Dans ce projet, j'utilise des simulations de Monte-Carlo pour sélectionner les pondérations optimales d'un portefeuille d'actifs qui offre le meilleur rendement ajusté au risque.
Il y a deux classes principales : **Security** et **Portfolio**. Les Securities (titres) sont des actifs individuels pour lesquels nous pouvons récupérer les prix historiques via l'API [Alphavantage](https://www.alphavantage.co/). Les Portfolios (portefeuilles) sont composés de paniers de titres avec leurs pondérations associées. L'annexe contient les détails sur les attributs et méthodes utilisés dans chaque classe.

Dans une simulation de Monte-Carlo, nous pouvons déduire le rendement et le risque les plus probables d'un panier d'actifs en nous basant sur les rendements de chaque actif (mu), le risque (sigma), et la covariance entre les actifs du portefeuille. Nous générons des rendements aléatoires normaux pour chaque titre du portefeuille, qui sont corrélés avec les autres rendements aléatoires générés. Nous pouvons ensuite déterminer le rendement pondéré du portefeuille à chaque pas de temps sur une période fixe. Cela nous permet de déterminer la valeur finale du portefeuille. Si nous générons de nombreux chemins, nous pouvons déterminer le rendement et le risque moyens du portefeuille pour un portefeuille donné.

## Installation

### Environnement Virtuel Python (Recommandé)

Je recommande l'utilisation d'un environnement virtuel pour isoler les dépendances du projet :

```bash
# Créer un environnement virtuel
python -m venv venv

# Activer l'environnement virtuel
# Sur Windows :
venv\Scripts\activate
# Sur macOS/Linux :
source venv/bin/activate
```

### Dépendances

Installer les dépendances requises :

```bash
pip install -r requirements.txt
```

### Configuration de la Clé API

Ce projet utilise l'API [Alphavantage](https://www.alphavantage.co/) pour récupérer les prix historiques des titres. Vous aurez besoin de :

1. Obtenir une clé API gratuite depuis [Alphavantage](https://www.alphavantage.co/support/#api-key)
2. Créer un répertoire `keys` à la racine du projet
3. Sauvegarder votre clé API dans `keys/alphavantage.txt`

```bash
mkdir keys
echo "API_KEY" > keys/alphavantage.txt
```

### Utilisation

Ouvrir et exécuter le notebook Jupyter :

```bash
jupyter notebook monte_carlo.ipynb
```

## Prix des Securities (Actifs)

L'API Alphavantage a été utilisée pour collecter les prix mensuels de 8 titres. À partir des prix, nous pouvons calculer le rendement moyen, la volatilité (écart-type des rendements), et les corrélations des rendements entre chaque titre sous forme de matrice de covariance.
![image](https://user-images.githubusercontent.com/1649676/215936189-a17f9410-1bc5-4f7e-a7c0-63912f9bdfea.png)
![image](https://user-images.githubusercontent.com/1649676/215936252-53c6971d-ecd0-43c5-b9f4-a51713f9e08f.png)

## Monte-Carlo : Simulation de Portefeuille

Nous commençons avec un portefeuille de titres pondérés et générons 100 chemins possibles. Pour que notre simulation soit réaliste, il est important que nos rendements générés aléatoirement soient **corrélés** entre eux.

Par exemple, si les rendements de deux titres ont une corrélation de +0.80, nous devons nous assurer que nos rendements générés aléatoirement ont une corrélation similaire. Le graphique ci-dessous montre les 100 chemins possibles qu'un portefeuille avec une valeur initiale de 100k a suivi sur une période de 10 ans. À partir d'une simulation test, nous pouvons calculer le rendement annualisé moyen du portefeuille (2.0%) et la volatilité (9.68%).

![image](https://user-images.githubusercontent.com/1649676/216056238-5ca11045-4dc9-4fd4-8a7d-775712c594fb.png)

## Frontière Efficiente

La théorie moderne du portefeuille suggère que nous devrions être récompensés pour investir dans des actifs risqués. Une mesure standard du rapport risque/récompense est le **ratio de Sharpe**. C'est un ratio du rendement attendu du portefeuille moins le rendement que nous pouvons attendre d'un actif sans risque (comme un bon du Trésor), divisé par la volatilité attendue du portefeuille. Plus le ratio est élevé, meilleur est le rendement ajusté au risque.

`ratio de Sharpe = (Rendement Attendu - Taux Sans Risque) / Volatilité Attendue`

Notre portefeuille de départ avec des titres équipondérés a un ratio de Sharpe de -0.20\*. Cela signifie que nous ne sommes pas suffisamment récompensés pour investir dans des actifs risqués sur une base ajustée au risque. Autrement dit, nous ferions mieux d'investir dans des bons du Trésor qui ont un rendement garanti susceptible d'entraîner une valeur de portefeuille plus élevée.

\*_en supposant un taux sans risque de 4.0%, ce que les bons du Trésor américains rapportaient au moment de l'écriture_

Nous pouvons utiliser des simulations de Monte-Carlo pour trouver le portefeuille optimal à partir du panier de titres en ajustant systématiquement chaque pondération de titre, en exécutant une simulation, et en suivant le rendement et le risque attendus. Nous pouvons ensuite tracer chaque portefeuille par rendement/risque. Le graphique ci-dessous montre les résultats de la simulation suivante :

- Ajuster les pondérations des 8 titres par incréments de 10% (c.-à-d. 0%, 10%, 20%, 30%)
- Aucun titre individuel ne devrait avoir une pondération supérieure à 30% du portefeuille
- Nous n'avons pas besoin de détenir les huit titres
- Simuler chaque combinaison de pondérations possible et calculer le rendement attendu, le risque et le ratio de Sharpe

Nous pouvons voir sur le graphique que nous préférerions les portefeuilles sur le bord supérieur car ils ont des rendements attendus plus élevés par unité de risque attendu. Ce graphique est connu sous le nom de frontière efficiente. Les couleurs représentent le ratio de Sharpe.

![image](https://user-images.githubusercontent.com/1649676/216845740-c0d9067e-4022-4783-8073-d851d4b097bf.png)

### Pondérations les Plus Efficientes

Le tableau ci-dessous montre les pondérations des cinq meilleurs portefeuilles triés par ratio de Sharpe. Comme vous pouvez le voir, la simulation a révélé que les meilleurs portefeuilles ajustés au risque détiennent entre 4 et 6 des 8 titres.

![image](https://user-images.githubusercontent.com/1649676/216845702-0c7e8dc0-86af-4b64-bb53-697f7d99f571.png)

## Simulation de Monte-Carlo : Risque, Rendement et Ratio de Sharpe Attendus

Nous pouvons maintenant sélectionner l'un des meilleurs portefeuilles et relancer la simulation de Monte-Carlo pour voir comment notre portefeuille optimal se compare à notre portefeuille de départ. Dans ce cas, nous avons exécuté la simulation 2 500 fois.

### Chemins de Simulation

Le premier graphique montre les 2 500 chemins possibles qu'a pris notre portefeuille dans la simulation.

![image](https://user-images.githubusercontent.com/1649676/216846052-2ab2d807-e192-47e4-9796-1c032d34ac6a.png)

### Distributions Rendement/Risque

Les boîtes à moustaches montrent la distribution des rendements et de la volatilité du portefeuille. Les distributions de la valeur conditionnelle à risque (CVaR) et du drawdown maximum fournissent des informations sur le potentiel de baisse du portefeuille.

![image](https://user-images.githubusercontent.com/1649676/216846071-527a9b2b-1ae0-49fa-838e-c8634d03b2a3.png)
![image](https://user-images.githubusercontent.com/1649676/216846086-61980918-28d3-4af5-94fd-d2b8bf7a9e23.png)

### Valeur Attendue du Portefeuille après 10 Ans

Le portefeuille optimal a un rendement attendu de 5.51%, un risque de 13.08%, et un ratio de Sharpe de 0.12. Le portefeuille a le plus de chances d'avoir une valeur finale entre $103,268 et $267,331. C'est une grande amélioration par rapport au portefeuille de départ avec des titres équipondérés.

Allocations

- SPDR S&P 500 ETF Trust : 30.0%
- iShares MSCI EAFE : 10.0%
- Mackenzie Canadian Equity Index : 30.0%
- BMO Aggregate Bond Index : 10.0%
- Gold ETF : 20.0%

Vous pouvez également utiliser cette méthode pour évaluer des portefeuilles avec différents titres et pondérations.

![image](https://user-images.githubusercontent.com/1649676/216846107-0d9beec6-0150-479c-849d-c979ec131056.png)

## Annexe

#### Attributs de Security

- name = nom du titre
- identifier (utilisé pour interroger alphavantage pour les prix)
- mu (rendement moyen)
- sigma (écart-type moyen des rendements)

#### Méthodes de Security

- generate_returns(self.mu, self.sigma, self.n). Cette fonction simule les rendements à partir de la distribution normale basée sur mu et sigma

#### Attributs de Portfolio

Un portefeuille contient des titres avec des pondérations spécifiques

- name = nom du portefeuille
- securities = une liste de titres
- target_weights = une liste de pondérations associées à chaque titre
- portfolio_value = valeur initiale du portefeuille
- rf = taux sans risque (format décimal)
- cov = matrice de covariance des rendements de chaque titre dans le portefeuille
- simulation_results = un dictionnaire de diverses sorties de la simulation de Monte-Carlo
- mean_return = rendement moyen du portefeuille. Calculé par la simulation
- mean_volatility = écart-type moyen des rendements du portefeuille - calculé par la simulation
- sharpe_ratio = ratio de Sharpe du portefeuille - calculé par la simulation
- expected_values = valeurs attendues du portefeuille - calculées par la simulation

#### Méthodes de Portfolio

- calc_sharpe_ratio() : Calculer le ratio de Sharpe du portefeuille
- calc_covariance() : Calculer la covariance entre les rendements des titres
- convert_from_annual(metric,freq,measure) : convertir d'annuel à une autre fréquence (rendement, risque)
- convert_to_annual(ret,vol,freq,periods) : convertir rendement/risque en annuel depuis une autre fréquence
- get_security_prices(yr, key, plot=False,requests_per_min=5) : obtenir les prix des titres depuis alpha vantage
- run_simulation(simulations, years, frequency,robust=False) : exécuter la simulation de Monte-Carlo
- get_expected_values() : obtenir les valeurs attendues de la simulation
- weight_combinations(, total_portfolio_wt, min_security_wt, max_security_wt, weight_increment) : obtenir toutes les combinaisons possibles de pondérations de portefeuille
- build_efficient_frontier(total_portfolio_wt, min_security_wt, max_security_wt, weight_increment, num_sims, years, frequency, verbose=0) : construire la frontière efficiente
- get_optimal_portfolios(top_n) : obtenir le portefeuille optimal de la frontière efficiente
- plot_boxplots() : tracer rendement/risque en boîtes à moustaches
- plot_histograms() : tracer rendement/risque en histogrammes
- plot_portfolio_simulations() : tracer les résultats de la simulation
- plot_efficient_frontier() : tracer la frontière efficiente
