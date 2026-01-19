# =============================================================================
# FONCTIONS DE VISUALISATION
# =============================================================================
"""
Ce module contient toutes les fonctions de visualisation Plotly
pour les titres et portefeuilles.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from .config import Colors, FREQUENCY_LABELS


# =============================================================================
# VISUALISATION DES TITRES (SECURITY)
# =============================================================================

def plot_security_prices(security):
    """
    Affiche le graphique des prix de clôture d'un titre.
    
    Args:
        security: objet Security avec attributs prices, name, identifier
    """
    if security.prices is None:
        print(f"❌ Pas de données de prix pour {security.name}")
        return
        
    df = security.prices
    hover_temp = "<b>Date:</b>%{x}<br>" + "<b>Prix de clôture:</b>%{y:,.0f}<extra></extra>"
    fig = go.Figure(
        go.Scatter(
            x=df.index,
            y=df.close,
            line=dict(color=Colors.PRIMARY, width=2),
            fill='tozeroy',
            fillcolor=Colors.PRIMARY_ALPHA,
            hovertemplate=hover_temp
        )
    )
    fig.update_layout(
        template='plotly_white',
        title=f"{security.name} ({security.identifier})",
        yaxis_title='Prix de clôture',
        width=500, height=500
    )
    fig.show()


# =============================================================================
# VISUALISATION DES PORTEFEUILLES
# =============================================================================

def plot_portfolio_boxplots(portfolio):
    """
    Affiche les distributions de rendement/risque en boxplots.
    
    Args:
        portfolio: objet Portfolio avec simulation_results
    """
    if portfolio.simulation_results is None:
        print("❌ Pas de résultats de simulation. Exécutez run_simulation() d'abord.")
        return
        
    pr = [v['portfolio-total-return'] * 100 for v in portfolio.simulation_results.values()]
    pv = [v['portfolio-std'] * 100 for v in portfolio.simulation_results.values()]
    cvar = [v['portfolio-cvar'] * 100 for v in portfolio.simulation_results.values()]
    mdd = [v['portfolio-mdd'] * 100 for v in portfolio.simulation_results.values()]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Taux de croissance annuel composé', 'Volatilité (Annuelle)', 'Valeur à risque conditionnelle', 'Drawdown Max'],
        shared_yaxes=True
    )

    fig.add_trace(
        go.Box(
            y=pr,
            name="CAGR(%)",
            boxpoints='all', 
            marker_color=Colors.PRIMARY_LIGHT,
            line_color=Colors.PRIMARY
        ),
        row=1,
        col=1
    )
    fig.add_trace(
        go.Box(
            y=pv,
            name="Risque(%)",
            boxpoints='all',
            marker_color=Colors.PRIMARY_LIGHT,
            line_color=Colors.PRIMARY
        ),
        row=1,
        col=2
    )
    fig.add_trace(
        go.Box(
            y=cvar,
            name="CVaR(%)",
            boxpoints='all',
            marker_color=Colors.PRIMARY_LIGHT,
            line_color=Colors.PRIMARY
        ),
        row=2,
        col=1
    )
    fig.add_trace(
        go.Box(
            y=mdd,
            name="Drawdown Max(%)",
            boxpoints='all',
            marker_color=Colors.PRIMARY_LIGHT,
            line_color=Colors.PRIMARY
        ),
        row=2,
        col=2
    )
    fig.update_layout(template='plotly_white', width=700, height=900, showlegend=False, title="Distributions Rendement/Risque")
    fig.show()


def plot_portfolio_histograms(portfolio):
    """
    Affiche les histogrammes de rendement/risque.
    
    Args:
        portfolio: objet Portfolio avec simulation_results
    """
    if portfolio.simulation_results is None:
        print("❌ Pas de résultats de simulation. Exécutez run_simulation() d'abord.")
        return
    
    pr = [v['portfolio-total-return'] * 100 for v in portfolio.simulation_results.values()]
    pv = [v['portfolio-std'] * 100 for v in portfolio.simulation_results.values()]
    
    hover_temp_return = "<b>Plage CAGR:</b>%{x}<br><b>Fréquence:</b>%{y:,.0f}<extra></extra>"
    hover_temp_risk = "<b>Plage Volatilité:</b>%{x}<br><b>Fréquence:</b>%{y:,.0f}<extra></extra>"

    fig = make_subplots(rows=1, cols=2, subplot_titles=['CAGR (%)', 'Volatilité Annualisée (%)'], shared_xaxes=True, shared_yaxes=True)
    fig.add_trace(go.Histogram(x=pr, name='CAGR', marker_color=Colors.PRIMARY_DARK, hovertemplate=hover_temp_return), row=1, col=1)
    fig.add_trace(go.Histogram(x=pv, name='Volatilité', marker_color=Colors.PRIMARY_ALPHA, hovertemplate=hover_temp_risk), row=1, col=2)
    fig.update_layout(template='plotly_white', title_text='Histogrammes', width=700, height=500)
    fig.show()


def plot_portfolio_simulations(portfolio):
    """
    Affiche les chemins de simulation du portefeuille.
    
    Args:
        portfolio: objet Portfolio avec simulation_results, portfolio_value, cagr_mean, mean_volatility
    """
    if portfolio.simulation_results is None:
        print("❌ Pas de résultats de simulation. Exécutez run_simulation() d'abord.")
        return
        
    sims = len(portfolio.simulation_results)
    title = f"Croissance de ${portfolio.portfolio_value:,.0f}: (CAGR:{round(portfolio.cagr_mean*100, 2)}%, Volatilité:{round(portfolio.mean_volatility*100, 2)}%)<br>Nombre de simulations:{sims:,.0f}"

    hover_temp = "<b>Simulation:</b>%{x}<br><b>Valeur du portefeuille:</b>${y:,.0f}<extra></extra>"
    
    fig = go.Figure()
    for sim_id, v in portfolio.simulation_results.items():
        fig.add_trace(
            go.Scatter(
                x=list(range(len(v['portfolio-values']))),
                y=v['portfolio-values'],
                name=f'Simulation:{sim_id + 1}',
                hovertemplate=hover_temp,
                marker=dict(color=Colors.PRIMARY_ALPHA)
            )
        )
    fig.update_layout(
        template='plotly_white',
        title=title,
        yaxis_title='Valeur du portefeuille',
        xaxis_title=FREQUENCY_LABELS[portfolio.simulation_results[0]['frequency']],
        width=700,
        height=500,
        showlegend=False
    )
    fig.show()


def plot_portfolio_expected_values(portfolio):
    """
    Affiche les valeurs attendues avec bandes de confiance.
    
    Args:
        portfolio: objet Portfolio avec expected_values, simulation_results, portfolio_value, sharpe_ratio, cagr_mean
    """
    if portfolio.expected_values is None or portfolio.simulation_results is None:
        print("❌ Pas de résultats de simulation. Exécutez run_simulation() d'abord.")
        return
        
    bands = {
        'sig1': {'upper': {'series': 'sig1-upper', 'name': '1-Sigma (68%)'},
                 'lower': {'series': 'sig1-lower', 'name': '1-Sigma (68%)'},
                 'color': Colors.PRIMARY_ALPHA},
        'sig2': {'upper': {'series': 'sig2-upper', 'name': '2-Sigma (95%)'},
                 'lower': {'series': 'sig2-lower', 'name': '2-Sigma (95%)'},
                 'color': Colors.PRIMARY_LIGHTER},
        'sig3': {'upper': {'series': 'sig3-upper', 'name': '3-Sigma (99%)'},
                 'lower': {'series': 'sig3-lower', 'name': '3-Sigma (99%)'},
                 'color': Colors.PRIMARY_LIGHTEST}
    }
    df = portfolio.expected_values
    hover_temp = "<b>Période:</b>%{x}<br><b>Valeur du portefeuille:</b>${y:,.0f}"
    
    fig = go.Figure()
    
    for _, band_config in bands.items():
        fig.add_trace(
            go.Scatter(
                name=band_config['upper']['name'],
                x=df.index,
                y=df[band_config['upper']['series']],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hovertemplate=hover_temp
            )
        )
        fig.add_trace(
            go.Scatter(
                name=band_config['lower']['name'],
                x=df.index,
                y=df[band_config['lower']['series']],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor=band_config['color'],
                showlegend=True,
                hovertemplate=hover_temp
            )
        )
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['mean'],
            name='moyenne',
            mode='lines',
            fill=None,
            line=dict(width=3, color=Colors.ACCENT),
            hovertemplate=hover_temp
        )
    )
    
    lower = df.at[df.shape[0] - 1, 'sig1-lower']
    upper = df.at[df.shape[0] - 1, 'sig1-upper']
    freq = FREQUENCY_LABELS[portfolio.simulation_results[0]['frequency']]
    periods = len(portfolio.simulation_results[0]['portfolio-values']) - 1
    title_1 = f"Croissance attendue d'un portefeuille de ${portfolio.portfolio_value:,.0f} après {periods} {freq}<br>"
    title_2 = f"Valeur finale: ${lower:,.0f} à ${upper:,.0f} (Confiance 68%)<br>"
    title_3 = f"Sharpe Ratio: {portfolio.sharpe_ratio:.2f} | CAGR: {portfolio.cagr_mean*100:.2f}%"

    fig.update_layout(
        template='plotly_white',
        title=title_1 + title_2 + title_3,
        xaxis_title=freq,
        yaxis_title='Valeur du portefeuille',
        width=700,
        height=500,
        legend=dict(yanchor='top', y=0.97, xanchor='left', x=0.03)
    )
    fig.show()


def plot_efficient_frontier(portfolio):
    """
    Affiche le graphique de la frontière efficiente.
    
    Args:
        portfolio: objet Portfolio avec efficient_frontier, securities
    """
    if portfolio.efficient_frontier is None:
        print("❌ Pas de frontière efficiente. Exécutez build_efficient_frontier() d'abord.")
        return
        
    frontier_df = pd.DataFrame(portfolio.efficient_frontier)
    
    frontier_df['expected-risk'] = frontier_df['expected-risk'] * 100
    frontier_df['cagr_mean'] = frontier_df['cagr_mean'] * 100

    x_min = min(frontier_df['expected-risk']) - 2
    x_max = max(frontier_df['expected-risk']) + 2
    y_min = min(frontier_df['cagr_mean']) - 2
    y_max = max(frontier_df['cagr_mean']) + 2

    weight_label = []
    for weights in portfolio.efficient_frontier['weights']:
        w = [f"{round(w * 100, 2)}%" for w in weights]
        w = "|".join(w)
        weight_label.append(w)
    frontier_df['portfolio-weights'] = weight_label

    secs = "|".join([sec.identifier for sec in portfolio.securities])
    
    title = "Simulation Risque/Rendement du Portefeuille (Student-t)<br>" + secs
    
    fig = px.scatter(
        frontier_df,
        x='expected-risk',
        y='cagr_mean',
        hover_data=['portfolio-weights', 'sharpe-ratio', 'sortino-ratio', 'cvar-ratio'],
        color='sharpe-ratio',
        opacity=0.6,
        color_continuous_scale=px.colors.diverging.Earth
    )
    
    fig.update_traces(marker_size=14)
    fig.update_layout(template='plotly_white', width=700, height=500, title=title, legend_title='Poids du Portefeuille', hovermode="y")
    fig.update_xaxes(range=[x_min, x_max], title='Risque (%)')
    fig.update_yaxes(range=[y_min, y_max], title='Rendement (%)')
    fig.show()
