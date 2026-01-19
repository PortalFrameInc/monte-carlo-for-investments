"""
Ce module contient les fonctions utilitaires utilisées dans tout le projet.
"""

import os
from pathlib import Path

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

def get_api_key(path: str = "keys/alphavantage.txt") -> str:
    """
    Lit la clé API Alpha Vantage depuis un fichier.
    
    Args:
        path: chemin vers le fichier contenant la clé API
        
    Returns:
        La clé API Alpha Vantage
        
    Raises:
        FileNotFoundError: si le fichier n'existe pas
    """
    key_path = Path(path)
    if not key_path.exists():
        raise FileNotFoundError(
            f"❌ Fichier de clé API non trouvé: {path}\n"
            f"Créez le fichier et ajoutez votre clé Alpha Vantage."
        )
    
    with open(key_path) as f:
        key = f.read().strip()
    
    if not key:
        raise ValueError(f"❌ Le fichier {path} est vide. Ajoutez votre clé API.")
    
    return key
