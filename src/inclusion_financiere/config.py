"""Chemins, graine et coûts métier : définis une seule fois pour tout le projet."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "inclusion-financiere-en-afrique.csv"
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "reports" / "figures"
DECISIONS = PROCESSED / "decisions.json"  # hyperparamètres et seuil retenus (étapes 7 à 9)

TARGET = "bank_account"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Coûts métier KaziPay (hypothèses pédagogiques)
COST_FP = 0.5  # contacter une personne sans compte (USD)
COST_FN = 5.0  # oublier une personne bancarisée (USD)
