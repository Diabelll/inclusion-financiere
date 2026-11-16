"""Chargement, nettoyage et découpage train/test."""
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from .config import RANDOM_STATE, RAW, TARGET, TEST_SIZE

CAT_COLS = [
    "country", "location_type", "cellphone_access", "gender_of_respondent",
    "relationship_with_head", "marital_status", "education_level", "job_type",
]
NUM_COLS = ["household_size", "age_of_respondent"]


def load_raw(path=RAW) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Sépare X et y ; retire les colonnes qui ne doivent pas servir de variables."""
    y = (df[TARGET] == "Yes").astype(int)
    X = df.drop(columns=[TARGET, "uniqueid", "year"])
    return X, y


def split(X: pd.DataFrame, y: pd.Series):
    """Découpage train/test stratifié, identique dans tous les notebooks."""
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y,
                            random_state=RANDOM_STATE)


def make_cv(n_splits: int = 5) -> StratifiedKFold:
    """Le seul objet de validation croisée du projet (mêmes plis pour tous les modèles)."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
