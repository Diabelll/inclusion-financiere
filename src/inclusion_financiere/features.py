"""Feature engineering et préprocesseurs."""
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CAT_COLS, NUM_COLS

EDU_ORDER = {"No formal education": 0, "Primary education": 1, "Secondary education": 2,
             "Vocational/Specialised training": 3, "Tertiary education": 4}
FORMAL_JOBS = {"Formally employed Private", "Formally employed Government"}

# Variables du guide (point de départ)
CAT_GUIDE = ["age_group"]
NUM_GUIDE = ["education_rank", "is_head", "formal_job", "urban_phone"]
# Variables ajoutées « de mon cru » (Q3.5)
CAT_OWN = ["household_group"]
NUM_OWN = ["is_alone"]

CAT_FE = CAT_COLS + CAT_GUIDE + CAT_OWN
NUM_FE = NUM_COLS + NUM_GUIDE + NUM_OWN


class AddFeatures(BaseEstimator, TransformerMixin):
    """Feature engineering sans apprentissage : aucune statistique tirée des données."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Variables du guide
        X["education_rank"] = X["education_level"].map(EDU_ORDER).fillna(-1)
        X["is_head"] = (X["relationship_with_head"] == "Head of Household").astype(int)
        X["formal_job"] = X["job_type"].isin(FORMAL_JOBS).astype(int)
        X["urban_phone"] = ((X["location_type"] == "Urban")
                            & (X["cellphone_access"] == "Yes")).astype(int)
        X["age_group"] = pd.cut(X["age_of_respondent"], [15, 25, 35, 50, 65, 120],
                                labels=["16-25", "26-35", "36-50", "51-65", "66+"]).astype(str)
        # Variables ajoutées (Q3.5)
        X["is_alone"] = (X["household_size"] == 1).astype(int)
        X["household_group"] = pd.cut(X["household_size"], [-1, 2, 5, 9, 100],
                                      labels=["0-2", "3-5", "6-9", "10+"]).astype(str)
        return X


def onehot_preprocessor(cat_cols, num_cols):
    """One-hot (modalités rares regroupées) + standardisation. Sortie dense (requise par HistGB)."""
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20),
         cat_cols),
        ("num", StandardScaler(), num_cols),
    ], sparse_threshold=0)


def pipe(model, own_features: bool = True):
    """Pipeline complet : feature engineering -> encodage -> modèle.

    own_features=False retire les variables ajoutées (Q3.5) pour mesurer leur effet.
    """
    cat = CAT_FE if own_features else CAT_COLS + CAT_GUIDE
    num = NUM_FE if own_features else NUM_COLS + NUM_GUIDE
    return make_pipeline(AddFeatures(), onehot_preprocessor(cat, num), model)
