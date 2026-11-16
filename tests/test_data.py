import numpy as np

from inclusion_financiere.data import clean, load_raw, split
from inclusion_financiere.evaluate import cout_total, expected_calibration_error
from inclusion_financiere.features import AddFeatures


def test_clean_retire_les_colonnes_interdites():
    X, y = clean(load_raw())
    assert {"uniqueid", "year", "bank_account"}.isdisjoint(X.columns)
    assert set(y.unique()) == {0, 1}


def test_add_features_garde_toutes_les_lignes():
    X, _ = clean(load_raw())
    Xf = AddFeatures().fit_transform(X)
    assert len(Xf) == len(X)
    assert Xf["education_rank"].between(-1, 4).all()


def test_split_point_de_controle():
    X, y = clean(load_raw())
    X_train, X_test, y_train, y_test = split(X, y)
    assert (len(X_train), len(X_test)) == (18819, 4705)
    assert abs(y_train.mean() - y_test.mean()) < 0.001


def test_ece_nulle_si_parfaitement_calibre():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.5, 0.5, 0.5, 0.5])
    assert expected_calibration_error(y, p) == 0


def test_cout_total():
    # 1 faux positif (0,5 USD) + 1 faux négatif (5 USD) = 5,5 USD
    assert cout_total([0, 1, 1, 0], [1, 0, 1, 0]) == 5.5
