"""Contrôles des risques principaux : fuite de données et pipeline réel."""

import unittest

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit

from src.data import NUMERIC, RAW, REG_FEATURES, load_classification, load_regression
from src.experiments import make_pipeline
from src.models import TorchRegressor


@unittest.skipUnless((RAW / "prices-split-adjusted.csv").exists(),
                     "Télécharger les données avec python -m src.data")
class DataContracts(unittest.TestCase):
    def test_temporal_target_and_purged_boundaries(self):
        raw, prepared, parts = load_regression()
        apple = raw[raw.symbol.eq("AAPL")].sort_values("date").set_index("date")
        np.testing.assert_allclose(
            prepared.next_close, apple.loc[prepared.target_date, "close"])
        self.assertLess(parts["train"].target_date.max(), parts["validation"].date.min())
        self.assertLess(parts["validation"].target_date.max(), parts["test"].date.min())
        for train, valid in TimeSeriesSplit(n_splits=3, gap=1).split(parts["train"]):
            self.assertLess(parts["train"].iloc[train].target_date.max(),
                            parts["train"].iloc[valid].date.min())

    def test_classification_splits_and_augmentation(self):
        _, _, parts = load_classification()
        for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
            self.assertFalse(set(parts[left].UDI) & set(parts[right].UDI))
        self.assertEqual(parts["train"]["Failure Type"].nunique(), 6)
        # Un petit échantillon réel, équilibré ici pour limiter le temps du test.
        small = parts["train"].groupby("Failure Type").head(12)
        pipe = make_pipeline("classification").set_params(net__epochs=2)
        columns = NUMERIC + ["Type"]
        pipe.fit(small[columns], small["Failure Type"])
        prepared = pipe.named_steps["prepare"].transform(small[columns])
        augmented, labels = clone(pipe.named_steps["balance"]).fit_resample(
            prepared, small["Failure Type"])
        self.assertEqual(len(set(np.unique(labels, return_counts=True)[1])), 1)
        self.assertTrue(set(augmented[:, 5]).issubset({0., 1., 2.}))
        test = parts["test"][columns].iloc[:20]
        probabilities = pipe.predict_proba(test)
        self.assertEqual(probabilities.shape, (20, 6))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1., atol=1e-6)
        names = pipe.named_steps["prepare"].get_feature_names_out()
        self.assertFalse(any("Target" in name or "UDI" in name or "Product ID" in name for name in names))

    def test_regression_fit_clone_and_target_scaling(self):
        _, _, parts = load_regression()
        train = parts["train"].iloc[:100]
        pipe = clone(make_pipeline("regression")).set_params(net__epochs=2)
        pipe.fit(train[REG_FEATURES], train.next_close)
        self.assertAlmostEqual(pipe.named_steps["net"].y_mean_, train.next_close.mean())
        prediction = pipe.predict(parts["test"][REG_FEATURES].iloc[:10])
        self.assertEqual(prediction.shape, (10,))
        self.assertTrue(np.isfinite(prediction).all())

    def test_posthoc_test_curve_cannot_change_model(self):
        X = np.arange(60, dtype=np.float32).reshape(20, 3) / 60
        y = X.sum(axis=1)
        model = TorchRegressor(epochs=2).fit(X[:12], y[:12],
                                            validation_data=(X[12:16], y[12:16]),
                                            record_states=True)
        before = model.predict(X[16:]).copy()
        epoch = model.selected_epoch_
        model._measure(X[16:], y[16:])
        np.testing.assert_array_equal(before, model.predict(X[16:]))
        self.assertEqual(epoch, model.selected_epoch_)
        self.assertFalse(any("test" in key for row in model.history_ for key in row))


if __name__ == "__main__":
    unittest.main()
