import unittest
import numpy as np
from src.models import SAMME, Bagging, samme_update, vote_proba, disagreement


class ModelChecks(unittest.TestCase):
    def test_hand_calculated_samme_round(self):
        w = np.full(6, 1/6)
        wrong = np.array([0, 0, 1, 0, 0, 1], dtype=bool)
        error, alpha, new, status = samme_update(w, wrong, 3)
        self.assertAlmostEqual(error, 1/3)
        self.assertAlmostEqual(alpha, np.log(4))
        np.testing.assert_allclose(new, [1/12, 1/12, 1/3, 1/12, 1/12, 1/3])
        self.assertEqual(status, 'accepted')

    def test_samme_boundary_and_perfect(self):
        w = np.full(6, 1/6)
        self.assertIsNone(samme_update(w, [1, 1, 1, 1, 0, 0], 3)[1])
        self.assertIsNone(samme_update(w, np.ones(6), 3)[1])
        result = samme_update(w, np.zeros(6), 3)
        self.assertTrue(np.isfinite(result[1]))
        self.assertEqual(result[3], 'perfect')

    def test_hard_votes_and_disagreement(self):
        pred = np.array([[0, 1], [1, 1], [0, 2]])
        np.testing.assert_allclose(vote_proba(pred, 3), [[2/3,1/3,0],[0,2/3,1/3]])
        self.assertAlmostEqual(disagreement(pred), 2/3)
        self.assertIsNone(disagreement(pred[:1]))

    def test_fit_and_oob_reproducibility(self):
        X = np.arange(36).reshape(18, 2)
        y = np.repeat(np.arange(3), 6)
        a = Bagging(12, 2, 1, 10, 3).fit(X, y)
        b = Bagging(12, 2, 1, 10, 3).fit(X, y)
        np.testing.assert_array_equal(a.samples, b.samples)
        np.testing.assert_array_equal(a.predict_proba(X), b.predict_proba(X))
        score, coverage = a.oob(X, y)
        self.assertGreater(coverage, .8)
        self.assertGreater(score, .7)
        boost = SAMME(10, 1, 11, 3).fit(X, y)
        np.testing.assert_allclose(boost.sample_weights.sum(axis=1), 1)
        np.testing.assert_allclose(boost.predict_proba(X).sum(axis=1), 1)

    def test_count_compression_matches_literal_bootstrap(self):
        rng=np.random.default_rng(42)
        X=rng.normal(size=(45,12)).astype(np.float32)
        y=np.repeat(np.arange(3),15)
        ix=rng.integers(0,45,75)
        for depth in [2,16,None]:
            literal=Bagging(10,depth,1,123,3).fit(X[ix],y[ix])
            compressed=Bagging(10,depth,1,123,3,compress_duplicates=True).fit(X[ix],y[ix])
            np.testing.assert_array_equal(literal.samples,compressed.samples)
            np.testing.assert_array_equal(literal.predict_proba(X),compressed.predict_proba(X))
            for a,b in zip(literal.trees,compressed.trees):
                np.testing.assert_array_equal(a.tree_.feature,b.tree_.feature)
                np.testing.assert_array_equal(a.tree_.threshold,b.tree_.threshold)


if __name__ == '__main__':
    unittest.main()
