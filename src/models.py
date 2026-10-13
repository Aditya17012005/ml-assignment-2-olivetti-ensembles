"""Bagging and discrete multiclass SAMME, implemented without sklearn ensembles.

All ties use the lowest class label. Bagging class vectors are hard-vote
fractions; trees use leaf probabilities; SAMME uses normalized weighted votes.
"""
import time
import numpy as np
from sklearn.tree import DecisionTreeClassifier


def aligned_proba(tree, X, n_classes):
    out = np.zeros((len(X), n_classes))
    out[:, tree.classes_.astype(int)] = tree.predict_proba(X)
    return out


def vote_proba(predictions, n_classes, weights=None):
    predictions = np.asarray(predictions, dtype=int)
    if not len(predictions):
        raise ValueError("At least one accepted learner is required")
    if weights is None:
        weights = np.ones(len(predictions))
    out = np.zeros((predictions.shape[1], n_classes))
    for pred, weight in zip(predictions, weights):
        out[np.arange(len(pred)), pred] += weight
    return out / np.sum(weights)


def samme_update(weights, incorrect, n_classes):
    """Same update used by toy example and full experiments.

    A learner at or worse than random is rejected. A perfect learner is
    assigned the finite alpha corresponding to machine epsilon, then fitting
    stops; weights remain unchanged. No invalid learner enters the ensemble.
    """
    weights = np.asarray(weights, dtype=float)
    incorrect = np.asarray(incorrect, dtype=bool)
    error = float(weights @ incorrect)
    if error >= 1 - 1 / n_classes - 1e-14:
        return error, None, weights.copy(), "not_better_than_random"
    clipped = max(error, np.finfo(float).eps)
    alpha = float(np.log((1 - clipped) / clipped) + np.log(n_classes - 1))
    if error <= np.finfo(float).eps:
        return error, alpha, weights.copy(), "perfect"
    log_w = np.log(weights) + alpha * incorrect
    log_w -= np.max(log_w)
    updated = np.exp(log_w)
    updated /= updated.sum()
    return error, alpha, updated, "accepted"


class Bagging:
    def __init__(self, n_estimators, max_depth, min_samples_leaf, seed, n_classes=40, compress_duplicates=False):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.seed = seed
        self.n_classes = n_classes
        self.compress_duplicates = compress_duplicates

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        self.trees, self.samples, self.times = [], [], []
        if self.compress_duplicates:
            if self.min_samples_leaf != 1:
                raise ValueError('Count compression is exact here only for minimum leaf size 1')
            # Group identical (pixels, label) rows, never columns/features.
            # Drawing n indices still creates an ordinary size-n bootstrap.
            # Integer multiplicities encode exactly that same multiset.
            grouped, inverse = np.unique(np.column_stack([X,y]),axis=0,return_inverse=True)
            unique_X, unique_y = grouped[:,:-1].astype(X.dtype), grouped[:,-1].astype(y.dtype)
        start = time.perf_counter()
        for _ in range(self.n_estimators):
            sample = rng.integers(0, len(y), len(y))
            tree = DecisionTreeClassifier(max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=int(rng.integers(0, 2**31 - 1)))
            if self.compress_duplicates:
                counts = np.bincount(inverse[sample],minlength=len(grouped))
                tree.fit(unique_X,unique_y,sample_weight=counts)
            else:
                tree.fit(X[sample], y[sample])
            self.trees.append(tree)
            self.samples.append(sample)
            self.times.append(time.perf_counter() - start)
        return self

    def predictions(self, X, B=None):
        return np.array([t.predict(X) for t in self.trees[:B]], dtype=int)

    def predict_proba(self, X, B=None):
        return vote_proba(self.predictions(X, B), self.n_classes)

    def oob(self, X, y, B=None):
        counts = np.zeros((len(y), self.n_classes))
        totals = np.zeros(len(y), dtype=int)
        for tree, sample in zip(self.trees[:B], self.samples[:B]):
            mask = np.ones(len(y), dtype=bool)
            mask[np.unique(sample)] = False
            ix = np.flatnonzero(mask)
            counts[ix, tree.predict(X[ix]).astype(int)] += 1
            totals[ix] += 1
        covered = totals > 0
        return float(np.mean(counts[covered].argmax(1) == y[covered])), float(covered.mean())


class SAMME:
    def __init__(self, n_estimators, max_depth, seed, n_classes=40):
        self.n_estimators, self.max_depth = n_estimators, max_depth
        self.seed, self.n_classes = seed, n_classes

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        w = np.full(len(y), 1 / len(y))
        self.trees, self.alphas, self.history, self.times = [], [], [], []
        self.sample_weights, self.train_predictions = [w.copy()], []
        start = time.perf_counter()
        self.stop_reason = "round_limit"
        for t in range(self.n_estimators):
            tree = DecisionTreeClassifier(max_depth=self.max_depth,
                random_state=int(rng.integers(0, 2**31 - 1)))
            tree.fit(X, y, sample_weight=w)
            pred = tree.predict(X).astype(int)
            error, alpha, updated, status = samme_update(w, pred != y, self.n_classes)
            if alpha is None:
                self.stop_reason = status
                if not self.trees:
                    raise ValueError("First SAMME learner is not better than random")
                break
            self.trees.append(tree)
            self.alphas.append(alpha)
            self.train_predictions.append(pred)
            self.history.append(dict(round=t + 1, error=error, alpha=alpha,
                max_weight_before=float(w.max()), top10_weight_before=float(np.sort(w)[-10:].sum()),
                max_weight_after=float(updated.max()), top10_weight_after=float(np.sort(updated)[-10:].sum())))
            self.times.append(time.perf_counter() - start)
            w = updated
            self.sample_weights.append(w.copy())
            if status == "perfect":
                self.stop_reason = status
                break
        self.alphas = np.asarray(self.alphas)
        self.sample_weights = np.asarray(self.sample_weights)
        self.train_predictions = np.asarray(self.train_predictions)
        return self

    def predictions(self, X, B=None):
        return np.array([t.predict(X) for t in self.trees[:B]], dtype=int)

    def predict_proba(self, X, B=None):
        return vote_proba(self.predictions(X, B), self.n_classes, self.alphas[:B])


def disagreement(predictions):
    """Mean unordered-pair disagreement on the explicitly supplied dataset."""
    B, N = predictions.shape
    if B < 2:
        return None
    counts = np.apply_along_axis(lambda x: np.bincount(x, minlength=40), 0, predictions)
    agreeing = (counts * (counts - 1)).sum(axis=0)
    return float(np.mean(1 - agreeing / (B * (B - 1))))


def margin(probabilities):
    ordered = np.sort(probabilities, axis=1)
    return float(np.mean(ordered[:, -1] - ordered[:, -2]))


def leaf_stats(tree, X):
    _, occupancy = np.unique(tree.apply(X), return_counts=True)
    return dict(leaves=int(len(occupancy)), mean_samples_per_leaf=float(occupancy.mean()),
        singleton_leaf_fraction=float(np.mean(occupancy == 1)), actual_depth=int(tree.get_depth()))
