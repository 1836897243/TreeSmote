"""SMOTE variant employing some tree before the generation."""
# Adapted from imbalanced-learn

# Authors: Guillaume Lemaitre
#          Fernando Nogueira
#          Christos Aridas
#          Zhining Liu <zhining.liu@outlook.com>
# License: MIT

# %%
LOCAL_DEBUG = False

if not LOCAL_DEBUG:
    from .base import BaseSMOTE
    from ..base import BaseOverSampler
    from ....utils._docstring import _n_jobs_docstring, Substitution
    from ....utils._docstring import _random_state_docstring
    from ....utils._validation import _deprecate_positional_args
else:           # pragma: no cover
    import sys  # For local test
    sys.path.append("../../..")
    from sampler._over_sampling._smote.base import BaseSMOTE
    from sampler._over_sampling.base import BaseOverSampler
    from utils._docstring import _n_jobs_docstring, Substitution
    from utils._docstring import _random_state_docstring
    from utils._validation import _deprecate_positional_args

import math

import numpy as np
from scipy import sparse

import warnings
warnings.filterwarnings('ignore')

from sklearn.base import clone
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import pairwise_distances
from sklearn.utils import _safe_indexing
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier, _tree

@Substitution(
    sampling_strategy=BaseOverSampler._sampling_strategy_docstring,
    n_jobs=_n_jobs_docstring,
    random_state=_random_state_docstring,
)
class TreeSMOTE(BaseSMOTE):
    """Tree-based SMOTE over-sampling."""
    @_deprecate_positional_args
    def __init__(
        self,
        *,
        sampling_strategy="auto",
        random_state=None,
        k_neighbors=5,
        n_jobs=None,
        decision_tree_estimator=None,
        max_leaf_neighbors=3,
    ):
        super().__init__(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=k_neighbors,
            n_jobs=n_jobs,
        )
        self.decision_tree_estimator = decision_tree_estimator
        self.max_leaf_neighbors = max_leaf_neighbors

    def _validate_estimator(self):
        super()._validate_estimator()
        if self.decision_tree_estimator is None:
            self.tree_ = DecisionTreeClassifier(random_state=self.random_state)
        else:
            self.tree_ = clone(self.decision_tree_estimator)

    def _fit_resample(self, X, y, sample_weight=None):
        rng = np.random.default_rng(self.random_state)
        self._validate_estimator()

        # 1️⃣ 拟合树
        self.tree_.fit(X, y)
        leaf_ids = self.tree_.apply(X)

        # 2️⃣ leaf → sample indices
        leaf_to_idx = {}
        for i, leaf in enumerate(leaf_ids):
            leaf_to_idx.setdefault(leaf, []).append(i)

        X_res, y_res = [X], [y]

        classes, counts = np.unique(y, return_counts=True)
        majority = counts.max()
        if self.sampling_strategy == "auto":
            sampling = {c: majority - cnt for c, cnt in zip(classes, counts) if cnt < majority}
        else:
            sampling = self.sampling_strategy_
        # 3️⃣ 对每个少数类
        for cls, n_gen in sampling.items():
            if n_gen <= 0:
                continue

            cls_idx = np.flatnonzero(y == cls)
            rng.shuffle(cls_idx)

            # 4️⃣ 按 leaf 分组 minority
            leaf_groups = {}
            for idx in cls_idx:
                leaf = leaf_ids[idx]
                leaf_groups.setdefault(leaf, []).append(idx)

            for leaf, idxs in leaf_groups.items():
                if n_gen <= 0:
                    break

                idxs = np.array(idxs)
                if len(idxs) < 2:
                    continue

                X_leaf = X[idxs]

                # 5️⃣ KNN（只 fit 一次）
                k = min(self.k_neighbors + 1, len(X_leaf))
                knn = NearestNeighbors(n_neighbors=k, n_jobs=self.n_jobs)
                knn.fit(X_leaf)
                neigh = knn.kneighbors(X_leaf, return_distance=False)[:, 1:]

                # 6️⃣ 这一个 leaf 要生成多少样本
                n_leaf_gen = min(n_gen, len(idxs) * 2)

                rows = rng.integers(0, len(idxs), size=n_leaf_gen)
                cols = rng.integers(0, neigh.shape[1], size=n_leaf_gen)
                steps = rng.random(n_leaf_gen)

                x_i = X_leaf[rows]
                x_nn = X_leaf[neigh[rows, cols]]
                X_new = x_i + steps[:, None] * (x_nn - x_i)

                X_res.append(X_new)
                y_res.append(np.full(n_leaf_gen, cls))

                n_gen -= n_leaf_gen

        X_res = np.vstack(X_res)
        y_res = np.hstack(y_res)
        if sample_weight is not None:
            w_new = np.full(len(y_res) - len(y), np.mean(sample_weight))
            w = np.hstack([sample_weight, w_new])
            return X_res, y_res, w
        
        return X_res, y_res
        


# %%

if __name__ == "__main__":  # pragma: no cover
    # rng = np.random.RandomState(42)
    # X = rng.randn(30, 2)
    # y = np.array([1] * 20 + [0] * 10)
    # smote = KMeansSMOTE(random_state=42, kmeans_estimator=30, k_neighbors=2)
    # smote.fit_resample(X, y)

    X = np.array(
    [
        [0.11622591, -0.0317206],
        [0.77481731, 0.60935141],
        [1.25192108, -0.22367336],
        [0.53366841, -0.30312976],
        [1.52091956, -0.49283504],
        [-0.28162401, -2.10400981],
        [0.83680821, 1.72827342],
        [0.3084254, 0.33299982],
        [0.70472253, -0.73309052],
        [0.28893132, -0.38761769],
        [1.15514042, 0.0129463],
        [0.88407872, 0.35454207],
        [1.31301027, -0.92648734],
        [-1.11515198, -0.93689695],
        [-0.18410027, -0.45194484],
        [0.9281014, 0.53085498],
        [-0.14374509, 0.27370049],
        [-0.41635887, -0.38299653],
        [0.08711622, 0.93259929],
        [1.70580611, -0.11219234],
    ])
    y = np.array([0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0])

    smote = TreeSMOTE(
        random_state=42,
        density_exponent="auto",
        cluster_balance_threshold=0.8,
    )
    smote.fit_resample(X, y)

# %%
