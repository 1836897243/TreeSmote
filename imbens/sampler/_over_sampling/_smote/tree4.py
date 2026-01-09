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
class TreeSMOTE4(BaseSMOTE):
    """Tree-based SMOTE over-sampling (完全向量化 + 全叶拼接版)"""
    
    @_deprecate_positional_args
    def __init__(
        self,
        *,
        sampling_strategy="auto",
        random_state=None,
        k_neighbors=5,
        n_jobs=None,
        dt_max_depth=None,
        over_sampling_ratio=1.0,
    ):
        super().__init__(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=k_neighbors,
            n_jobs=n_jobs,
        )
        self.dt_max_depth = dt_max_depth
        self.over_sampling_ratio = over_sampling_ratio
        self.tree_ = DecisionTreeClassifier(max_depth=dt_max_depth, random_state=self.random_state)

    def _validate_estimator(self):
        super()._validate_estimator()
        if self.tree_ is None:
            self.tree_ = DecisionTreeClassifier(random_state=self.random_state, max_depth=self.dt_max_depth)

    def _fit_resample(self, X, y, sample_weight=None):
        rng = np.random.default_rng(self.random_state)
        self._validate_estimator()

        # 拟合决策树
        self.tree_.fit(X, y)
        leaf_ids = self.tree_.apply(X)

        X_res, y_res = [X], [y]
        classes, counts = np.unique(y, return_counts=True)
        majority = counts.max()

        # 计算每个类需要生成的样本数
        if self.sampling_strategy == "auto":
            sampling = {c: majority - cnt for c, cnt in zip(classes, counts) if cnt < majority}
        elif self.sampling_strategy == "ratio":
            if self.over_sampling_ratio <= 1.0:
                raise ValueError("over_sampling_ratio must be > 1.0 when sampling_strategy='ratio'")
            sampling = {
                c: math.ceil(cnt * self.over_sampling_ratio) - cnt
                for c, cnt in zip(classes, counts)
            }
        else:
            sampling = self.sampling_strategy_

        # 对每个少数类
        for cls, n_gen in sampling.items():
            if n_gen <= 0:
                continue

            cls_idx = np.flatnonzero(y == cls)
            if len(cls_idx) < 2:
                continue  # 样本太少无法生成

            # 构建 leaf_knn_cache 并剔除不合格 leaf
            leaf_knn_cache = []
            cls_leaf_ids = np.unique(leaf_ids[cls_idx])
            for leaf in cls_leaf_ids:
                leaf_samples = cls_idx[leaf_ids[cls_idx] == leaf]
                if len(leaf_samples) < self.k_neighbors + 1:
                    continue  # 样本太少无法构建邻居
                X_leaf = X[leaf_samples]
                knn = NearestNeighbors(n_neighbors=self.k_neighbors, n_jobs=self.n_jobs)
                knn.fit(X_leaf)
                neigh = knn.kneighbors(X_leaf, return_distance=False)
                neigh = neigh[:, 1:] if neigh.shape[1] > 1 else neigh
                leaf_knn_cache.append((leaf_samples, X_leaf, neigh))

            if not leaf_knn_cache:
                continue

            # 合并所有 leaf 样本和邻居到全局矩阵
            X_all = np.vstack([X_leaf for _, X_leaf, _ in leaf_knn_cache])
            neigh_all = []
            offset = 0
            for _, X_leaf, neigh in leaf_knn_cache:
                neigh_all.append(neigh + offset)
                offset += len(X_leaf)
            neigh_all = np.vstack(neigh_all)

            # 向量化生成 SMOTE 样本
            rows = rng.integers(0, X_all.shape[0], size=n_gen)
            cols = rng.integers(0, neigh_all.shape[1], size=n_gen)
            steps = rng.random(n_gen)

            X_new = X_all[rows] + steps[:, None] * (X_all[neigh_all[rows, cols]] - X_all[rows])
            y_new = np.full(n_gen, cls)

            X_res.append(X_new)
            y_res.append(y_new)

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

    smote = TreeSMOTE4(
        random_state=42,
        density_exponent="auto",
        cluster_balance_threshold=0.8,
    )
    smote.fit_resample(X, y)

# %%
