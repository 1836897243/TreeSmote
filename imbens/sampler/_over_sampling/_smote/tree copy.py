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
        k_neighbors=2,
        n_jobs=None,
        decision_tree_estimator = None
    ):
        super().__init__(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=k_neighbors,
            n_jobs=n_jobs,
        )
        self.decision_tree_estimator = decision_tree_estimator
        self.sampling_strategy_ = sampling_strategy


    def _validate_estimator(self):
        super()._validate_estimator()
        if self.decision_tree_estimator is None:
            self.decision_tree_estimator = DecisionTreeClassifier()
        else:
            self.decision_tree_estimator = clone(self.decision_tree_estimator)


    def _fit_resample(self, X, y, sample_weight=None):
        rng = np.random.default_rng(self.random_state)

        # 拟合决策树
        self._validate_estimator()
        tree = self.decision_tree_estimator
        tree.fit(X, y)

        leaf_nodes = tree.apply(X)
        children_left = tree.tree_.children_left
        children_right = tree.tree_.children_right
        is_leaf = (children_left == _tree.TREE_LEAF)
        leaf_ids = np.where(is_leaf)[0]

        leaf_to_samples = {leaf: np.flatnonzero(leaf_nodes == leaf) for leaf in leaf_ids}

        X_resampled = [X.copy()]
        y_resampled = [y.copy()]

        classes, counts = np.unique(y, return_counts=True)
        majority_count = max(counts)
        minority_classes = classes[counts < majority_count]
        if self.sampling_strategy_ == 'auto':
            self.sampling_strategy_ = {cls: majority_count for cls in minority_classes}
        def get_neighbors(leaf, cls_idx, visited=None, depth=0, max_depth=5):
            """递归搜索叶子节点邻居: 叶子→兄弟→堂兄顺序"""
            if visited is None:
                visited = set()
            if leaf in visited or depth > max_depth:
                return np.array([], dtype=int)
            visited.add(leaf)

            candidates = np.intersect1d(leaf_to_samples.get(leaf, []), cls_idx)
            if len(candidates) >= self.k_neighbors:
                return candidates

            # 搜索兄弟叶
            parents = np.where((children_left == leaf) | (children_right == leaf))[0]
            for p in parents:
                sibling = children_left[p] if children_left[p] != leaf else children_right[p]
                candidates = np.concatenate([
                    candidates,
                    get_neighbors(sibling, cls_idx, visited, depth+1, max_depth)
                ])
            return candidates
        for cls in self.sampling_strategy_.keys():
            cls_idx = np.flatnonzero(y == cls)
            n_gen = self.sampling_strategy_[cls]
            try_cnt = n_gen * 5
            while n_gen > 0:
                for leaf in leaf_ids:
                    leaf_cls_idx = np.intersect1d(leaf_to_samples[leaf], cls_idx)
                    if len(leaf_cls_idx) == 0:
                        continue

                    neighbors_idx = get_neighbors(leaf, cls_idx)
                    neighbors_idx = neighbors_idx.astype(np.int64)
                    if len(neighbors_idx) <= 1:
                        continue
                    nbrs = NearestNeighbors(n_neighbors=min(self.k_neighbors, len(neighbors_idx))).fit(X[neighbors_idx])
                    nn_idx = nbrs.kneighbors(X[leaf_cls_idx], return_distance=False)[:, 1:]

                    for i, idx in enumerate(leaf_cls_idx):
                        if n_gen <= 0:
                            break
                        chosen = rng.choice(nn_idx[i])
                        X_resampled.append((X[idx] + rng.random() * (X[neighbors_idx[chosen]] - X[idx])).reshape(1, -1))
                        y_resampled.append(np.array([cls]))
                        n_gen -= 1
                        try_cnt -= 1
                    if n_gen <= 0:
                        break
                    if try_cnt <= 0:
                        assert False, "Cannot generate enough samples, please try to increase k_neighbors."
        X_resampled = np.vstack(X_resampled)
        y_resampled = np.hstack(y_resampled)
        
        # If given sample_weight
        if sample_weight is not None:
            # sample_weight is already validated in self.fit_resample()
            sample_weight_new = \
                np.empty(y_resampled.shape[0] - y.shape[0], dtype=np.float64)
            sample_weight_new[:] = np.mean(sample_weight)
            sample_weight_resampled = np.hstack([sample_weight, sample_weight_new]).reshape(-1, 1)
            sample_weight_resampled = \
                np.squeeze(normalize(sample_weight_resampled, axis=0, norm='l1'))
            return X_resampled, y_resampled, sample_weight_resampled
        else: return X_resampled, y_resampled
        


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
