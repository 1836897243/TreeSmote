"""SMOTE variant employing some tree before the generation."""
'''
Iterative Process(迭代式):不要只用一棵树。
第1轮:训练一个分类器(如 XGBoost)
第2轮:找到上一轮分类器预测置信度低(Decision Boundary附近)的那些少数类样本。
第3轮:只针对这些“困难样本”，利用它们所在的树路径(Leaf Path)进行微小的局部扰动生成。
第4轮:把新数据加入，重新训练。
'''
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
class TreeSMOTE6(BaseSMOTE):
    """Iterative Tree-based SMOTE. Each iteration generates a portion of samples until all classes are balanced."""

    @_deprecate_positional_args
    def __init__(
        self,
        *,
        sampling_strategy="auto",
        random_state=None,
        k_neighbors=2,
        n_jobs=None,
        decision_tree_estimator=None,
        max_iter=5
    ):
        super().__init__(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=k_neighbors,
            n_jobs=n_jobs,
        )
        self.decision_tree_estimator = decision_tree_estimator
        self.max_iter = max_iter

    def _validate_estimator(self):
        super()._validate_estimator()
        if self.decision_tree_estimator is None:
            self.decision_tree_estimator = DecisionTreeClassifier()
        else:
            self.decision_tree_estimator = clone(self.decision_tree_estimator)

    def _select_difficult_samples(self, tree, X, y, cls, max_count):
        """Select minority samples with lowest confidence up to max_count"""
        proba = tree.predict_proba(X)
        cls_idx = np.flatnonzero(y == cls)
        if len(cls_idx) == 0:
            return np.array([], dtype=int)
        conf = proba[cls_idx, cls]
        sorted_idx = cls_idx[np.argsort(conf)]
        return sorted_idx[:max_count]

    def _generate_samples_for_indices(self, X, y, indices):
        rng = np.random.default_rng(self.random_state)

        self._validate_estimator()
        tree = self.decision_tree_estimator
        tree.fit(X, y)

        leaf_nodes = tree.apply(X)
        children_left = tree.tree_.children_left
        children_right = tree.tree_.children_right
        is_leaf = (children_left == _tree.TREE_LEAF)
        leaf_ids = np.where(is_leaf)[0]
        leaf_to_samples = {leaf: np.flatnonzero(leaf_nodes == leaf) for leaf in leaf_ids}

        cls = y[int(indices[0])]
        cls_idx = np.flatnonzero(y == cls)

        def get_neighbors(leaf, visited=None, depth=0, max_depth=5):
            if visited is None:
                visited = set()
            if leaf in visited or depth > max_depth:
                return np.array([], dtype=int)
            visited.add(leaf)
            candidates = np.intersect1d(leaf_to_samples.get(leaf, []), cls_idx)
            if len(candidates) >= self.k_neighbors:
                return candidates
            parents = np.where((children_left == leaf) | (children_right == leaf))[0]
            for p in parents:
                sibling = children_left[p] if children_left[p] != leaf else children_right[p]
                candidates = np.concatenate([candidates, get_neighbors(sibling, visited, depth + 1, max_depth)])
            return candidates

        X_new, y_new = [], []
        for idx in np.atleast_1d(indices).astype(int):
            leaf = leaf_nodes[idx]
            nbrs_idx = get_neighbors(leaf)
            nbrs_idx = np.atleast_1d(nbrs_idx).astype(int)
            if len(nbrs_idx) <= 1:
                continue
            chosen = int(rng.choice(nbrs_idx))
            synthetic = X[idx] + rng.random() * (X[chosen] - X[idx])
            X_new.append(synthetic)
            y_new.append(cls)
        if len(X_new) == 0:
            return np.empty((0, X.shape[1])), np.empty((0,), dtype=y.dtype)
        return np.array(X_new), np.array(y_new)

    def _fit_resample(self, X, y, sample_weight=None):
        rng = np.random.default_rng(self.random_state)
        X_aug, y_aug = X.copy(), y.copy()
        classes, counts = np.unique(y_aug, return_counts=True)
        majority_count = counts.max()

        for it in range(self.max_iter):
            classes, counts = np.unique(y_aug, return_counts=True)
            majority_count = counts.max()
            class_gaps = {cls: majority_count - cnt for cls, cnt in zip(classes, counts) if cnt < majority_count}
            if len(class_gaps) == 0:
                break  # all classes balanced

            self._validate_estimator()
            tree = clone(self.decision_tree_estimator)
            tree.fit(X_aug, y_aug)

            for cls, gap in class_gaps.items():
                # 每轮生成数量 = ceil(差距 / 剩余轮数)
                n_generate = max(1, int(np.ceil(gap / (self.max_iter - it))))
                # 每轮最多使用原始样本数
                max_samples = min(n_generate, np.sum(y_aug == cls))
                selected_idx = self._select_difficult_samples(tree, X_aug, y_aug, cls, max_samples)
                if len(selected_idx) == 0:
                    continue
                X_new, y_new = self._generate_samples_for_indices(X_aug, y_aug, selected_idx)
                # 控制生成数量不要超过差距
                if len(X_new) > gap:
                    X_new = X_new[:gap]
                    y_new = y_new[:gap]
                if len(X_new) > 0:
                    X_aug = np.vstack([X_aug, X_new])
                    y_aug = np.hstack([y_aug, y_new])

        return X_aug, y_aug
        


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
