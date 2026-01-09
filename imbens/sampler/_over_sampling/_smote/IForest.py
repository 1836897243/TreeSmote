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

import numpy as np
from sklearn.base import clone
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import normalize
import warnings
warnings.filterwarnings('ignore')


@Substitution(
    sampling_strategy=BaseOverSampler._sampling_strategy_docstring,
    n_jobs=_n_jobs_docstring,
    random_state=_random_state_docstring,
)
class IForestSMOTE(BaseSMOTE):
    """Depth-Weighted SMOTE using Isolation Forest path depth to guide sampling."""

    @_deprecate_positional_args
    def __init__(
        self,
        *,
        sampling_strategy="auto",
        random_state=None,
        k_neighbors=2,
        n_jobs=None,
        decision_tree_estimator=None,
        iforest_estimator=None,
    ):
        super().__init__(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=k_neighbors,
            n_jobs=n_jobs,
        )
        self.decision_tree_estimator = decision_tree_estimator
        self.iforest_estimator = iforest_estimator

    def _validate_estimator(self):
        super()._validate_estimator()

        # 决策树（用于样本分叶）
        if self.decision_tree_estimator is None:
            self.decision_tree_estimator = DecisionTreeClassifier()
        else:
            self.decision_tree_estimator = clone(self.decision_tree_estimator)

        # 孤立森林（用于路径深度）
        if self.iforest_estimator is None:
            self.iforest_estimator = IsolationForest(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
        else:
            self.iforest_estimator = clone(self.iforest_estimator)

    def _compute_depth(self, X_minority):
        """计算 Minority 样本的平均路径深度。"""
        iforest = self.iforest_estimator.fit(X_minority)

        # 逐棵树计算路径长度
        depths = []
        for est in iforest.estimators_:
            node_indicator = est.decision_path(X_minority)
            depth = np.array(node_indicator.sum(axis=1)).flatten()
            depths.append(depth)

        depths = np.mean(np.vstack(depths), axis=0)
        depths = depths - depths.min() + 1e-6
        return depths / depths.sum()

    def _fit_resample(self, X, y, sample_weight=None):
        rng = np.random.default_rng(self.random_state)
        self._validate_estimator()

        # 决策树拟合
        tree = self.decision_tree_estimator
        tree.fit(X, y)

        leaf_nodes = tree.apply(X)
        children_left = tree.tree_.children_left
        children_right = tree.tree_.children_right

        leaf_ids = np.where(children_left == _tree.TREE_LEAF)[0]
        leaf_to_samples = {leaf: np.flatnonzero(leaf_nodes == leaf) for leaf in leaf_ids}

        X_resampled = [X.copy()]
        y_resampled = [y.copy()]

        # 少数类检测
        classes, counts = np.unique(y, return_counts=True)
        majority_count = counts.max()
        minority_classes = classes[counts < majority_count]

        # ★ 深度计算
        for cls in minority_classes:
            cls_idx = np.flatnonzero(y == cls)
            X_minority = X[cls_idx]
            weights = self._compute_depth(X_minority)

            n_gen_total = majority_count - len(cls_idx)

            # Nearest Neighbor
            nbrs = NearestNeighbors(
                n_neighbors=min(self.k_neighbors + 1, len(X_minority))
            ).fit(X_minority)
            nn_idx = nbrs.kneighbors(X_minority, return_distance=False)[:, 1:]

            # 逐样本按深度权重生成
            for i, x_i in enumerate(X_minority):
                n_gen_i = int(np.round(weights[i] * n_gen_total))

                for _ in range(n_gen_i):
                    # neighbor selection
                    neighbor = rng.choice(nn_idx[i])
                    x_nn = X_minority[neighbor]

                    # ★ α 随深度缩放：深度浅 → 变化少
                    alpha = rng.random() * (0.3 + 0.7 * weights[i])

                    x_new = x_i + alpha * (x_nn - x_i)

                    X_resampled.append(x_new.reshape(1, -1))
                    y_resampled.append(np.array([cls]))

        X_resampled = np.vstack(X_resampled)
        y_resampled = np.hstack(y_resampled)

        if sample_weight is not None:
            sample_weight_new = np.empty(
                y_resampled.shape[0] - y.shape[0], dtype=np.float64
            )
            sample_weight_new[:] = np.mean(sample_weight)
            sample_weight_resampled = np.hstack(
                [sample_weight, sample_weight_new]
            ).reshape(-1, 1)
            sample_weight_resampled = np.squeeze(
                normalize(sample_weight_resampled, axis=0, norm="l1")
            )
            return X_resampled, y_resampled, sample_weight_resampled
        else:
            return X_resampled, y_resampled
        


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

    smote = IForestSMOTE(
        random_state=42,
        density_exponent="auto",
        cluster_balance_threshold=0.8,
    )
    smote.fit_resample(X, y)

# %%
