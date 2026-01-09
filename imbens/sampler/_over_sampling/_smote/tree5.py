import numpy as np
import math
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

import numpy as np
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from .base import BaseSMOTE  # 确保 BaseSMOTE 在同目录或正确导入

class TreeSMOTE5(BaseSMOTE):
    """Tree-based ensemble SMOTE with multiple decision trees."""

    def __init__(
        self,
        *,
        sampling_strategy="auto",
        random_state=None,
        k_neighbors=2,
        n_jobs=None,
        decision_tree_estimator=None,
        max_trees=None,
    ):
        super().__init__(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=k_neighbors,
            n_jobs=n_jobs,
        )
        self.decision_tree_estimator = decision_tree_estimator
        self.max_trees = max_trees

    def _validate_estimator(self):
        super()._validate_estimator()
        if self.decision_tree_estimator is None:
            self.decision_tree_estimator = DecisionTreeClassifier(
                max_depth=None, random_state=self.random_state
            )
        else:
            self.decision_tree_estimator = clone(self.decision_tree_estimator)

    def _fit_resample(self, X, y, sample_weight=None):
        rng = np.random.default_rng(self.random_state)
        self._validate_estimator()

        classes, counts = np.unique(y, return_counts=True)
        majority_count = max(counts)
        minority_count = min(counts)
        minority_classes = classes[counts == minority_count]

        # 根据不平衡比例决定树数量
        imb_ratio = majority_count / minority_count
        n_trees = int(np.ceil(imb_ratio))
        if self.max_trees:
            n_trees = min(n_trees, self.max_trees)

        # 训练多棵决策树
        trees = []
        for i in range(n_trees):
            tree = clone(self.decision_tree_estimator)
            tree.random_state = None if self.random_state is None else self.random_state + i
            tree.fit(X, y)
            trees.append(tree)

        # 初始化结果容器
        X_resampled = [X.copy()]
        y_resampled = [y.copy()]

        # 定义叶子邻居搜索函数
        def get_leaf_info(tree, X, y):
            leaf_nodes = tree.apply(X)
            children_left = tree.tree_.children_left
            children_right = tree.tree_.children_right
            is_leaf = (children_left == _tree.TREE_LEAF)
            leaf_ids = np.where(is_leaf)[0]
            leaf_to_samples = {leaf: np.flatnonzero(leaf_nodes == leaf) for leaf in leaf_ids}
            return leaf_ids, leaf_to_samples, children_left, children_right

        # 对每个少数类生成样本
        for cls in minority_classes:
            cls_idx = np.flatnonzero(y == cls)
            n_gen = majority_count - counts[classes == cls][0]

            while n_gen > 0:
                # 随机选一棵树
                tree = rng.choice(trees)
                leaf_ids, leaf_to_samples, children_left, children_right = get_leaf_info(tree, X, y)

                # 定义递归邻居查找
                def get_neighbors(leaf, cls_idx, visited=None, depth=0, max_depth=5):
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
                        sibling = (
                            children_left[p] if children_left[p] != leaf else children_right[p]
                        )
                        candidates = np.concatenate([
                            candidates,
                            get_neighbors(sibling, cls_idx, visited, depth + 1, max_depth),
                        ])
                    return candidates

                for leaf in leaf_ids:
                    leaf_cls_idx = np.intersect1d(leaf_to_samples[leaf], cls_idx)
                    if len(leaf_cls_idx) == 0:
                        continue

                    neighbors_idx = get_neighbors(leaf, cls_idx)
                    if len(neighbors_idx) <= 1:
                        continue
                    neighbors_idx = neighbors_idx.astype(np.int64)
                    nbrs = NearestNeighbors(
                        n_neighbors=min(self.k_neighbors, len(neighbors_idx))
                    ).fit(X[neighbors_idx])
                    nn_idx = nbrs.kneighbors(X[leaf_cls_idx], return_distance=False)[:, 1:]

                    for i, idx in enumerate(leaf_cls_idx):
                        if n_gen <= 0:
                            break
                        chosen = rng.choice(nn_idx[i])
                        X_new_sample = (
                            X[idx] + rng.random() * (X[neighbors_idx[chosen]] - X[idx])
                        ).reshape(1, -1)

                        # 只有树预测正确才保留
                        y_pred = tree.predict(X_new_sample)
                        if y_pred[0] == cls:
                            X_resampled.append(X_new_sample)
                            y_resampled.append(np.array([cls]))
                            n_gen -= 1

                    if n_gen <= 0:
                        break

        X_resampled = np.vstack(X_resampled)
        y_resampled = np.hstack(y_resampled)

        # sample_weight 处理
        if sample_weight is not None:
            sample_weight_new = np.empty(y_resampled.shape[0] - y.shape[0], dtype=np.float64)
            sample_weight_new[:] = np.mean(sample_weight)
            sample_weight_resampled = np.hstack([sample_weight, sample_weight_new]).reshape(-1, 1)
            sample_weight_resampled = np.squeeze(
                normalize(sample_weight_resampled, axis=0, norm="l1")
            )
            return X_resampled, y_resampled, sample_weight_resampled

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

    smote = TreeSMOTE(
        random_state=42,
        density_exponent="auto",
        cluster_balance_threshold=0.8,
    )
    smote.fit_resample(X, y)

# %%
