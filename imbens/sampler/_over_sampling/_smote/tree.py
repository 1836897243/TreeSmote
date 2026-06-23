import numpy as np
import math
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

import numpy as np
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from .base import BaseSMOTE  
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




@Substitution(
    sampling_strategy=BaseOverSampler._sampling_strategy_docstring,
    n_jobs=_n_jobs_docstring,
    random_state=_random_state_docstring,
)
class TreeSmote(BaseSMOTE):
    """ """
    
    @_deprecate_positional_args
    def __init__(
        self,
        *,
        sampling_strategy="auto",
        random_state=None,
        k_neighbors=5,
        n_jobs=None,
        rf_n_estimators=100,
        rf_max_depth=None,
        rf_min_samples_leaf=1,
        over_sampling_ratio=1.0,
        filter = True,
        file_name = None,

    ):
        super().__init__(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=k_neighbors,
            n_jobs=n_jobs,
        )
        self.rf_n_estimators = rf_n_estimators
        self.rf_max_depth = rf_max_depth
        self.rf_min_samples_leaf = rf_min_samples_leaf
        self.rf_ = None
        self.over_sampling_ratio = over_sampling_ratio
        self.filter = filter
        self.file_name = file_name if file_name is not None else "tree_smote"

    def _validate_estimator(self):
        if self.rf_ is None:
            self.rf_ = RandomForestClassifier(
                n_estimators=self.rf_n_estimators,
                max_depth=self.rf_max_depth,
                min_samples_leaf=self.rf_min_samples_leaf,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )

    def _fit_resample(self, X, y, sample_weight=None):
        import numpy as np
        from scipy.sparse import csr_matrix
        from sklearn.neighbors import NearestNeighbors

        rng = np.random.default_rng(self.random_state)
        self._validate_estimator()

        self.rf_.fit(X, y)
        leaf_indices = self.rf_.apply(X)  # (n_samples, n_trees)

        n_samples, n_trees = leaf_indices.shape
        X_res, y_res = [], []

        classes, counts = np.unique(y, return_counts=True)
        majority = counts.max()


        if self.sampling_strategy == "auto":
            sampling = {c: majority - cnt for c, cnt in zip(classes, counts) if cnt < majority}
        elif self.sampling_strategy == "ratio":
            if self.over_sampling_ratio <= 1.0:
                raise ValueError("over_sampling_ratio must be > 1.0 when sampling_strategy='ratio'")
            sampling = {c: math.ceil(cnt * self.over_sampling_ratio) - cnt for c, cnt in zip(classes, counts)}
        else:
            sampling = self.sampling_strategy_


        for cls, n_gen in sampling.items():
            if n_gen <= 0:
                continue

            cls_idx = np.flatnonzero(y == cls)
            if len(cls_idx) < 2:
                continue  


            n_nodes = max(leaf_indices.max() + 1, X.shape[1])
            sim_matrix = np.zeros((len(cls_idx), len(cls_idx)), dtype=np.float32)
   
            for t in range(n_trees):
                tree  = self.rf_.estimators_[t].tree_
                node_indicator = tree.decision_path(X[cls_idx].astype('float32'))
                sim_matrix += (node_indicator @ node_indicator.T).toarray()

            sim_matrix /= n_trees
            sim_matrix_norm = sim_matrix / sim_matrix.max(axis=1).reshape(-1, 1)
            dist_matrix = 1.0 - sim_matrix_norm
            dist_matrix = np.asarray(dist_matrix)
            knn = NearestNeighbors(
                n_neighbors=min(self.k_neighbors + 1, len(cls_idx)),
                metric="precomputed",
                n_jobs=self.n_jobs
            )
            knn.fit(dist_matrix)
            neigh = knn.kneighbors(dist_matrix, return_distance=False)[:, 1:]

            rows = rng.integers(0, len(cls_idx), size=n_gen)
            cols = rng.integers(0, neigh.shape[1], size=n_gen)
            steps = rng.random(n_gen)

            X_cls = X[cls_idx]
            X_new = X_cls[rows] + steps[:, None] * (X_cls[neigh[rows, cols]] - X_cls[rows])
            y_new = np.full(n_gen, cls)

            X_res.append(X_new)
            y_res.append(y_new)

        if not X_res:
            return X, y if sample_weight is None else (X, y, sample_weight)

        X_res = np.vstack(X_res)
        y_res = np.hstack(y_res)

        if self.filter:
            y_pred = self.rf_.predict(X_res)
            mask = y_pred == y_res
   
            X_res = X_res[mask]
            y_res = y_res[mask]

        X_res = np.vstack([X, X_res])
        y_res = np.hstack([y, y_res])

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

    smote = TreeSmote(
        random_state=42,
        density_exponent="auto",
        cluster_balance_threshold=0.8,
    )
    smote.fit_resample(X, y)

# %%
