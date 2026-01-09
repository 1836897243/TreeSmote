"""Class to perform random under-sampling."""

# Adapted from imbalanced-learn

# Authors: Guillaume Lemaitre
#          Christos Aridas
#          Zhining Liu <zhining.liu@outlook.com>
# License: MIT

# %%
LOCAL_DEBUG = False

if not LOCAL_DEBUG:
    from ....utils._docstring import Substitution, _random_state_docstring
    from ....utils._validation import _deprecate_positional_args, check_target_type
    from ..base import BaseUnderSampler
    # from ../../../over_sampling._smote.base import SMOTE
    from ....sampler._over_sampling._smote.base import SMOTE
else:  # pragma: no cover
    import sys  # For local test

    sys.path.append("../../..")
    from sampler._under_sampling.base import BaseUnderSampler
    from utils._docstring import Substitution
    from utils._docstring import _random_state_docstring
    from utils._validation import _deprecate_positional_args, check_target_type
    from sampler._over_sampling._smote.base import SMOTE
import numpy as np
from sklearn.utils import _safe_indexing, check_random_state
from sklearn.utils.validation import validate_data


@Substitution(
    sampling_strategy=BaseUnderSampler._sampling_strategy_docstring,
    random_state=_random_state_docstring,
)
class RandomUnderOverSampler(BaseUnderSampler):
    """Class to perform random under-sampling.

    Under-sample the majority class(es) by randomly picking samples
    with or without replacement.

    Read more in the `User Guide <https://imbalanced-learn.org/stable/under_sampling.html#controlled-under-sampling>`_.

    Parameters
    ----------
    {sampling_strategy}

    {random_state}

    replacement : bool, default=False
        Whether the sample is with or without replacement.

    Attributes
    ----------
    sample_indices_ : ndarray of shape (n_new_samples,)
        Indices of the samples selected.

    See Also
    --------
    NearMiss : Undersample using near-miss samples.

    Notes
    -----
    Supports multi-class resampling by sampling each class independently.
    Supports heterogeneous data as object array containing string and numeric
    data.

    Examples
    --------
    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imbens.sampler._under_sampling import \
RandomUnderSampler # doctest: +NORMALIZE_WHITESPACE
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ...  weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape %s' % Counter(y))
    Original dataset shape Counter({{1: 900, 0: 100}})
    >>> rus = RandomUnderSampler(random_state=42)
    >>> X_res, y_res = rus.fit_resample(X, y)
    >>> print('Resampled dataset shape %s' % Counter(y_res))
    Resampled dataset shape Counter({{0: 100, 1: 100}})
    """

    @_deprecate_positional_args
    def __init__(
        self, *, sampling_strategy="auto", random_state=None, replacement=False, smote_ratio=1.5, k_neighbors=2
    ):
        super().__init__(sampling_strategy=sampling_strategy)
        self.random_state = random_state
        self.replacement = replacement
        self.smote_ratio = smote_ratio  # 默认 SMOTE 扩增比例 1.5
        self.k_neighbors = k_neighbors

    def _check_X_y(self, X, y):
        y, binarize_y = check_target_type(y, indicate_one_vs_all=True)
        X, y = validate_data(
            self,
            X,
            y,
            reset=True,
            accept_sparse=["csr", "csc"],
            dtype=None,
            ensure_all_finite=False,
        )
        return X, y, binarize_y

    def _build_smote(self, smote_strategy, k_neighbors, random_state, X, y):
        """自动调整 k_neighbors，直到 SMOTE 可用"""
        while k_neighbors > 0:
            try:
                smote = SMOTE(
                    sampling_strategy=smote_strategy,
                    k_neighbors=k_neighbors,
                    random_state=random_state
                )
                X_res, y_res = smote.fit_resample(X, y)
                return X_res, y_res
            except ValueError as e:
                k_neighbors -= 1  # 尝试更小的 k

        raise ValueError("所有 k_neighbors 尝试均失败（已降到 0）")

    def _fit_resample(self, X, y, sample_weight=None, sample_proba=None):
        """
        修改版：
        1) 先执行欠采样，使所有类别样本数量一致（原逻辑）
        2) 再根据原始类别样本数量比例，用 SMOTE 进行按比例过采样

        self 需要有属性：
            - self.random_state
            - self.sampling_strategy_
            - self.replacement
            - self.smote_ratio   # 比如 1.5
        """

        random_state = check_random_state(self.random_state)

        # -----------------------------
        # 验证 sample_proba
        # -----------------------------
        if sample_proba is None:
            pass
        elif not isinstance(sample_proba, (np.ndarray, list)):
            raise TypeError(
                f"`sample_proba` should be an array-like of shape (n_samples,),"
                f" got {type(sample_proba)} instead."
            )
        else:
            sample_proba = np.asarray(sample_proba)
            if sample_proba.shape != y.shape:
                raise ValueError(
                    f"`sample_proba` should be of shape {y.shape}, got {sample_proba.shape}."
                )
            else:
                try:
                    sample_proba = sample_proba.astype(float)
                except Exception as e:
                    e_args = list(e.args)
                    e_args[0] += (
                        f"\n`sample_proba` should be an array-like with dtype == float,"
                        + f" please check your usage."
                    )
                    e.args = tuple(e_args)
                    raise e

        # ------------------------------------------------
        # 第一步：欠采样（保持你的全部原始逻辑，使所有类数量一致）
        # ------------------------------------------------
        idx_under = np.empty((0,), dtype=int)

        for target_class in np.unique(y):
            class_idx = y == target_class

            if target_class in self.sampling_strategy_.keys():
                # 基于 sample_proba 进行概率采样
                if sample_proba is not None:
                    probabilities = np.array(sample_proba[class_idx]).astype(float)
                    probabilities /= probabilities.sum()
                else:
                    probabilities = None

                n_samples = self.sampling_strategy_[target_class]
                index_target_class = random_state.choice(
                    range(np.count_nonzero(class_idx)),
                    size=n_samples,
                    replace=self.replacement,
                    p=probabilities,
                )
            else:
                index_target_class = slice(None)

            idx_under = np.concatenate(
                (
                    idx_under,
                    np.flatnonzero(class_idx)[index_target_class],
                ),
                axis=0,
            )

        # 保存欠采样索引
        self.sample_indices_ = idx_under
        # -------------------------------------------------
        # --------------------------------------------------
        # 取得欠采样数据
        X_under = _safe_indexing(X, idx_under)
        y_under = _safe_indexing(y, idx_under)

        if sample_weight is not None:
            sample_weight_under = _safe_indexing(sample_weight, idx_under)
        else:
            sample_weight_under = None

        # ---------------------------------------------------------
        # 第二步：按原始类别比例 SMOTE 过采样，使少数类扩增 more
        # ---------------------------------------------------------

        # 原始类别分布
        orig_unique, orig_counts = np.unique(y, return_counts=True)
        orig_count_map = dict(zip(orig_unique, orig_counts))

        orig_min = min(orig_counts)
        orig_max = max(orig_counts)
        #   ------------------------------------
        self.smote_ratio = orig_max/orig_min
        #   ------------------------------------

        # 欠采样后每个类别数量（相同）
        _, under_counts = np.unique(y_under, return_counts=True)
        N_under = under_counts[0]

        smote_strategy = {}

        for cls in orig_unique:
            orig_c = orig_count_map[cls]

            # 计算该类的扩增比例
            if orig_max == orig_min:
                ratio = 1.0
            else:
                # 最少类 → ratio = smote_ratio；最多类 → ratio = 1；中间类线性插值
                # ratio = self.smote_ratio - (self.smote_ratio - 1) * ((orig_c - orig_min) / (orig_max - orig_min))
                ratio = self.smote_ratio
                # orig_max

            n_target = int(N_under * ratio)

            if n_target > N_under:
                smote_strategy[cls] = n_target

        # 若所有类不需要过采样，直接返回欠采样结果
        if len(smote_strategy) == 0:
            if sample_weight_under is not None:
                return X_under, y_under, sample_weight_under
            return X_under, y_under

        # 执行 SMOTE
        X_res, y_res =_safe_indexing(X, idx_under), _safe_indexing(y, idx_under)
        X_res, y_res = self._build_smote(smote_strategy, self.k_neighbors, random_state, X_res, y_res)
        # ---------------------------
        # sample_weight 扩展（SMOTE）
        # ---------------------------
        if sample_weight_under is not None:
            n_orig = len(sample_weight_under)
            n_new = len(y_res) - n_orig

            # 新的合成样本权重设为原样本平均值
            new_weights = np.full(n_new, sample_weight_under.mean())
            sample_weight_res = np.concatenate([sample_weight_under, new_weights])

            return X_res, y_res, sample_weight_res
 
        return X_res, y_res

    def _more_tags(self):  # pragma: no cover
        return {
            "X_types": ["2darray", "string", "sparse", "dataframe"],
            "sample_indices": True,
            "allow_nan": True,
        }

    def __sklearn_tags__(self):  # pragma: no cover
        tags = super().__sklearn_tags__()
        tags.input_tags.two_d_array = True
        tags.input_tags.sparse = True
        tags.input_tags.allow_nan = True
        # tags.sample_indices = True
        return tags


# %%

if __name__ == "__main__":  # pragma: no cover
    from collections import Counter

    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_classes=3,
        class_sep=2,
        weights=[0.1, 0.3, 0.6],
        n_informative=3,
        n_redundant=1,
        flip_y=0,
        n_features=20,
        n_clusters_per_class=1,
        n_samples=1000,
        random_state=10,
    )
    print("Original dataset shape %s" % Counter(y))

    origin_distr = Counter(y)
    target_distr = {2: 200, 1: 100, 0: 100}

    undersampler = RandomUnderOverSampler(random_state=42, sampling_strategy=target_distr)
    X_res, y_res, weight_res = undersampler.fit_resample(X, y, sample_weight=y)

    print("Resampled dataset shape %s" % Counter(y_res))
    print("Test resampled weight shape %s" % Counter(weight_res))

# %%
