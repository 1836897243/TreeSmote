"""
Base class for the under-sampling method.
"""
# Adapted from imbalanced-learn

# Authors: Guillaume Lemaitre
# License: MIT

# %%
LOCAL_DEBUG = False

if not LOCAL_DEBUG:
    from ..base import BaseSampler
    from .._over_sampling._smote.base import SMOTE
    from .._over_sampling._smote.tree import TreeSMOTE
    from .._over_sampling._smote.tree3 import TreeSMOTE3
    from .._over_sampling._smote.tree2 import TreeSMOTE2
    from .._over_sampling._smote.tree4 import TreeSMOTE4
else:
    import sys  # For local test

    sys.path.append("../..")
    # from .base import BaseSampler
    # from ._over_sampling._smote.base import SMOTE
    # from ._over_sampling._smote.tree import TreeSMOTE
    # from ._over_sampling._smote.tree3 import TreeSMOTE3

import numpy as np
class BaseUnderSampler(BaseSampler):
    """Base class for under-sampling algorithms.

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """

    _sampling_type = "under-sampling"

    _sampling_strategy_docstring = """sampling_strategy : float, str, dict, callable, default='auto'
        Sampling information to sample the data set.

        - When ``float``, it corresponds to the desired ratio of the number of
          samples in the minority class over the number of samples in the
          majority class after resampling. Therefore, the ratio is expressed as
          :math:`\\alpha_{us} = N_{m} / N_{rM}` where :math:`N_{m}` is the
          number of samples in the minority class and
          :math:`N_{rM}` is the number of samples in the majority class
          after resampling.

          .. warning::
             ``float`` is only available for **binary** classification. An
             error is raised for multi-class classification.

        - When ``str``, specify the class targeted by the resampling. The
          number of samples in the different classes will be equalized.
          Possible choices are:

            ``'majority'``: resample only the majority class;

            ``'not minority'``: resample all classes but the minority class;

            ``'not majority'``: resample all classes but the majority class;

            ``'all'``: resample all classes;

            ``'auto'``: equivalent to ``'not minority'``.

        - When ``dict``, the keys correspond to the targeted classes. The
          values correspond to the desired number of samples for each targeted
          class.

        - When callable, function taking ``y`` and returns a ``dict``. The keys
          correspond to the targeted classes. The values correspond to the
          desired number of samples for each class.
        """.rstrip()
    
    def __init__(self, over_sampling_config:dict = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.over_sampling_config = over_sampling_config

    def fit_resample(self, X, y, *, sample_weight=None, **kwargs):
        if self.over_sampling_config is None:
          return super().fit_resample(X, y, sample_weight=sample_weight, **kwargs)
        else:

          if sample_weight is not None:
              res =  super().fit_resample(X, y, sample_weight=sample_weight, **kwargs)
              if len(res)==3:
                X_under, y_under, sample_weight_under = res
                more_res = None
              else:
                X_under, y_under, sample_weight_under, more_res = res
              X_under, y_under, sample_weight_under = super().fit_resample(X, y, sample_weight=sample_weight, **kwargs)
          else:
              res = super().fit_resample(X, y, **kwargs)
              if len(res)==2:
                X_under, y_under = res
                more_res = None
              else:
                X_under, y_under, more_res = res
              sample_weight_under = None

          # ---------------------------------------------------------
          # 按原始类别比例 SMOTE 过采样，使少数类扩增 more
          # ---------------------------------------------------------

          # 原始类别分布
          orig_unique, orig_counts = np.unique(y, return_counts=True)
          orig_count_map = dict(zip(orig_unique, orig_counts))

          orig_min = min(orig_counts)
          orig_max = max(orig_counts)
          #   ------------------------------------
          if self.over_sampling_config['ratio'] == 'auto':
            smote_ratio = orig_max/orig_min
          else:
            smote_ratio = float(self.over_sampling_config['ratio'])
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
                  if self.over_sampling_config['type'] == 'verse_imbalance':
                    ratio = smote_ratio - (smote_ratio - 1) * ((orig_c - orig_min) / (orig_max - orig_min))
                  elif self.over_sampling_config['type'] == 'balance':
                    ratio = smote_ratio
                  else:
                     raise ValueError(f"未知的 over_sampling_config['type']: {self.over_sampling_config['type']}")
                  # orig_max

              n_target = int(N_under * ratio)

              if n_target > N_under:
                smote_strategy[cls] = n_target
          
          # 若所有类不需要过采样，直接返回欠采样结果
          if len(smote_strategy) == 0:
              if sample_weight_under is not None:
                  if more_res is None:
                    return X_under, y_under, sample_weight_under
                  else:
                    return X_under, y_under, sample_weight_under, more_res
              if more_res is None:
                return X_under, y_under
              else:
                return X_under, y_under, more_res
          k_neighbors = self.over_sampling_config['k_neighbors']
          while k_neighbors > 0:
            try:
                if "over_sampling_type" not in self.over_sampling_config or self.over_sampling_config["over_sampling_type"] == "SMOTE":
                  smote = SMOTE(
                      sampling_strategy=smote_strategy,
                      k_neighbors=k_neighbors,
                      random_state=self.random_state
                  )
                elif self.over_sampling_config['over_sampling_type'] == 'TreeSMOTE':
                  smote = TreeSMOTE(
                      sampling_strategy=smote_strategy,
                      k_neighbors=k_neighbors,
                      random_state=self.random_state
                  )
                elif self.over_sampling_config['over_sampling_type'] == 'TreeSMOTE2':
                  smote = TreeSMOTE2(
                      sampling_strategy=smote_strategy,
                      k_neighbors=k_neighbors,
                      random_state=self.random_state,
                  )
                elif self.over_sampling_config['over_sampling_type'] == 'TreeSMOTE3':
                  smote = TreeSMOTE3(
                      sampling_strategy=smote_strategy,
                      k_neighbors=k_neighbors,
                      random_state=self.random_state,
                  )
                elif self.over_sampling_config['over_sampling_type'] == 'TreeSMOTE4':
                  smote = TreeSMOTE4(
                      sampling_strategy=smote_strategy,
                      k_neighbors=k_neighbors,
                      dt_max_depth=self.over_sampling_config.get('dt_max_depth', None),
                      over_sampling_ratio=self.over_sampling_config.get('ratio', 1.0),
                      random_state=self.random_state,
                  )
                Res = smote.fit_resample(X_under, y_under, sample_weight=sample_weight_under)
                if more_res is not None:
                  return (*Res, more_res)
                return Res

            except ValueError as e:
                print(f'exceptions: {e}')
                k_neighbors -= 1  # 尝试更小的 k

        raise ValueError("所有 k_neighbors 尝试均失败（已降到 0）")

class BaseCleaningSampler(BaseSampler):
    """Base class for under-sampling algorithms.

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """

    _sampling_type = "clean-sampling"

    _sampling_strategy_docstring = """sampling_strategy : str, list or callable
        Sampling information to sample the data set.

        - When ``str``, specify the class targeted by the resampling. Note the
          the number of samples will not be equal in each. Possible choices
          are:

            ``'majority'``: resample only the majority class;

            ``'not minority'``: resample all classes but the minority class;

            ``'not majority'``: resample all classes but the majority class;

            ``'all'``: resample all classes;

            ``'auto'``: equivalent to ``'not minority'``.

        - When ``list``, the list contains the classes targeted by the
          resampling.

        - When callable, function taking ``y`` and returns a ``dict``. The keys
          correspond to the targeted classes. The values correspond to the
          desired number of samples for each class.
        """.rstrip()


# %%
