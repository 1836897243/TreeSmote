from imbens.datasets import fetch_openml_datasets
from Submitter import SlurmJobSubmitter, ConcurrentJobSubmitter
import itertools
import os
imbalance_types = ['low', 'medium', 'high', 'extreme']
data_names = []
size = []
for imbalance_type_ in imbalance_types:
    datasets_dict = fetch_openml_datasets(imalance_type=imbalance_type_)
    print(f'----------{imbalance_type_}----------')
    datasets = datasets_dict.keys()
    for data in datasets:
        print(data)
        X, y = datasets_dict[data]['data'], datasets_dict[data]['target']
        print(f'X.shape={X.shape}\ty.shape={y.shape}')
        print('\n')
        data_names.append(data)
        size.append(X.shape[0])

#sort by size
sorted_data = sorted(zip(data_names, size), key=lambda x: x[1])
print('sorted datasets by size:')
for data, s in sorted_data:
    print(f'{data}, size={s}')

