from imbens.datasets import fetch_openml_datasets
from Submitter import SlurmJobSubmitter, ConcurrentJobSubmitter
import itertools
import os
imbalance_types = ['low', 'medium', 'high', 'extreme']
for imbalance_type_ in imbalance_types:
    datasets = fetch_openml_datasets(imalance_type=imbalance_type_)
    print(f'----------{imbalance_type_}----------')
    datasets = datasets.keys()
    for data in datasets:
        print(data)

