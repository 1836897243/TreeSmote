from imbens.datasets import fetch_zenodo_datasets
from Submitter import SlurmJobSubmitter, ConcurrentJobSubmitter
import itertools
import os
'''
optical_digits
satimage
pen_digits
sick_euthyroid
isolet
thyroid_sick
coil_2000
wine_quality
letter_img
abalone_19
'''
datasets = ['optical_digits', 'satimage', 'pen_digits', 'sick_euthyroid', 'thyroid_sick', 'coil_2000', 'wine_quality', 'letter_img', 'abalone_19', 'isolet']#

samplers = ['IDENTITY', 'TVAE', 'CTGAN', 'CTGANENN', 'DGOT',"TreeSmote"]
# 
models = ["LinearSVC","DecisionTree","MLP", "LGBM"]#, "LinearSVC"]

import numpy as np

seeds = range(5)       
commands = []
data = []
metrics = ["AUPRC_macro", "BAC_macro", "F1_macro", "G_mean_macro", "MCC_macro"]

def get_data(metrics, models, dataset, sampler, seed):
    results = []
    for metric in metrics:
        model_results = []
        for model in models:
            if sampler == 'TreeSmote':
                file = f'results_smote_zenodo/{dataset}-{model}-{sampler}-k_neighbor(5).toml-{seed}.json'
            elif sampler == 'IDENTITY':
                file = f'results_smote_zenodo/{dataset}-{model}-None-k_neighbor(5).toml-{seed}.json'
            else:
                file = f'results_smote_GM/{dataset}-{model}-{sampler}-{seed}.json'
            if not os.path.exists(file):
                raise ValueError(f'Missing result for {file}')
            else:
                import json
                with open(file, 'r') as f:
                    result = json.load(f)
                if metric not in result:
                    raise ValueError(f'Missing {metric} in result for {file}')
                # return result[metric]*100
                model_results.append(result[metric]*100)
        results.append(np.mean(model_results))
    return np.mean(results)

def get_df(metrics, models, datasets,samplers, seeds):# format mean+-std
    results = {dataset: {sampler: [] for sampler in samplers} for dataset in datasets}
    for dataset, sampler, seed in itertools.product(datasets, samplers, seeds):
        results[dataset][sampler].append(get_data(metrics, models, dataset, sampler, seed))
    import pandas as pd
    df = pd.DataFrame(columns=['Dataset'] + samplers)
    for dataset in datasets:
        row = {'Dataset': dataset}
        for sampler in samplers:
            mean = np.mean(results[dataset][sampler])
            std = np.std(results[dataset][sampler])
            row[sampler] = f'{mean:.2f}±{std:.2f}'
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    return df


print('Computing mean results across metrics and models...')
df = get_df(metrics, models, datasets, samplers, seeds)
print(df)

print('Computing mean results across models on AUPRC.')
df_auprc = get_df(['AUPRC_macro'], models, datasets, samplers, seeds)
print(df_auprc)

print('Computing mean results across models on BAC.')
df_bac = get_df(['BAC_macro'], models, datasets, samplers, seeds)
print(df_bac)

print('Computing mean results across models on f1-score.')
df_f1 = get_df(['F1_macro'], models, datasets, samplers, seeds)
print(df_f1)

print('Computing mean results across models on G-mean.')
df_gmean = get_df(['G_mean_macro'], models, datasets, samplers, seeds)
print(df_gmean)

print('Computing mean results across models on MCC.')
df_mcc = get_df(['MCC_macro'], models, datasets, samplers, seeds)
print(df_mcc)

