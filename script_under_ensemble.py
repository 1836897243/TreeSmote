from imbens.datasets import fetch_openml_datasets
import itertools
import os
datasets = fetch_openml_datasets()
datasets = datasets.keys()
methods = ['SelfPacedEnsemble', "BalancedRandomForest", "RUSBoost", "UnderBagging", "BalanceCascade", "EasyEnsemble"]
models = ["DecisionTree"]
seeds = range(5)       
commands = []
for dataset, model, method, seed  in itertools.product(datasets, models, methods, seeds):
    cmd = f'python smote_under_ensemble.py --dataset \'{dataset}\' --method {method} --model {model} --config \'TreeSmote-k_neighbor(5)-over_sampling_ratio(2).toml\' --seed {seed}'
    if not os.path.exists(f'results_under_ensemble/dataset[{dataset}]-model[{model}]-method[{method}]-config[TreeSmote-k_neighbor(5)-over_sampling_ratio(2).toml]-seed[{seed}].json'):
        commands.append(cmd)
    cmd = f'python smote_under_ensemble.py --dataset \'{dataset}\' --method {method} --model {model} --seed {seed}'
    if not os.path.exists(f'results_under_ensemble/dataset[{dataset}]-model[{model}]-method[{method}]-config[None]-seed[{seed}].json'):
        commands.append(cmd)

for cmd in commands:
    print(cmd)
print(f'{len(commands)} commands to run')

# save
file = "cmds.txt"
with open (file, "w") as f:
    for cmd in commands:
        f.write(cmd + "\n")
