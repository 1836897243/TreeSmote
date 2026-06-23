from imbens.datasets import fetch_openml_datasets
import itertools
import os
datasets = fetch_openml_datasets()
low_datasets = fetch_openml_datasets(imalance_type='low') 
medium_datasets = fetch_openml_datasets(imalance_type='medium')
high_datasets = fetch_openml_datasets(imalance_type='high')
extreme_datasets = fetch_openml_datasets(imalance_type='extreme')
datasets =   set(medium_datasets.keys()) | set(low_datasets.keys()) | set(high_datasets.keys()) | set(extreme_datasets.keys()) #

samplers = ["TreeSmote", 'SVMSMOTE', 'BorderlineSMOTE', 'None', 'SMOTE', 'SMOTE_TL']

models =["MLP", "LGBM", "LinearSVC","DecisionTree"]
configs = ['k_neighbor(5).toml']
seeds = range(5)       
commands = []
data = []
for seed, dataset, model, sampler, config  in itertools.product(seeds, datasets, models, samplers, configs):
    cmd = f'python smote.py --seed {seed} --dataset \'{dataset}\' --sampler {sampler} --model {model} --config \'{config}\''
    if not os.path.exists(f'results_smote/{dataset}-{model}-{sampler}-{config}-{seed}.json'):
        commands.append(cmd)
    else:
        data.append(dataset)
data = set(data)
print(data)
for cmd in commands:
    print(cmd)
print(f'{len(commands)} commands to run')


# save
file = "cmds.txt"
with open (file, "w") as f:
    for cmd in commands:
        f.write(cmd + "\n")


