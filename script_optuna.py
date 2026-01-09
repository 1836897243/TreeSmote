from imbens.datasets import fetch_openml_datasets
from Submitter import SlurmJobSubmitter, ConcurrentJobSubmitter
import itertools
import os
datasets = fetch_openml_datasets()
datasets = datasets.keys()

methods = ['SelfPacedEnsemble', "BalancedRandomForest", "RUSBoost", "UnderBagging", "EasyEnsemble", "BalanceCascade"]
models = ["DecisionTree"]
seeds = range(1)       
commands = []
metric = 'auprc'
for dataset, model, method,  in itertools.product(datasets, models, methods):
    cmd = f'python smote_under_ensemble_optuna.py --dataset \'{dataset}\' --method {method} --smote_type TreeSmote4 --model {model} --metric {metric}'
    if not os.path.exists(f'results_optuna_{metric}/{dataset}-{model}-{method}-TreeSmote4.json'):
        commands.append(cmd)
    cmd = f'python smote_under_ensemble_optuna.py --dataset \'{dataset}\' --method {method} --model {model} --metric {metric}'
    if not os.path.exists(f'results_optuna_{metric}/{dataset}-{model}-{method}-None.json'):
        commands.append(cmd)

for cmd in commands:
    print(cmd)
print(f'{len(commands)} commands to run')


import random
random.shuffle(commands)


half = len(commands)//2
commands1 = commands[:half]
commands2 = commands[half:]
submitter = SlurmJobSubmitter(file_prefix='job1', ntasks=2, ncpus=64, require_gpu=False, mem=64, partition='gpujl')
submitter.truncate(0)
submitter.addJobs(commands1)
submitter.submit(job_name='job1',repeat_last=False)


submitter = SlurmJobSubmitter(file_prefix='job2', ntasks=2, ncpus=64, require_gpu=False, mem=64, partition='gpujl')
submitter.truncate(0)
submitter.addJobs(commands2)
submitter.submit(job_name='job2',repeat_last=False)