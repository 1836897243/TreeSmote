from imbens.datasets import fetch_zenodo_datasets
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
datasets = ['optical_digits', 'satimage', 'pen_digits', 'sick_euthyroid', 'isolet', 'thyroid_sick', 'coil_2000', 'wine_quality', 'letter_img', 'abalone_19']
samplers = ["TreeSmote", 'SVMSMOTE', 'BorderlineSMOTE', 'None', 'SMOTE']

models = ["DecisionTree", "LinearSVC", 'LGBM', 'MLP']
configs = ['k_neighbor(5).toml']


seeds = range(5)   
commands = []
data = []

for seed, dataset, model, sampler, config  in itertools.product(seeds, datasets, models, samplers, configs):
    cmd = f'python smote_zenodo.py --seed {seed} --dataset \'{dataset}\' --sampler {sampler} --model {model} --config \'{config}\''
    if not os.path.exists(f'results_smote_zenodo/{dataset}-{model}-{sampler}-{config}-{seed}.json'):
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



