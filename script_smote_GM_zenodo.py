from imbens.datasets import fetch_openml_datasets
import itertools
import os
datasets = ['optical_digits', 'satimage', 'pen_digits', 'sick_euthyroid', 'isolet', 'thyroid_sick', 'coil_2000', 'wine_quality', 'letter_img', 'abalone_19']# isolet pen_digits 


models = ["LinearSVC","DecisionTree","MLP", 'LGBM']
samplers = ['DGOT']

seeds = range(5)       
commands = []
data = []

for seed, dataset, model, sampler  in itertools.product(seeds, datasets, models, samplers):
    for epoch in [800]:
        cmd = f'python smote_GM_zenodo.py --seed {seed} --dataset \'{dataset}\' --sampler {sampler} --model {model} --GDOT_epoch {epoch}'
        if not os.path.exists(f'results_smote_GM/{dataset}-{model}-{sampler}-{seed}-epoch_{epoch}.json'):
            commands.append(cmd)

samplers = ["TVAE", 'CTGAN','CTGANENN', 'None']    
for seed, dataset, model, sampler  in itertools.product(seeds, datasets, models, samplers):
    cmd = f'python smote_GM_zenodo.py --seed {seed} --dataset \'{dataset}\' --sampler {sampler} --model {model}'
    if not os.path.exists(f'results_smote_GM/{dataset}-{model}-{sampler}-{seed}.json'):
        commands.append(cmd)


for cmd in commands:
    print(cmd)
print(f'{len(commands)} commands to run')


# save
file = "cmds.txt"
with open (file, "w") as f:
    for cmd in commands:
        f.write(cmd + "\n")



