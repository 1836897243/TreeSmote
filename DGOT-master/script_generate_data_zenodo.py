import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, parent_dir)
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
seeds = range(5)     
commands = []
data = []

for seed, dataset  in itertools.product(seeds, datasets):
    cmd = f'python run.py --seed {seed} --dataset \'{dataset}\''
    if not os.path.exists(f'datasets/{dataset}/seed_{seed}/X_augmented_DGOT_epoch_800.npy'):
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



