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
datasets = ['abalone', 'sick_euthyroid', 'spectrometer', 'us_crime', 'yeast_ml8', 'scene', 'car_eval_4', 'coil_2000',]

samplers = ["TVAE", "CTGAN", "CTGANENN"]
seeds = range(5)     
commands = []
data = []

for seed, dataset, sampler  in itertools.product(seeds, datasets, samplers):
    cmd = f'python generate_samples.py --seed {seed} --dataset \'{dataset}\' --sampler {sampler}'
    if not os.path.exists(f'datasets/{dataset}/seed_{seed}/X_augmented_{sampler}.npy'):
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


