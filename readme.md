# Environment Setup

All experiments were conducted using **Python 3.10**.

Create a new conda environment:

```bash
conda create -n TreeSmote python=3.10
conda activate TreeSmote
```

Install the required dependencies:

```bash
pip install \
numpy>=1.16.0 \
scikit-learn>=1.6.0 \
pandas>=2.1.1 \
matplotlib>=3.3.2 \
seaborn>=0.13.2 \
tqdm>=4.50.2 \
openml>=0.14.0 \
platformdirs>=3.0.0 \
lightgbm>=4.6.0 \
toml>=0.10.2 \
smote_variants>=1.0.1 \
sdv>=1.37.2 \
imbalanced-learn \
einops>=0.8.2 \
openpyxl>=3.1.5
```

Alternatively, users may install the dependencies manually according to their platform and package manager.

---

# Reproducibility Instructions

This repository contains the code used in our experiments. The implementation is organized into three categories: **SMOTE-based methods**, **Generative-model-based methods**, and **Ensemble-based methods**.

## SMOTE-Based Methods

Generate the experiment commands:

```bash
python script_smote.py
```

Then execute the commands listed in `cmds.txt`.

The experimental results will be saved in:

```text
./results_smote
```

After all experiments have finished, aggregate and summarize the results by running:

```bash
python results_smote.py
```

---

## Generative-Model-Based Methods

Generate the experiment commands:

```bash
python script_smote_zenodo.py      # Our method
python script_smote_GM_zenodo.py   # Generative-model baselines
```

Then execute the commands listed in `cmds.txt`.

The results will be saved in:

```text
./results_smote_zenodo
./results_smote_GM
```

To aggregate the results after all experiments are completed:

```bash
python results_smote_zenodo.py
```

### Generating Synthetic Data

To generate synthetic samples produced by the generative models:

```bash
python script_generate_data_zenodo.py
```

For DGOT:

```bash
cd DGOT-master/
python script_generate_data_zenodo.py
```

---

## Ensemble-Based Methods

Generate the experiment commands:

```bash
python script_under_ensemble.py
```

Then execute the commands listed in `cmds.txt`.

The results will be saved in:

```text
./results_under_ensemble
```

To aggregate the results after all experiments are completed:

```bash
python results_under_ensemble.py
```

---

## Notes

- All generated command files (`cmds.txt`) may contain a large number of experiments. Depending on the available computational resources, execution can take a considerable amount of time.
- Please ensure that all dependencies listed in the environment configuration are installed before running the experiments.
- Results reported in the paper can be reproduced by aggregating the outputs generated in the corresponding `results_*` directories.

## Acknowledgements

Part of the ensemble-based implementation is adapted from the CLIMB benchmark and the Imbalanced-Ensemble project:

https://github.com/ZhiningLiu1998/imbalanced-ensemble