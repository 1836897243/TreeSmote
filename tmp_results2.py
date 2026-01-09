from imbens.datasets import fetch_openml_datasets
from Submitter import SlurmJobSubmitter, ConcurrentJobSubmitter
import itertools
import os
import json
# datasets = fetch_openml_datasets()
# datasets = datasets.keys()
# REMOVE HELEN, CreditCardFraudDetection
# import pdb; pdb.set_trace()
# datasets = [d for d in datasets if d not in ['helena','CreditCardFraudDetection']]
# datasets = ['helena','CreditCardFraudDetection']

samplers =["SelfPacedEnsemble"]#, "BalancedRandomForest", "RUSBoost", "UnderBagging", "EasyEnsemble", "BalanceCascade"]#, "BalanceCascade", "EasyEnsemble"
# samplers = ['RUSBoost', "SmoteEasyEnsembleClassifier", 'SmoteUnderBaggingClassifier']
# samplers = ['SMOTEBaggingClassifier_TreeSMOTE']#['SMOTEBaggingClassifier_TreeSMOTE3','SMOTEBaggingClassifier_SMOTE','SMOTEBaggingClassifier_TreeSMOTE']
#['SMOTEBoostClassifier_SMOTE', 'SMOTEBoostClassifier_TreeSMOTE2']

#['SMOTEBaggingClassifier_SMOTE', 'SMOTEBoostClassifier_SMOTE', 'SMOTEBoostClassifier_TreeSMOTE2', 'SMOTEBaggingClassifier_TreeSMOTE2']#, 'SMOTEBoostClassifier_TreeSMOTE', 'SMOTEBaggingClassifier_TreeSMOTE']
models = ["DecisionTree"]#, "LinearSVC"]
seeds = range(1)       
#configs = ["auto_balance_kn5_TreeSMOTE2.toml",]#["auto_balance_kn5.toml", "auto_balance_kn5_TreeSMOTE.toml", "None"]#, "auto_balance_kn5_TreeSMOTE3.toml"]#['3_verse_imbalance_kn5.toml','auto_balance_kn3.toml', '3_balance_kn5.toml']#, 'auto_verse_imbalance_kn5.toml', "None"]
commands = []
filter_data = ['mc2', 'planning-relax', 'machine_cpu', 'cpu', 'SPECTF', 'heart-h', 'haberman', 'vertebra-column', 'ecoli', 'user-knowledge', 'wholesale-customers', 'thoracic-surgery', 'bwin_amlb', 'arsenic-female-bladder', 'ilpd-numeric', 'blood-transfusion-service-center', 'vehicle', 'vowel', 'Credit_Approval_Classification']

datasets = ['mc2', 'SPECTF', 'planning-relax', 'machine_cpu', 'ecoli', 'user-knowledge', 'ilpd-numeric', 'Credit_Approval_Classification', 'vertebra-column']
imbalance_types = ['low', 'medium', 'high', 'extreme']
for sampler in samplers:
    auprc = []
    bac = []
    f1 = []
    baseline_auprc = []
    baseline_bac = []
    baseline_f1 = []
    
    cnt = 0
    win_cnt = 0
    lose_cnt = 0
    for seed, dataset, model  in itertools.product(seeds, datasets, models):
        # cmd = f'python smote_under_ensemble.py --seed {seed} --dataset \'{dataset}\' --sampler {sampler} --model {model} --config {conf}'
        # if not os.path.exists(f'results/{dataset}-{model}-{sampler}-{conf}-{seed}.json'):
        # # os.remove(f'results/{dataset}-{model}-{sampler}-{conf}-{seed}.json') if os.path.exists(f'results/{dataset}-{model}-{sampler}-{conf}-{seed}.json') else None
        #     commands.append(cmd)
        # import pdb;pdb.set_trace()
        # if dataset in filter_data:
        #     continue
    
        with open(f'results_optuna_auprc/{dataset}-{model}-{sampler}-TreeSmote4.json', "r") as f:
            res = json.load(f)
            auprc.append(res['auprc'])

        with open(f'results_optuna_bac/{dataset}-{model}-{sampler}-TreeSmote4.json', "r") as f:
            res = json.load(f)
            bac.append(res['bac'])
        with open(f'results_optuna_f1/{dataset}-{model}-{sampler}-TreeSmote4.json', "r") as f:
            res = json.load(f)
            f1.append(res['f1'])
        with open(f'results_optuna_auprc/{dataset}-{model}-{sampler}-None.json', "r") as f:
            res = json.load(f)
            baseline_auprc.append(res['auprc'])

        with open(f'results_optuna_bac/{dataset}-{model}-{sampler}-None.json', "r") as f:
            res = json.load(f)
            baseline_bac.append(res['bac'])

        with open(f'results_optuna_f1/{dataset}-{model}-{sampler}-None.json', "r") as f:
            res = json.load(f)
            baseline_f1.append(res['f1'])
    print(f"Sampler: {sampler}")
    print(f"AUPRC: {auprc}, \nBaseline AUPRC: {baseline_auprc}")
    print(f"BAC: {bac}, \nBaseline BAC: {baseline_bac}")
    print(f"F1: {f1}, \nBaseline F1: {baseline_f1}")
    for a, b in zip(auprc, baseline_auprc):
        cnt +=1
        if a > b:
            win_cnt += 1
        elif a < b:
            lose_cnt += 1
    print(f"AUPRC Win: {win_cnt}, Lose: {lose_cnt}, Total: {cnt}")
    cnt, win_cnt, lose_cnt = 0, 0, 0
    for a, b in zip(bac, baseline_bac):
        cnt +=1
        if a > b:
            win_cnt += 1
        elif a < b:
            lose_cnt += 1
    print(f"BAC Win: {win_cnt}, Lose: {lose_cnt}, Total: {cnt}")
    cnt, win_cnt, lose_cnt = 0, 0, 0
    for a, b in zip(f1, baseline_f1):
        cnt +=1
        if a > b:
            win_cnt += 1
        elif a < b:
            lose_cnt += 1
    print(f"F1 Win: {win_cnt}, Lose: {lose_cnt}, Total: {cnt}")
    print(f'mean AUPRC: {sum(auprc)/len(auprc)}, mean Baseline AUPRC: {sum(baseline_auprc)/len(baseline_auprc)}')
    print(f'mean BAC: {sum(bac)/len(bac)}, mean Baseline BAC: {sum(baseline_bac)/len(baseline_bac)}')
    print(f'mean F1: {sum(f1)/len(f1)}, mean Baseline F1: {sum(baseline_f1)/len(baseline_f1)}')