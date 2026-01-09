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

samplers =["SelfPacedEnsemble", "BalancedRandomForest", "RUSBoost", "UnderBagging", "EasyEnsemble", "BalanceCascade"]#, "BalanceCascade", "EasyEnsemble"
# samplers = ['RUSBoost', "SmoteEasyEnsembleClassifier", 'SmoteUnderBaggingClassifier']
# samplers = ['SMOTEBaggingClassifier_TreeSMOTE']#['SMOTEBaggingClassifier_TreeSMOTE3','SMOTEBaggingClassifier_SMOTE','SMOTEBaggingClassifier_TreeSMOTE']
#['SMOTEBoostClassifier_SMOTE', 'SMOTEBoostClassifier_TreeSMOTE2']

#['SMOTEBaggingClassifier_SMOTE', 'SMOTEBoostClassifier_SMOTE', 'SMOTEBoostClassifier_TreeSMOTE2', 'SMOTEBaggingClassifier_TreeSMOTE2']#, 'SMOTEBoostClassifier_TreeSMOTE', 'SMOTEBaggingClassifier_TreeSMOTE']
models = ["DecisionTree"]#, "LinearSVC"]
seeds = range(1)       
#configs = ["auto_balance_kn5_TreeSMOTE2.toml",]#["auto_balance_kn5.toml", "auto_balance_kn5_TreeSMOTE.toml", "None"]#, "auto_balance_kn5_TreeSMOTE3.toml"]#['3_verse_imbalance_kn5.toml','auto_balance_kn3.toml', '3_balance_kn5.toml']#, 'auto_verse_imbalance_kn5.toml', "None"]
commands = []
filter_data = ['mc2', 'planning-relax', 'machine_cpu', 'cpu', 'SPECTF', 'heart-h', 'haberman', 'vertebra-column', 'ecoli', 'user-knowledge', 'wholesale-customers', 'thoracic-surgery', 'bwin_amlb', 'arsenic-female-bladder', 'ilpd-numeric', 'blood-transfusion-service-center', 'vehicle', 'vowel', 'Credit_Approval_Classification']

filter_data = ['mc2', 'SPECTF', 'planning-relax', 'machine_cpu', 'ecoli', 'user-knowledge', 'ilpd-numeric', 'Credit_Approval_Classification', 'vertebra-column']
imbalance_types = ['low', 'medium', 'high', 'extreme']
for sampler, imbalance_type_ in itertools.product(samplers, imbalance_types):
    datasets = fetch_openml_datasets(imalance_type=imbalance_type_)
    datasets = datasets.keys()
    auprc = 0
    bac = 0
    f1 = 0
    g_mean = 0
    mcc = 0
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
        if os.path.exists(f'results_optuna/{dataset}-{model}-{sampler}-TreeSmote4-5cv.json') and os.path.exists(f'results_optuna/{dataset}-{model}-{sampler}-None-5cv.json'):
            with open(f'results_optuna/{dataset}-{model}-{sampler}-TreeSmote4-5cv.json', "r") as f:
                res = json.load(f)
                
                auprc_1 = res['test_metrics']['AUPRC']
                bac_1 = res['test_metrics']['BAC']
                f1_1= res['test_metrics']['F1']
                g_mean_1 = res['test_metrics']['GMean']
                mcc_1 = res['test_metrics']['MCC']
                
            with open(f'results_optuna/{dataset}-{model}-{sampler}-None-5cv.json', "r") as f:
                res = json.load(f)
                auprc_2 = res['test_metrics']['AUPRC']
                bac_2 = res['test_metrics']['BAC']
                f1_2 = res['test_metrics']['F1']
                g_mean_2 = res['test_metrics']['GMean']
                mcc_2 = res['test_metrics']['MCC']
            
            # threshold = 0.05
            # if(auprc_2 - auprc_1)>threshold:
            #     print(dataset, sampler, auprc_1, auprc_2)
            # if(bac_2 - bac_1)>threshold:
            #     print(dataset, sampler, bac_1, bac_2)
            # if(f1_2 - f1_1)>threshold:
            #     print(dataset, sampler, f1_1, f1_2)
            # if(g_mean_2 - g_mean_1)>threshold:
            #     print(dataset, sampler, g_mean_1, g_mean_2)
            # if(mcc_2 - mcc_1)>threshold:
            #     print(dataset, sampler, mcc_1, mcc_2)
            win_cnt += int(auprc_1 >= auprc_2)
            lose_cnt += int(auprc_1 < auprc_2)
            # win_cnt += int(bac_1 >= bac_2)
            # lose_cnt += int(bac_1 < bac_2)
            # win_cnt += int(f1_1 >= f1_2)
            # lose_cnt += int(f1_1 < f1_2)
            # win_cnt += int(g_mean_1 >= g_mean_2)
            # lose_cnt += int(g_mean_1 < g_mean_2)
            # win_cnt += int(mcc_1 >= mcc_2)
            # lose_cnt += int(mcc_1 < mcc_2)
            auprc = auprc + (auprc_1 - auprc_2)
            bac = bac + (bac_1 - bac_2)
            f1 = f1 + (f1_1 - f1_2)
            g_mean = g_mean + (g_mean_1 - g_mean_2)
            mcc = mcc + (mcc_1 - mcc_2)
            cnt+=1
    print(imbalance_type_)
    print(sampler)
    if cnt==0:
        continue
    print(f'auprc:{auprc/cnt*100}')
    # print(f'bac:{bac/cnt*100}')
    # print(f'f1:{f1/cnt*100}')
    # print(f'g_mean:{g_mean/cnt*100}')
    # print(f'mcc:{mcc/cnt*100}')
    

    print(f'win_cnt:{win_cnt}')
    print(f'lose_cnt:{lose_cnt}')
    print('-------------------------')

