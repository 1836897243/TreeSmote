from imbens.datasets import fetch_openml_datasets
from Submitter import SlurmJobSubmitter, ConcurrentJobSubmitter
import itertools
import os
datasets = fetch_openml_datasets()
datasets = datasets.keys()
# REMOVE HELEN, CreditCardFraudDetection
# import pdb; pdb.set_trace()
# datasets = [d for d in datasets if d not in ['helena','CreditCardFraudDetection']]
# datasets = ['helena','CreditCardFraudDetection']
samplers = ["TreeSMOTE", "SMOTE", "BorderlineSMOTE", "SVMSMOTE", "None"]

# samplers = ["SelfPacedEnsemble", "BalancedRandomForest", "RUSBoost", "UnderBagging", "EasyEnsemble"]#, "BalanceCascade", "EasyEnsemble"
# samplers = ['RUSBoost', "SmoteEasyEnsembleClassifier", 'SmoteUnderBaggingClassifier']
# samplers = ['SMOTEBaggingClassifier_TreeSMOTE']#['SMOTEBaggingClassifier_TreeSMOTE3','SMOTEBaggingClassifier_SMOTE','SMOTEBaggingClassifier_TreeSMOTE']
#['SMOTEBoostClassifier_SMOTE', 'SMOTEBoostClassifier_TreeSMOTE2']

#['SMOTEBaggingClassifier_SMOTE', 'SMOTEBoostClassifier_SMOTE', 'SMOTEBoostClassifier_TreeSMOTE2', 'SMOTEBaggingClassifier_TreeSMOTE2']#, 'SMOTEBoostClassifier_TreeSMOTE', 'SMOTEBaggingClassifier_TreeSMOTE']
models = ["DecisionTree"]#, "LinearSVC"]
seeds = range(10)       
# configs = ["auto_balance_kn5.toml", "auto_balance_kn5_TreeSMOTE.toml", "None"]#, "auto_balance_kn5_TreeSMOTE3.toml"]#['3_verse_imbalance_kn5.toml','auto_balance_kn3.toml', '3_balance_kn5.toml']#, 'auto_verse_imbalance_kn5.toml', "None"]
commands = []
for seed, dataset, model, sampler,  in itertools.product(seeds, datasets, models, samplers):
    cmd = f'python smote.py --seed {seed} --dataset \'{dataset}\' --sampler {sampler} --model {model}'
    if not os.path.exists(f'results_smote/{dataset}-{model}-{sampler}-5-{seed}.json'):
    # os.remove(f'results_smote/{dataset}-{model}-{sampler}-{conf}-{seed}.json') if os.path.exists(f'results_smote/{dataset}-{model}-{sampler}-{conf}-{seed}.json') else None
        commands.append(cmd)

for cmd in commands:
    print(cmd)
print(f'{len(commands)} commands to run')


# submmiter = SlurmJobSubmitter(commands=commands, job_name='smote', max_jobs=20,require_gpu=False, mem=16)
# commands = ['python test.py']*10
#reverse commands to test ConcurrentJobSubmitter
# commands = reversed(commands)

# submmiter = ConcurrentJobSubmitter(commands, 9)
# submmiter.submit()

# import random
# random.shuffle(commands)


quarter = int(len(commands)//4)
submitter = SlurmJobSubmitter(file_prefix='1test_jobs', ntasks=7, ncpus=10, require_gpu=False, mem=256, partition='gpujl')
submitter.truncate(0)
submitter.addJobs(commands[:2*quarter])
# submitter.submit(job_name='1test_job',repeat_last=False)


submitter = SlurmJobSubmitter(file_prefix='2test_jobs', ntasks=7, ncpus=10, require_gpu=False, mem=256, partition='gpujl')
submitter.truncate(0)
submitter.addJobs(commands[2*quarter:])
# submitter.submit(job_name='2test_job',repeat_last=False)

# quarter = int(len(commands)//4)
# submitter = SlurmJobSubmitter(file_prefix='3test_jobs', ntasks=7, ncpus=10, require_gpu=False, mem=256, partition='gpujl')
# submitter.truncate(0)
# submitter.addJobs(commands[:2*quarter])
# # submitter.submit(job_name='3test_job',repeat_last=False)


# submitter = SlurmJobSubmitter(file_prefix='4test_jobs', ntasks=7, ncpus=10, require_gpu=False, mem=256, partition='gpujl')
# submitter.truncate(0)
# submitter.addJobs(commands[2*quarter:])
# # submitter.submit(job_name='4test_job',repeat_last=False)