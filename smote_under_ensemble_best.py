import argparse
import numpy as np
from sklearn.model_selection import train_test_split
# from sklearn.metrics import average_precision_score, f1_score, balanced_accuracy_score
from sklearn.metrics import (
    average_precision_score, f1_score, balanced_accuracy_score,
    precision_score, recall_score, matthews_corrcoef, confusion_matrix
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC

from imbens import pipeline as pl
from imbens.sampler import TreeSMOTE, TreeSMOTE2, TreeSMOTE3, TreeSMOTE4, TreeSMOTE5, TreeSMOTE6, SMOTE, KMeansSMOTE, BorderlineSMOTE, SVMSMOTE, IForestSMOTE
from imbens.ensemble import SelfPacedEnsembleClassifier, BalanceCascadeClassifier, EasyEnsembleClassifier, RUSBoostClassifier, UnderBaggingClassifier, BalancedRandomForestClassifier
# SMOTEBaggingClassifier, SMOTEBoostClassifier, \
#     SmoteRUSBoostClassifier, SmoteEasyEnsembleClassifier, SmoteBalancedRandomForestClassifier, SmoteUnderBaggingClassifier
from imbens.datasets import fetch_openml_datasets
import json
import toml
# ---------------------------------------------------------
# 命令行参数
# ---------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser(description="Run imbalanced learning experiment with configurable sampler and model.")
    parser.add_argument("--method", type=str, default="None",
                        choices=["None","SelfPacedEnsemble", "BalanceCascade", "BalancedRandomForest", "EasyEnsemble", "RUSBoost", "UnderBagging"],
                        help="Resampling method to use.")
    parser.add_argument("--model", type=str, default="DecisionTree",
                        choices=["DecisionTree", "LinearSVC"], help="Model to train.")
    parser.add_argument("--dataset", type=str, default='', help="name of dataset from fetch_openml_datasets().")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--config", type=str, default="None", help="Path to config file (TOML format). If provided, overrides other arguments.")
    return parser.parse_args()


# ---------------------------------------------------------
# 指标计算函数
# ---------------------------------------------------------
def compute_metrics(y_true, y_score, y_pred):
    """
    Compute AUPRC, Macro F1, and Balanced Accuracy.
    y_score: probability or decision function.
    """
    # For LinearSVC, decision_function gives continuous score
    if y_score.ndim == 2 and y_score.shape[1] == 2:
        y_score = y_score[:, 1]
    AUPRC_macro = average_precision_score(y_true, y_score, average="macro")
    F1_macro = f1_score(y_true, y_pred, average="macro")
    BAC_macro = balanced_accuracy_score(y_true, y_pred)


    precisions = precision_score(y_true, y_pred, average=None, zero_division=0)
    recalls = recall_score(y_true, y_pred, average=None, zero_division=0)

    # ensure arrays
    precisions = np.asarray(precisions, dtype=float)
    recalls = np.asarray(recalls, dtype=float)

    # per-class G = sqrt(precision * recall). protect negatives/zeros (zero_division handled above)
    per_class_g = np.sqrt(np.clip(precisions * recalls, 0.0, None))

    # macro G-mean as arithmetic mean of per-class G (user can change to geometric if desired)
    if per_class_g.size == 0:
        G_mean_macro = float("nan")
    else:
        G_mean_macro = float(np.mean(per_class_g))

    unique_labels = np.unique(y_true)
    if unique_labels.size == 2:
        # binary: compute TP, TN, FP, FN for the positive label
        # confusion_matrix with labels sorted; pick positive as label with greater value? choose unique_labels[1]
        labels_sorted = np.sort(unique_labels)
        pos_label = labels_sorted[-1]  # treat the larger label value as "positive"
        cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)
        # confusion_matrix for binary with labels [neg, pos]:
        # cm = [[TN, FP],
        #       [FN, TP]]
        if cm.shape == (2,2):
            TN, FP, FN, TP = cm.ravel()[0], cm.ravel()[1], cm.ravel()[2], cm.ravel()[3]
            num = (TP * TN) - (FP * FN)
            den = (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
            if den <= 0:
                MCC = 0.0  # define as 0 when denominator is zero (common practical fallback)
            else:
                MCC = num / np.sqrt(den)
        else:
            # fallback: use sklearn
            MCC = float(matthews_corrcoef(y_true, y_pred))
    else:
        # multiclass: rely on sklearn implementation
        MCC = float(matthews_corrcoef(y_true, y_pred))

    return AUPRC_macro, F1_macro, BAC_macro, G_mean_macro, MCC


# ---------------------------------------------------------
# 主流程
# ---------------------------------------------------------
def main():
    args = get_args()
    print(f"\n=== Running with parameters ===")
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    # 固定随机种子
    np.random.seed(args.seed)

    # -----------------------------------------------------
    # 1. 加载数据集
    # -----------------------------------------------------
    datasets = fetch_openml_datasets()
    X, y = datasets[args.dataset]['data'], datasets[args.dataset]['target']
    print(f"\nLoaded dataset #{args.dataset}: X.shape={X.shape}, y.shape={y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=args.seed, stratify=y)

    # -----------------------------------------------------
    # 2. 选择模型
    # -----------------------------------------------------
    if args.model == "DecisionTree":
        model = DecisionTreeClassifier(random_state=args.seed)
    elif args.model == "LinearSVC":
        model = SVC(kernel='linear', probability=True, random_state=args.seed)#LinearSVC(random_state=args.seed)
    else:
        raise ValueError(f"Unknown model {args.model}")
    
    if args.model == "DecisionTree":
        model = DecisionTreeClassifier(random_state=args.seed)
    else:
        model = SVC(kernel="linear", probability=True, random_state=args.seed)

    if args.method == "None":
        clf = pl.make_pipeline(model)
    else:
        method_map = {
            "SelfPacedEnsemble": SelfPacedEnsembleClassifier,
            "BalanceCascade": BalanceCascadeClassifier,
            "BalancedRandomForest": BalancedRandomForestClassifier,
            "EasyEnsemble": EasyEnsembleClassifier,
            "RUSBoost": RUSBoostClassifier,
            "UnderBagging": UnderBaggingClassifier,
        }
        method_params = {'k_bins': 5} if args.method == "SelfPacedEnsemble" else {}
        smote_params = None #{'k_neighbors': 5, 'over_sampling_ratio': 1.5, 'dt_max_depth': 3, 'ratio': 1.5, 'over_sampling_type': 'TreeSMOTE4', 'type': 'balance'} if 
        cls = method_map[args.method]

        if args.method == "BalancedRandomForest":
            clf = cls(
                random_state=args.seed,
                **method_params,
                over_sampling_config=smote_params,
            )
        else:
            clf = cls(
                estimator=model,
                random_state=args.seed,
                **method_params,
                over_sampling_config=smote_params,
            )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_score = (
        clf.decision_function(X_test)
        if hasattr(clf, "decision_function")
        else clf.predict_proba(X_test)
    )

    # # -----------------------------------------------------
    # # 3. 选择采样器
    # # -----------------------------------------------------
    # sampler_dict = {
    #     "None": None
    # }
    # if args.config == "None":
    #     config = None
    # else:
    #     with open(f'configs/{args.config}', 'r') as f:
    #         config = toml.load(f)
    # config = {
    #     "k_neighbors": 5,
    #     "over_sampling_ratio": 1.5,
    #     "dt_max_depth": 3,
    #     "ratio": 1.5,
    #     "over_sampling_type": "TreeSMOTE4",
    #     "type": "balance"
    # }
    # ensemble_dict = {
    #     "SelfPacedEnsemble": SelfPacedEnsembleClassifier(random_state=args.seed, estimator=model, k_bins=5, over_sampling_config=config),
    #     "BalanceCascade": BalanceCascadeClassifier(random_state=args.seed, estimator=model, over_sampling_config=config),
    #     "BalancedRandomForest": BalancedRandomForestClassifier(random_state=args.seed, over_sampling_config=config),
    #     "EasyEnsemble": EasyEnsembleClassifier(random_state=args.seed, over_sampling_config=config),
    #     "RUSBoost": RUSBoostClassifier(random_state=args.seed, estimator=model, over_sampling_config=config),
    #     "UnderBagging": UnderBaggingClassifier(random_state=args.seed, estimator=model, over_sampling_config=config),
    # }
    # if args.sampler in sampler_dict.keys():
    #     sampler = sampler_dict[args.sampler]
    #     if sampler is not None:
    #         clf = pl.make_pipeline(sampler, model)
    #     else:
    #         clf = pl.make_pipeline(model)
    # elif args.sampler in ensemble_dict.keys():
    #     clf = ensemble_dict[args.sampler]

    

    

    # # -----------------------------------------------------
    # # 4. 构建流水线
    # # -----------------------------------------------------
    

    # # -----------------------------------------------------
    # # 5. 训练与预测
    # # -----------------------------------------------------
    # clf.fit(X_train, y_train)
    # y_pred = clf.predict(X_test)

    # # 获取预测概率或决策分数（用于 AUPRC）
    # if hasattr(clf, "decision_function"):
    #     y_score = clf.decision_function(X_test)
    # elif hasattr(clf, "predict_proba"):
    #     y_score = clf.predict_proba(X_test)
    # else:
    #     # 回退到预测标签（不理想，仅供无概率输出模型）
    #     y_score = y_pred

    # -----------------------------------------------------
    # 6. 计算指标
    # -----------------------------------------------------
    AUPRC_macro, F1_macro, BAC_macro, G_mean_macro, MCC_macro = compute_metrics(y_test, y_score, y_pred)

    print("\n=== Evaluation Metrics ===")
    print(f'method: {args.method}, model: {args.model}')
    print(f"AUPRC (macro): {AUPRC_macro:.4f}")
    print(f"F1 Score (macro): {F1_macro:.4f}")
    print(f"Balanced Accuracy (macro): {BAC_macro:.4f}")
    print(f"G_mean_macro: {G_mean_macro}")
    print(f"MCC_macro: {MCC_macro}")
    print(f'seed: {args.seed}, dataset: {args.dataset}')
    # print(f'=========================\n')
    # file_prefix = f'{args.dataset}-{args.model}-{args.sampler}-{args.config}-{args.seed}'
    # # Save results to a file
    # results = {
    #     "dataset": args.dataset,
    #     "model": args.model,
    #     "sampler": args.sampler,
    #     "seed": args.seed,
    #     "AUPRC_macro": AUPRC_macro,
    #     "F1_macro": F1_macro,
    #     "BAC_macro": BAC_macro,
    #     "G_mean_macro":G_mean_macro,
    #     "MCC_macro":MCC_macro
    # }
    # with open(f"results/{file_prefix}.json", "w") as f:
    #     json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()