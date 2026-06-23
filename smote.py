import argparse
import numpy as np
from sklearn.model_selection import train_test_split
# from sklearn.metrics import average_precision_score, f1_score, balanced_accuracy_score
from sklearn.metrics import (
    average_precision_score, f1_score, balanced_accuracy_score,
    precision_score, recall_score, matthews_corrcoef, confusion_matrix
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import toml
from imbens import pipeline as pl
from imbens.sampler import TreeSmote, SMOTE, KMeansSMOTE, BorderlineSMOTE, SVMSMOTE
from imbens.ensemble import SelfPacedEnsembleClassifier, BalanceCascadeClassifier, EasyEnsembleClassifier, RUSBoostClassifier, UnderBaggingClassifier, SMOTEBaggingClassifier, SMOTEBoostClassifier
import smote_variants as sv
from imbens.datasets import fetch_openml_datasets, fetch_zenodo_datasets
import json

def get_args():
    parser = argparse.ArgumentParser(description="Run imbalanced learning experiment with configurable sampler and model.")
    parser.add_argument("--sampler", type=str, default="None",
                        choices=["TreeSmote", "SMOTE", "KMeansSMOTE", "BorderlineSMOTE", "SVMSMOTE", "None", "SMOTE_TL"],
                        help="Resampling method to use.")
    parser.add_argument("--model", type=str, default="DecisionTree",
                        choices=["DecisionTree", "LinearSVC", "MLP", "LGBM"], help="Model to train.")
    parser.add_argument("--dataset", type=str, default='', help="name of dataset from fetch_openml_datasets().")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--config", type=str, default="None", help="Path to config file (TOML format). If provided, overrides other arguments.")
    return parser.parse_args()



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
        labels_sorted = np.sort(unique_labels)
        cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)
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




def main():
    args = get_args()
    print(f"\n=== Running with parameters ===")
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    np.random.seed(args.seed)


    datasets = fetch_openml_datasets()
    X, y = datasets[args.dataset]['data'], datasets[args.dataset]['target']
    print(f"\nLoaded dataset #{args.dataset}: X.shape={X.shape}, y.shape={y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=args.seed, stratify=y)
    


    if args.model == "DecisionTree":
        model = DecisionTreeClassifier(random_state=args.seed)
    elif args.model == "LinearSVC":
        model = LinearSVC(random_state=args.seed)
    elif args.model == "MLP":
        model = MLPClassifier(random_state=args.seed)
    elif args.model == "LGBM":
        model = LGBMClassifier(random_state=args.seed)
    else:
        raise ValueError(f"Unknown model {args.model}")
    


    # assert args.config != "None", "Please provide a config file."
    if args.config != "None":
        with open(f'config_smote/{args.config}', 'r') as f:
            config = toml.load(f)
    else:
        config = {}
        assert args.sampler == "None" or args.sampler == "RandomOverSampler", "If no config file is provided, sampler must be 'None'."
    smote_params = config
    
    if args.sampler == "TreeSmote":
        sampler = TreeSmote(random_state=args.seed, **smote_params)
    elif args.sampler == "SMOTE":
        sampler = SMOTE(random_state=args.seed, **smote_params)
    elif args.sampler == "KMeansSMOTE":
        sampler = KMeansSMOTE(random_state=args.seed)
    elif args.sampler == "BorderlineSMOTE":
        sampler = BorderlineSMOTE(random_state=args.seed, **smote_params)
    elif args.sampler == "SVMSMOTE":
        sampler = SVMSMOTE(random_state=args.seed, **smote_params)
    elif args.sampler == "SMOTE_TL":
        sampler = sv.SMOTE_TomekLinks(random_state=args.seed)
    elif args.sampler == "None":
        sampler = None
    else:
        raise ValueError(f"Unknown sampler {args.sampler}")

    X_train, y_train = sampler.fit_resample(X_train, y_train) if sampler is not None else (X_train, y_train)
    clf = pl.make_pipeline(model)
    

    

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)


    if hasattr(clf, "decision_function"):
        y_score = clf.decision_function(X_test)
    elif hasattr(clf, "predict_proba"):
        y_score = clf.predict_proba(X_test)
    else:

        y_score = y_pred


    AUPRC_macro, F1_macro, BAC_macro, G_mean_macro, MCC_macro = compute_metrics(y_test, y_score, y_pred)

    print("\n=== Evaluation Metrics ===")
    print(f'sampler: {args.sampler}, model: {args.model}')
    print(f"AUPRC (macro): {AUPRC_macro:.4f}")
    print(f"F1 Score (macro): {F1_macro:.4f}")
    print(f"Balanced Accuracy (macro): {BAC_macro:.4f}")
    print(f"G_mean_macro: {G_mean_macro}")
    print(f"MCC_macro: {MCC_macro}")
    print(f'seed: {args.seed}, dataset: {args.dataset}')
    print(f'=========================\n')
    file_prefix = f'{args.dataset}-{args.model}-{args.sampler}-{args.config}-{args.seed}'
    # Save results to a file
    results = {
        "dataset": args.dataset,
        "model": args.model,
        "sampler": args.sampler,
        "seed": args.seed,
        "AUPRC_macro": AUPRC_macro,
        "F1_macro": F1_macro,
        "BAC_macro": BAC_macro,
        "G_mean_macro":G_mean_macro,
        "MCC_macro":MCC_macro,
    }
    with open(f"results_smote/{file_prefix}.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()