import argparse
import json
import optuna
import numpy as np
import os

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    average_precision_score, f1_score, balanced_accuracy_score,
    precision_score, recall_score, matthews_corrcoef, confusion_matrix
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from imbens import pipeline as pl
from imbens.ensemble import (
    SelfPacedEnsembleClassifier, BalanceCascadeClassifier,
    EasyEnsembleClassifier, RUSBoostClassifier,
    UnderBaggingClassifier, BalancedRandomForestClassifier
)
from imbens.datasets import fetch_openml_datasets


# ---------------------------------------------------------
# Args
# ---------------------------------------------------------
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="None",
                        choices=[
                            "None", "SelfPacedEnsemble", "BalanceCascade",
                            "BalancedRandomForest", "EasyEnsemble",
                            "RUSBoost", "UnderBagging"
                        ])
    parser.add_argument("--model", type=str, default="DecisionTree",
                        choices=["DecisionTree", "LinearSVC"])
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smote_type", type=str, default="None",
                        choices=["None", "Smote", "TreeSmote4"])
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--metric", type=str,
                        choices=["auprc", "f1", "bac", "g_mean", "mcc"],
                        required=True)
    return parser.parse_args()


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------
def compute_metrics(y_true, y_score, y_pred):
    if y_score.ndim == 2 and y_score.shape[1] == 2:
        y_score = y_score[:, 1]

    AUPRC_macro = average_precision_score(y_true, y_score, average="macro")
    F1_macro = f1_score(y_true, y_pred, average="macro")
    BAC_macro = balanced_accuracy_score(y_true, y_pred)

    precisions = precision_score(y_true, y_pred, average=None, zero_division=0)
    recalls = recall_score(y_true, y_pred, average=None, zero_division=0)

    precisions = np.asarray(precisions, dtype=float)
    recalls = np.asarray(recalls, dtype=float)

    per_class_g = np.sqrt(np.clip(precisions * recalls, 0.0, None))
    G_mean_macro = float(np.mean(per_class_g)) if per_class_g.size > 0 else float("nan")

    unique_labels = np.unique(y_true)
    if unique_labels.size == 2:
        labels_sorted = np.sort(unique_labels)
        cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)
        if cm.shape == (2, 2):
            TN, FP, FN, TP = cm.ravel()
            den = (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
            MCC = 0.0 if den <= 0 else ((TP * TN) - (FP * FN)) / np.sqrt(den)
        else:
            MCC = float(matthews_corrcoef(y_true, y_pred))
    else:
        MCC = float(matthews_corrcoef(y_true, y_pred))

    return AUPRC_macro, F1_macro, BAC_macro, G_mean_macro, MCC


# ---------------------------------------------------------
# Search spaces
# ---------------------------------------------------------
def get_method_params(trial, method):
    if method == "SelfPacedEnsemble":
        return {"k_bins": trial.suggest_int("k_bins", 1, 10)}
    if method == "BalanceCascade":
        return {"replacement": trial.suggest_categorical("replacement", [True, False])}
    if method in {"BalancedRandomForest", "EasyEnsemble", "UnderBagging"}:
        return {
            "max_samples": trial.suggest_float("max_samples", 0.5, 1.0),
            "max_features": trial.suggest_float("max_features", 0.5, 1.0),
        }
    if method == "RUSBoost":
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-7, 1.0, log=True),
            "algorithm": trial.suggest_categorical("algorithm", ["SAMME", "SAMME.R"]),
        }
    return {}


def get_smote_params(trial, smote_type):
    if smote_type == "Smote":
        return {
            "k_neighbors": trial.suggest_int("smote_k_neighbors", 1, 10),
            "ratio": trial.suggest_float("over_sampling_ratio", 1.0, 2.0),
            "over_sampling_type": "TreeSMOTE4",
            "type": "balance",
        }
    if smote_type == "TreeSmote4":
        return {
            "k_neighbors": trial.suggest_int("smote_k_neighbors", 1, 10),
            "ratio": trial.suggest_float("over_sampling_ratio", 1.0, 3.0),
            "dt_max_depth": trial.suggest_int("dt_max_depth", 1, 5),
            "over_sampling_type": "TreeSMOTE4",
            "type": "balance",
        }
    return None


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    args = get_args()
    np.random.seed(args.seed)

    # -------- Make directories --------
    os.makedirs("optunaDB", exist_ok=True)
    os.makedirs("results_optuna2", exist_ok=True)

    # -------- Load dataset --------
    datasets = fetch_openml_datasets()
    X, y = datasets[args.dataset]["data"], datasets[args.dataset]["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, random_state=args.seed
    )

    # -----------------------------------------------------
    # Optuna objective
    # -----------------------------------------------------
    def objective(trial):
        scores = []
        skf = StratifiedKFold(
            n_splits=args.n_splits, shuffle=True, random_state=args.seed
        )

        for tr_idx, val_idx in skf.split(X_train, y_train):
            X_tr, X_val = X_train[tr_idx], X_train[val_idx]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]

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
                method_params = get_method_params(trial, args.method)
                smote_params = get_smote_params(trial, args.smote_type)
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

            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_val)
            y_score = (
                clf.decision_function(X_val)
                if hasattr(clf, "decision_function")
                else clf.predict_proba(X_val)
            )

            metrics = compute_metrics(y_val, y_score, y_pred)
            if args.metric == "auprc":
                scores.append(metrics[0])
            elif args.metric == "f1":
                scores.append(metrics[1])
            elif args.metric == "bac":
                scores.append(metrics[2])
            elif args.metric == "g_mean":
                scores.append(metrics[3])
            elif args.metric == "mcc":
                scores.append(metrics[4])

        return float(np.mean(scores))

    # -----------------------------------------------------
    # Optuna (checkpoint / resume)
    # -----------------------------------------------------
    study_name = (
        f"{args.dataset}-{args.model}-{args.method}-"
        f"{args.smote_type}-{args.metric}-seed{args.seed}"
    )
    storage = f"sqlite:///optunaDB/optuna_{study_name}.db"

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        storage=storage,
        load_if_exists=True,
    )

    # 计算剩余 trial
    completed = len(study.trials)
    remaining = max(0, args.n_trials - completed)
    print(f"[Optuna] Completed trials: {completed}, Remaining trials: {remaining}")

    if remaining > 0:
        study.optimize(objective, n_trials=remaining)
    else:
        print("[Optuna] Already reached n_trials, skip optimization.")

    # -----------------------------------------------------
    # Retrain best model
    # -----------------------------------------------------
    trial = optuna.trial.FixedTrial(study.best_params)

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
        method_params = get_method_params(trial, args.method)
        smote_params = get_smote_params(trial, args.smote_type)
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

    metrics = compute_metrics(y_test, y_score, y_pred)
    if args.metric == "auprc":
        score = metrics[0]
    elif args.metric == "f1":
        score = metrics[1]
    elif args.metric == "bac":
        score = metrics[2]
    elif args.metric == "g_mean":
        score = metrics[3]
    elif args.metric == "mcc":
        score = metrics[4]

    result = {
        "dataset": args.dataset,
        "model": args.model,
        "method": args.method,
        "smote_type": args.smote_type,
        "seed": args.seed,
        "best_cv_score": study.best_value,
        f"{args.metric}": score,
        "best_params": study.best_params,
    }
    if not os.path.exists(f"results_optuna_{args.metric}"):
        os.makedirs(f"results_optuna_{args.metric}")
    with open(
        f"results_optuna_{args.metric}/{args.dataset}-{args.model}-{args.method}-{args.smote_type}.json",
        "w",
    ) as f:
        json.dump(result, f, indent=4)

    print(result)
if __name__ == "__main__":
    main()
