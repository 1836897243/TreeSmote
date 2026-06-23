import pandas as pd
import numpy as np
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.sampling import Condition
from sklearn.model_selection import train_test_split

from sdv.single_table import CTGANSynthesizer

def augment_minority_ctgan(X: np.ndarray, y: np.ndarray):
    X_df = pd.DataFrame(X)
    y_df = pd.Series(y, name="label").astype(str)
    df = pd.concat([X_df, y_df], axis=1)
    df.columns = df.columns.astype(str)
    for col in df.columns:
        if col != "label":
            df[col] = df[col].astype(float)
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)
    metadata.update_column("label", sdtype="categorical") 
    synthesizer = CTGANSynthesizer(metadata)
    synthesizer.fit(df)
    counts = df["label"].value_counts()
    for minority, count in counts.items():
        if count < counts.max():
            n_to_generate = counts.max() - count
            synthetic_minority = synthesizer.sample_from_conditions(
                conditions=[Condition(column_values={"label": minority}, num_rows=int(n_to_generate))],
            )
            df = pd.concat([df, synthetic_minority], ignore_index=True)
    balanced_df = df
    X_balanced = balanced_df.drop(columns="label").to_numpy()
    y_balanced = balanced_df["label"].to_numpy()
    return X_balanced, y_balanced
   
def augment_minority_tvae(X: np.ndarray, y: np.ndarray):
    X_df = pd.DataFrame(X)
    y_df = pd.Series(y, name="label").astype(str)
    df = pd.concat([X_df, y_df], axis=1)
    df.columns = df.columns.astype(str)
    for col in df.columns:
        if col != "label":
            df[col] = df[col].astype(float)
    counts = df["label"].value_counts()
    minority = counts.idxmin()
    n_to_generate = counts.max() - counts.min()
    df2_train = df.copy()
    if counts.max() / counts.min() > 2:
        majority = counts.idxmax()
        n_majority_to_keep = 2*counts.min()
        df_majority = df[df["label"] == majority]
        df_minority = df[df["label"] == minority]
        df_majority_downsampled = df_majority.sample(n=n_majority_to_keep, random_state=0)
        df2_train = pd.concat([df_majority_downsampled, df_minority], ignore_index=True)
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df2_train)
    metadata.update_column("label", sdtype="categorical")
    synthesizer = TVAESynthesizer(metadata)
    synthesizer.fit(df2_train)
    for minority, count in counts.items():
        if count < counts.max():
            n_to_generate = counts.max() - count
            synthesic_minority = synthesizer.sample_from_conditions(
                conditions=[Condition(column_values={"label": minority}, num_rows=int(n_to_generate))],
            )
            df = pd.concat([df, synthesic_minority], ignore_index=True)
    X_balanced = df.drop(columns="label").to_numpy()
    y_balanced = df["label"].to_numpy()
    return X_balanced, y_balanced

from collections import Counter
from imblearn.under_sampling import EditedNearestNeighbours
def CTGANENN(df, targetLabel, epochs=300, batch_size=512):
    df = df.copy()
    class_counts = Counter(df[targetLabel])
    max_count = max(class_counts.values())
    data_concat = []
    for cls in class_counts.keys():
        minClass = df[df[targetLabel] == cls]
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(minClass)
        metadata.validate()

        n_current = len(minClass)
        genData = max_count - n_current
        if genData > 0:
            synthesizer = CTGANSynthesizer(metadata)
            synthesizer.fit(minClass)
            synthetic_data = synthesizer.sample(num_rows=genData)
            data_concat.append(pd.concat([minClass, synthetic_data], ignore_index=True))
        else:
            data_concat.append(minClass)

    data = pd.concat(data_concat, ignore_index=True)
    enn = EditedNearestNeighbours(n_neighbors=3)
    X=data.drop([targetLabel],axis=1)
    y=data[targetLabel]
    X, y = enn.fit_resample(X, y)
    return X,y

def augment_minority_ctganenn(X: np.ndarray, y: np.ndarray):
    X_df = pd.DataFrame(X)
    y_df = pd.Series(y, name="label").astype(str)
    df = pd.concat([X_df, y_df], axis=1)
    df.columns = df.columns.astype(str)
    for col in df.columns:
        if col != "label":
            df[col] = df[col].astype(float)
    
    X_balanced, y_balanced = CTGANENN(df, targetLabel="label")
    return X_balanced.to_numpy(), y_balanced.to_numpy()



from imbens.datasets import fetch_openml_datasets, fetch_zenodo_datasets
import argparse
import os
def get_args():
    parser = argparse.ArgumentParser(description="Run imbalanced learning experiment with configurable sampler and model.")
    parser.add_argument("--sampler", type=str, default="None",
                        choices=["TVAE", "CTGAN", "CTGANENN", "None"],
                        help="Resampling method to use.")
    parser.add_argument("--dataset", type=str, default='', help="name of dataset from fetch_openml_datasets().")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    print(f"\n=== Running with parameters ===")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    np.random.seed(args.seed)
    datasets = fetch_zenodo_datasets()
    X, y = datasets[args.dataset]['data'], datasets[args.dataset]['target']
    print(f"\nLoaded dataset #{args.dataset}: X.shape={X.shape}, y.shape={y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=args.seed, stratify=y)
    file_dir = f"datasets/{args.dataset}/seed_{args.seed}"
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    if args.sampler == "CTGAN":
        X_bal, y_bal = augment_minority_ctgan(X_train, y_train)
    elif args.sampler == "TVAE":
        X_bal, y_bal = augment_minority_tvae(X_train, y_train)
    elif args.sampler == "CTGANENN":
        X_bal, y_bal = augment_minority_ctganenn(X_train, y_train)
    else:
        raise ValueError(f"Unsupported sampler: {args.sampler}")
    

    y_test = y_test.astype(str)
    y_train = y_train.astype(str)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_bal = le.transform(y_bal)
    y_test = le.transform(y_test)
    np.save(f"{file_dir}/X_augmented_{args.sampler}.npy", X_bal)
    np.save(f"{file_dir}/y_augmented_{args.sampler}.npy", y_bal)
    np.save(f"{file_dir}/X_test.npy", X_test)
    np.save(f"{file_dir}/y_test.npy", y_test)
    np.save(f"{file_dir}/X_train.npy", X_train)
    np.save(f"{file_dir}/y_train.npy", y_train)
    
    

