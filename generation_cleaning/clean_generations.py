import os
import pandas as pd
from utils import get_split_datasets, get_metrics, analyze_misclassified_samples


def correct_anomalies(original_folder_path: str, symbols=False):
    name_suffix = "_sy" if symbols else ""
    open(f"../results/cleaned/misclassifed_groups/percentage_misclassified_group{name_suffix}.csv", "w").close()
    open(f"../results/cleaned/misclassifed_groups/count_misclassified_group{name_suffix}.csv", "w").close()
    open(f"../results/cleaned/metrics{name_suffix}.csv", "w").close()

    suffix = '_preds_sy.csv' if symbols else '_preds.csv'
    _, _, _, group_by_id = get_split_datasets(
        "../dataset/text/complete_dataset_text_level_symbols.csv", "", seed=42)

    all_files_paths = [os.path.join(folder_path, filename) for filename in os.listdir(original_folder_path) if suffix in filename]
    for f in all_files_paths:
        df = pd.read_csv(f)
        ids, preds, gt = df['id'].values.tolist(), df['predictions'].values.tolist(), df['true'].values.tolist()
        corrected_preds = []
        for i, p, t in zip(ids, preds, gt):
            if p not in ['control', 'dementia']:
                if 'control' in p:
                    corrected_preds.append('control')
                elif 'dementia' in p or '1' in p:
                    corrected_preds.append('dementia')
            else:
                corrected_preds.append(p)

        new_df = pd.DataFrame({'id': ids, 'predictions': corrected_preds, 'true': gt})

        model_name = f.split("/predictions\\")[1].split(suffix)[0]
        metrics = get_metrics(gt, corrected_preds, model_name)
        accuracy, precision, recall, f1 = metrics.values()
        preds_df = new_df

        preds_df.to_csv(f'../results/cleaned/predictions/{model_name}_preds{name_suffix}.csv', index=False)

        df_metrics = pd.DataFrame(
            {"Model": [model_name], "Accuracy": [accuracy], "Precision": [precision], "Recall": [recall], "F1": [f1]})
        ms_group_percent, ms_samples_per_group, misclassified_ids = analyze_misclassified_samples(ids, gt,
                                                                                                  corrected_preds,
                                                                                                  group_by_id,
                                                                                                  model_name)

        try:
            existing = pd.read_csv(f"../results/cleaned/metrics{name_suffix}.csv")
        except pd.errors.EmptyDataError:
            existing = pd.DataFrame()
        combined = pd.concat([existing, df_metrics], ignore_index=True)
        combined.to_csv(f"../results/cleaned/metrics{name_suffix}.csv", index=False)

        try:
            existing = pd.read_csv(f"../results/cleaned/misclassifed_groups/percentage_misclassified_group{name_suffix}.csv")
        except pd.errors.EmptyDataError:
            existing = pd.DataFrame()
        combined = pd.concat([existing, ms_group_percent], ignore_index=True)
        combined.to_csv(f"../results/cleaned/misclassifed_groups/percentage_misclassified_group{name_suffix}.csv", index=False)

        try:
            existing = pd.read_csv(f"../results/cleaned/misclassifed_groups/count_misclassified_group{name_suffix}.csv")
        except pd.errors.EmptyDataError:
            existing = pd.DataFrame()
        combined = pd.concat([existing, ms_samples_per_group], ignore_index=True)
        combined.to_csv(f"../results/cleaned/misclassifed_groups/count_misclassified_group{name_suffix}.csv", index=False)

        with open(f"../results/cleaned/misclassified_ids/{model_name}{name_suffix}.txt", "w") as f:
            f.write(f"Misclassified predictions of {model_name}{name_suffix}\n")
            for id in misclassified_ids:
                f.write(f"{id}\n")


def find_anomalies(file_path: str):
    df = pd.read_csv(file_path)
    preds, gt = df['predictions'].values.tolist(), df['true'].values.tolist()

    total_invalid = 0
    model_name = file_path.split('predictions')[1]

    for pred, gt in zip(preds, gt):
        if pred not in ['control', 'dementia']:
            print(f"{pred}, {gt}, {model_name}\n")
            if 'control' in pred or 'dementia' in pred:
                print(True)
            else:
                print(False)
            total_invalid += 1

    print(f"{model_name} total invalid: {total_invalid}\n")
    print("--------------------------------------------")


if __name__ == "__main__":
    # folder_path = "C:/Users/User/OneDrive/Десктоп/DEMENTIA_ARTICLE/new-experiments-dementia-results/train_01_12_symbols/results/predictions"
    folder_path = "C:/Users/User/OneDrive/Десктоп/DEMENTIA_ARTICLE/new-experiments-dementia-results/train_06_12_no_sy/results/predictions"

    all_files_paths = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if '.git' not in filename]
    """
    print(all_files_paths)
    print(len(all_files_paths))
    for f in all_files_paths:
        find_anomalies(f)"""

    correct_anomalies(folder_path, symbols=False)
