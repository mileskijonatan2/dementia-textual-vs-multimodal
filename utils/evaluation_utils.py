import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, classification_report


def get_metrics(true, predicted, model_name):
    """
    Get metrics scores based on true and predicted labels
    :param true: true labels
    :type true: list
    :param predicted: predicted labels
    :type predicted: list
    """
    predicted = predicted
    true = true

    for p in predicted:
        if p not in ["dementia", "control"]:
            print(f"{p} is not a dementia or control. Predicted by model: {model_name}. Metrics cannot be calculated.")
            return None

    accuracy = accuracy_score(true, predicted)
    precision = precision_score(true, predicted, pos_label='dementia')
    recall = recall_score(true, predicted, pos_label='dementia')
    f1 = f1_score(true, predicted, pos_label='dementia')

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


def analyze_misclassified_samples(ids, true, predicted, groups_dict, model_name):
    """
    Determines on the proportion of misclassified samples by the model for each dementia subgroup (ProbableAD, MCI, Vascular, etc.)
    :param ids: ids of samples
    :type ids: list
    :param true: true labels
    :type true: list
    :param predicted: predicted labels
    :type predicted: list
    :param groups_dict: dictionary containing the ids as keys and the specific group as value
    :type groups_dict: dict
    :param model_name: name of the model
    :type model_name: str
    """
    count_dementia, count_control = 0, 0  # misclassified samples
    groups = list(set(groups_dict.values()))  # diagnoses groups
    misclassified_samples_per_group = dict()
    misclassified_group_percent = dict()
    misclassified_ids = []

    for group in groups:
        misclassified_samples_per_group[group] = 0

    for id, gt, pred in zip(ids, true, predicted):
        if gt != pred:
            misclassified_ids.append(id)
            if gt == 'dementia':
                count_dementia += 1
                misclassified_samples_per_group[groups_dict[id]] += 1
            else:
                count_control += 1

    for k in misclassified_samples_per_group.keys():
        misclassified_group_percent[f'{k} %'] = [misclassified_samples_per_group[k] / count_dementia]
        misclassified_samples_per_group[k] = [misclassified_samples_per_group[k]]

    misclassified_group_percent['Model'] = [model_name]
    misclassified_group_percent['Control'] = [count_control / (count_control + count_dementia)]
    misclassified_group_percent['Dementia'] = [count_dementia / (count_control + count_dementia)]
    misclassified_samples_per_group['Model'] = [model_name]
    misclassified_samples_per_group['Control'] = [count_control]
    misclassified_samples_per_group['Dementia'] = [count_dementia]
    ms_group_percent = pd.DataFrame(misclassified_group_percent)
    ms_samples_per_group = pd.DataFrame(misclassified_samples_per_group)

    return ms_group_percent, ms_samples_per_group, misclassified_ids

