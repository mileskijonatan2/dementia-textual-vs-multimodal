import pandas as pd
import numpy as np
from datasets import Dataset


def create_splits(data):
    """
    Return the 3 balanced splits in terms of class and dementia diagnoses group
    :param data: pandas DataFrame representing the dataset to be split.
    :type data: pd.DataFrame
    :return: tuple of Dataset objects
    """
    train_split, test_split, val_split = [], [], []

    diagnoses = data['group'].value_counts().index.tolist()
    class_difference = data['label'].value_counts()['dementia'] - data['label'].value_counts()['control']

    np.random.seed(42)  # for same shuffling of dataset
    data_shuffled = data.sample(frac=1, random_state=42).reset_index(drop=True)

    for group in diagnoses:
        samples = data_shuffled[data_shuffled['group'] == group].values.tolist()
        if group == "ProbableAD":
            samples = samples[:-class_difference]  # for balanced dataset
        num_samples = len(samples)
        train, test, val = None, None, None
        if num_samples > 5:
            train = samples[:int(num_samples * 0.7)]
            test = samples[int(num_samples * 0.7):int(num_samples * 0.95)]
            val = samples[int(num_samples * 0.95):]
        elif group == "Vascular":
            train = samples[:2]
            test = samples[2:4]
            val = [samples[4]]
        elif group == "Memory":
            train, test, val = [samples[0]], [samples[1]], [samples[2]]
        elif group == "Other":
            train, test, val = [samples[0]], None, None
        if train is not None:
            train_split.extend(train)
        if test is not None:
            test_split.extend(test)
        if val is not None:
            val_split.extend(val)

    return train_split, test_split, val_split


def add_instruction(input_sentences, classify_direction):
    return [f'{classify_direction}{s}' for s in input_sentences]


def transform_dataset(split, classify_direction, seed=42):
    split_dataset = pd.DataFrame(split, columns=["id", "transcript", "label", "group"])

    input_sentences = split_dataset['transcript'].tolist()
    output_sentences = split_dataset['label'].tolist()
    ids = split_dataset['id'].tolist()
    input_sentences = add_instruction(input_sentences, classify_direction)

    transformed_dataset = Dataset.from_dict({"input_text": input_sentences, "target_text": output_sentences, "ids": ids})
    return transformed_dataset.shuffle(seed=seed)


def get_split_datasets(path, classify_direction, seed=42):
    """
    Returns Dataset objects for train, test, val splits for the specified dataset path and diagnoses groups by ids dictionary
    :param path: path to the .csv dataset to be split. Dataset should have columns 'id', 'transcript', 'label', 'group'.
    :type path: str
    :param classify_direction: instruction for fine-tuning
    :type classify_direction: str
    :param seed: seed for reproducibility
    :type seed: int
    """
    data = pd.read_csv(path)

    diagnoses_groups = get_diagnoses_groups(data)

    train_split, test_split, val_split = create_splits(data)
    train_dataset = transform_dataset(train_split, classify_direction, seed=seed)
    test_dataset = transform_dataset(test_split, classify_direction, seed=seed)
    val_dataset = transform_dataset(val_split, classify_direction, seed=seed)

    return train_dataset, test_dataset, val_dataset, diagnoses_groups


def get_diagnoses_groups(data):
    """
    Returns dictionary containing diagnosis group for each id for specified dataset path
    :param data: DataFrame with columns 'id', 'transcript', 'label', 'group'
    :type data: pd.DataFrame
    """
    ids = data['id'].tolist()
    groups = data['group'].tolist()
    group_by_id = dict(zip(ids, groups))
    return group_by_id



