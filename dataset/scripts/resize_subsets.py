import json
import os
import random
from collections import defaultdict

import pandas as pd
from pandas import DataFrame


def get_csv() -> list[DataFrame]:
    """[TODO:summary]

    [TODO:description]

    Returns:
        [TODO:description]
    """
    train_df = pd.read_csv("private_train_metadata.csv")
    test_df = pd.read_csv("private_test_metadata.csv")
    train_df = DataFrame(train_df[["image_id", "original_whale_id"]])
    test_df = DataFrame(test_df[["image_id", "original_whale_id"]])
    return [train_df, test_df]


def get_occurences(dfs: list[DataFrame]) -> defaultdict[str, list[str]]:
    """[TODO:summary]

    [TODO:description]

    Args:
        dfs: [TODO:description]

    Returns:
        [TODO:description]
    """
    occurences = defaultdict(list)
    for df in dfs:
        for image_id, whale_id in df.values:
            occurences["whale" + str(whale_id)].append(image_id)
    return occurences


def delete_single_instances(
    occurences: defaultdict[str, list[str]],
) -> defaultdict[str, list[str]]:
    """[TODO:summary]

    [TODO:description]
    Args:
        occurences: [TODO:description]

    Returns:
        [TODO:description]
    """
    one_count = []
    for i in occurences.items():
        if len(i[1]) == 1:
            one_count.append(i[0])
    for i in one_count:
        occurences.pop(i)
    return occurences


def stratified_class_split(
    occurences: defaultdict[str, list[str]],
    train_ratio: float,
    test_ratio: float,
    val_ratio: float,
):
    """generates subsets of given dataset.

    Splits a given whole dataset into train, test and validation sets based on user-defined proportions.

    Args:
        occurences: [TODO:description]
        train_ratio: Percentage of original dataset to be used as training data.
        test_ratio: Percentage of original dataset to be used as testing data.
        val_ratio: Percentage of original dataset to be used as validation data.
    """
    assert (
        round(train_ratio + test_ratio + val_ratio, 5) == 1.0
    ), "Proportions must sum to 1."

    # Extract unique classes (whale IDs)
    all_classes = list(occurences.keys())

    # Shuffle classes to ensure randomness
    random.seed(42)
    random.shuffle(all_classes)

    # Compute number of classes for test (completely unseen)
    num_test = int(len(all_classes) * test_ratio)
    test_classes = all_classes[:num_test]  # Assign these classes to test ONLY

    # Remaining classes for train/val
    remaining_classes = all_classes[num_test:]

    num_train = int(len(remaining_classes) * (train_ratio / (train_ratio + val_ratio)))
    train_classes = remaining_classes[:num_train]
    val_classes = remaining_classes[num_train:]

    train_samples = {cls: occurences[cls] for cls in train_classes}
    val_samples = {cls: occurences[cls] for cls in val_classes}
    test_samples = {cls: occurences[cls] for cls in test_classes}

    # Print summary
    print(
        f"""
    Train Classes: {len(train_classes)} → {len(train_samples)} images
    Validation Classes: {len(val_classes)} → {len(val_samples)} images
    Test Classes (Unseen): {len(test_classes)} → {len(test_samples)} images
    Total Images: {len(train_samples) + len(test_samples) + len(val_samples)}
    """
    )
    return ("train", train_samples), ("test", test_samples), ("validation", val_samples)


def restructure_folder(
    data_splits: tuple[tuple[str, dict[str, list[str]]], ...],
) -> None:
    """Restructures the dataset so that the computed splits are saved to the system.

    [TODO:description]

    Args:
        data_splits: [TODO:description]
    """
    root_dir = "./"
    print("Creating Subset Directories...")
    for i in ("train", "validation", "test"):
        os.mkdir(os.path.join(root_dir, i))
    print("Done")
    print("Restructuring The Dataset..")
    for data_type, data in data_splits:
        print(data_type)
        fp = open(f"./{data_type}/{data_type}_labels.json", "w")
        json.dump(data, fp, indent=4)
        for images in data.values():
            for image_id in images:
                source = os.path.join("../../dataset/images", image_id + ".jpg")
                dst = os.path.join("../../dataset/" + data_type, image_id + ".jpg")
                if os.path.exists(source):
                    os.rename(source, dst)
    print("Done...\nMetadata writting to disk")


if __name__ == "__main__":
    dfs = get_csv()
    occurences = delete_single_instances(get_occurences(dfs))
    restructure_folder(stratified_class_split(occurences, 0.8, 0.1, 0.1))
