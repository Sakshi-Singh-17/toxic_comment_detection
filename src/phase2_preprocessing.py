"""
Phase 2: Data Cleaning and Train/Test Split
Toxic Comment Classification Project

Tasks:
1. Load the original Jigsaw dataset
2. Validate required columns
3. Remove missing comments
4. Remove empty comments
5. Remove exact duplicate comments
6. Create binary toxicity target
7. Separate features and target
8. Perform stratified 80/20 train-test split
9. Save train/test datasets

IMPORTANT:
TF-IDF is NOT fitted in this phase.

The TF-IDF vectorizer will be fitted ONLY on the training
data in a later phase to prevent data leakage.
"""

import os

import pandas as pd

from sklearn.model_selection import train_test_split


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_PATH = os.path.join("data", "train.csv")

OUTPUT_DIR = "data"

TRAIN_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "train_clean.csv"
)

TEST_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "test_clean.csv"
)


# Original Jigsaw toxicity columns
TOXICITY_COLUMNS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]


REQUIRED_COLUMNS = [
    "id",
    "comment_text",
    *TOXICITY_COLUMNS
]


# ============================================================
# 2. LOAD DATASET
# ============================================================

def load_dataset(file_path):
    """
    Load the Jigsaw dataset.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"\nDataset not found: {file_path}\n"
            "Make sure train.csv is inside the data/ folder."
        )

    print("=" * 70)
    print("LOADING DATASET")
    print("=" * 70)

    df = pd.read_csv(file_path)

    print(f"File: {file_path}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    return df


# ============================================================
# 3. VALIDATE COLUMNS
# ============================================================

def validate_columns(df):
    """
    Check whether all required Jigsaw columns exist.
    """

    print("\n" + "=" * 70)
    print("VALIDATING DATASET COLUMNS")
    print("=" * 70)

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:

        print("Missing required columns:")

        for column in missing_columns:
            print("-", column)

        raise ValueError(
            "\nDataset structure does not match the expected "
            "Jigsaw Toxic Comment dataset."
        )

    print("All required columns are present.")

    print("\nColumns:")
    for column in df.columns:
        print("-", column)


# ============================================================
# 4. REMOVE MISSING COMMENTS
# ============================================================

def remove_missing_comments(df):
    """
    Remove rows where comment_text is missing.
    """

    print("\n" + "=" * 70)
    print("CHECKING MISSING COMMENTS")
    print("=" * 70)

    missing_count = df["comment_text"].isna().sum()

    print("Missing comments:", missing_count)

    if missing_count > 0:

        df = df.dropna(
            subset=["comment_text"]
        ).copy()

        print(
            "Rows remaining:",
            f"{len(df):,}"
        )

    return df


# ============================================================
# 5. REMOVE EMPTY COMMENTS
# ============================================================

def remove_empty_comments(df):
    """
    Remove comments containing only whitespace.
    """

    print("\n" + "=" * 70)
    print("CHECKING EMPTY COMMENTS")
    print("=" * 70)

    empty_mask = (
        df["comment_text"]
        .astype(str)
        .str.strip()
        .eq("")
    )

    empty_count = empty_mask.sum()

    print("Empty comments:", empty_count)

    if empty_count > 0:

        df = df.loc[~empty_mask].copy()

        print(
            "Rows remaining:",
            f"{len(df):,}"
        )

    return df


# ============================================================
# 6. CHECK AND REMOVE EXACT DUPLICATES
# ============================================================

def remove_duplicate_comments(df):
    """
    Remove exact duplicate comments.

    Why?

    If the same comment appears in both training and testing,
    test performance can become artificially optimistic.

    We remove duplicates BEFORE train/test splitting.
    """

    print("\n" + "=" * 70)
    print("CHECKING DUPLICATE COMMENTS")
    print("=" * 70)

    duplicate_count = df["comment_text"].duplicated().sum()

    print(
        "Duplicate comment rows:",
        duplicate_count
    )

    if duplicate_count > 0:

        print(
            "Removing exact duplicate comments..."
        )

        df = df.drop_duplicates(
            subset=["comment_text"],
            keep="first"
        ).copy()

        print(
            "Rows remaining:",
            f"{len(df):,}"
        )

    else:

        print(
            "No exact duplicate comments found."
        )

    return df


# ============================================================
# 7. CREATE BINARY TARGET
# ============================================================

def create_binary_target(df):
    """
    Create binary target.

    Rule:

    If ANY of the six toxicity labels is 1:
        target = 1 (Toxic)

    If ALL six labels are 0:
        target = 0 (Non-toxic)
    """

    print("\n" + "=" * 70)
    print("CREATING BINARY TARGET")
    print("=" * 70)

    df = df.copy()

    df["target"] = (
        df[TOXICITY_COLUMNS]
        .sum(axis=1)
        .gt(0)
        .astype(int)
    )

    print(
        "Binary target created successfully."
    )

    print("\nTarget rule:")
    print(
        "Any toxicity label = 1 -> Toxic"
    )
    print(
        "All toxicity labels = 0 -> Non-toxic"
    )

    return df


# ============================================================
# 8. DISPLAY CLASS DISTRIBUTION
# ============================================================

def show_class_distribution(df):
    """
    Display class counts and percentages.
    """

    print("\n" + "=" * 70)
    print("CLASS DISTRIBUTION")
    print("=" * 70)

    counts = (
        df["target"]
        .value_counts()
        .sort_index()
    )

    percentages = (
        df["target"]
        .value_counts(normalize=True)
        .sort_index()
        * 100
    )

    for target_value in [0, 1]:

        label = (
            "Non-toxic"
            if target_value == 0
            else "Toxic"
        )

        count = counts.get(
            target_value,
            0
        )

        percentage = percentages.get(
            target_value,
            0
        )

        print(
            f"{label:12s}: "
            f"{count:8,d} "
            f"({percentage:6.2f}%)"
        )


# ============================================================
# 9. PREPARE X AND y
# ============================================================

def prepare_features_and_target(df):
    """
    Separate comments (X) and binary target (y).
    """

    X = df["comment_text"]

    y = df["target"]

    return X, y


# ============================================================
# 10. STRATIFIED TRAIN/TEST SPLIT
# ============================================================

def split_dataset(X, y):
    """
    Split data into 80% training and 20% testing.

    stratify=y preserves approximately the same class
    proportions in both datasets.
    """

    print("\n" + "=" * 70)
    print("TRAIN / TEST SPLIT")
    print("=" * 70)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(
        f"Training samples: {len(X_train):,}"
    )

    print(
        f"Testing samples : {len(X_test):,}"
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# ============================================================
# 11. CHECK SPLIT DISTRIBUTION
# ============================================================

def check_split_distribution(
    y_train,
    y_test
):
    """
    Verify that stratification preserved class proportions.
    """

    print("\n" + "=" * 70)
    print("TRAIN / TEST CLASS DISTRIBUTION")
    print("=" * 70)

    train_distribution = (
        y_train
        .value_counts(normalize=True)
        .sort_index()
        * 100
    )

    test_distribution = (
        y_test
        .value_counts(normalize=True)
        .sort_index()
        * 100
    )

    print("\nTraining set:")

    for target_value in [0, 1]:

        label = (
            "Non-toxic"
            if target_value == 0
            else "Toxic"
        )

        print(
            f"{label:12s}: "
            f"{train_distribution.get(target_value, 0):.2f}%"
        )

    print("\nTesting set:")

    for target_value in [0, 1]:

        label = (
            "Non-toxic"
            if target_value == 0
            else "Toxic"
        )

        print(
            f"{label:12s}: "
            f"{test_distribution.get(target_value, 0):.2f}%"
        )


# ============================================================
# 12. SAVE TRAIN / TEST DATA
# ============================================================

def save_datasets(
    X_train,
    X_test,
    y_train,
    y_test
):
    """
    Save the cleaned train and test datasets.

    These files are useful for inspection and reproducibility.

    Note:
    TF-IDF is NOT fitted here.
    """

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    train_df = pd.DataFrame({
        "comment_text": X_train.values,
        "target": y_train.values
    })

    test_df = pd.DataFrame({
        "comment_text": X_test.values,
        "target": y_test.values
    })

    train_df.to_csv(
        TRAIN_OUTPUT,
        index=False
    )

    test_df.to_csv(
        TEST_OUTPUT,
        index=False
    )

    print("\n" + "=" * 70)
    print("DATASETS SAVED")
    print("=" * 70)

    print(
        f"Training data: {TRAIN_OUTPUT}"
    )

    print(
        f"Testing data : {TEST_OUTPUT}"
    )


# ============================================================
# 13. MAIN FUNCTION
# ============================================================

def main():

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_dataset(
        DATA_PATH
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_columns(df)

    # --------------------------------------------------------
    # Clean comments
    # --------------------------------------------------------

    df = remove_missing_comments(df)

    df = remove_empty_comments(df)

    # --------------------------------------------------------
    # Duplicate handling
    # --------------------------------------------------------

    df = remove_duplicate_comments(df)

    # --------------------------------------------------------
    # Binary target
    # --------------------------------------------------------

    df = create_binary_target(df)

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    show_class_distribution(df)

    # --------------------------------------------------------
    # Prepare X and y
    # --------------------------------------------------------

    X, y = prepare_features_and_target(df)

    # --------------------------------------------------------
    # Train/Test split
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = split_dataset(X, y)

    # --------------------------------------------------------
    # Verify stratification
    # --------------------------------------------------------

    check_split_distribution(
        y_train,
        y_test
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_datasets(
        X_train,
        X_test,
        y_train,
        y_test
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PHASE 2 COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print("\nImportant:")
    print(
        "TF-IDF has NOT been fitted yet."
    )

    print(
        "The test set must remain untouched "
        "until final model evaluation."
    )

    print(
        "\nNext phase:"
        "\nTF-IDF feature engineering and "
        "leakage-safe preprocessing."
    )


# ============================================================
# 14. RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()
