"""
Phase 1: Dataset Understanding
Toxic Comment Classification Project

Dataset:
Jigsaw Toxic Comment Classification Dataset

Goal:
1. Load the dataset
2. Inspect shape and columns
3. Check missing values
4. Check empty comments
5. Check duplicate comments
6. Inspect toxicity labels
7. Create a binary toxicity target
8. Analyze class distribution
9. Display sample comments
10. Visualize class distribution

This phase DOES NOT train any machine learning model.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. CONFIGURATION
# ============================================================

# Change this path if your CSV is stored somewhere else.
DATA_PATH = os.path.join("data", "train.csv")


# Original Jigsaw toxicity columns
TOXICITY_COLUMNS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]


# ============================================================
# 2. LOAD DATASET
# ============================================================

def load_dataset(file_path):
    """
    Load the Jigsaw dataset from a CSV file.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"\nDataset not found: {file_path}\n"
            "Please place the Jigsaw train.csv file inside the data/ folder."
        )

    df = pd.read_csv(file_path)

    print("=" * 70)
    print("DATASET LOADED SUCCESSFULLY")
    print("=" * 70)

    print(f"File: {file_path}")
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")

    return df


# ============================================================
# 3. BASIC DATASET INFORMATION
# ============================================================

def inspect_dataset(df):
    """
    Display basic information about the dataset.
    """

    print("\n" + "=" * 70)
    print("DATASET SHAPE")
    print("=" * 70)

    print(df.shape)

    print("\n" + "=" * 70)
    print("COLUMN NAMES")
    print("=" * 70)

    for column in df.columns:
        print("-", column)

    print("\n" + "=" * 70)
    print("DATA TYPES")
    print("=" * 70)

    print(df.dtypes)

    print("\n" + "=" * 70)
    print("FIRST 5 ROWS")
    print("=" * 70)

    print(df.head())

    print("\n" + "=" * 70)
    print("DATASET INFORMATION")
    print("=" * 70)

    df.info()


# ============================================================
# 4. CHECK MISSING VALUES
# ============================================================

def check_missing_values(df):
    """
    Check missing values in every column.
    """

    print("\n" + "=" * 70)
    print("MISSING VALUES")
    print("=" * 70)

    missing = df.isnull().sum()

    print(missing)

    print("\nTotal missing values:", missing.sum())


# ============================================================
# 5. CHECK EMPTY COMMENTS
# ============================================================

def check_empty_comments(df):
    """
    Check comments that are empty strings or contain only spaces.
    """

    print("\n" + "=" * 70)
    print("EMPTY COMMENTS")
    print("=" * 70)

    empty_comments = (
        df["comment_text"]
        .fillna("")
        .str.strip()
        .eq("")
    )

    count = empty_comments.sum()

    print("Empty comments:", count)

    if count > 0:
        print("\nExample empty rows:")
        print(df.loc[empty_comments].head())


# ============================================================
# 6. CHECK DUPLICATE COMMENTS
# ============================================================

def check_duplicates(df):
    """
    Check exact duplicate comments.

    Duplicate comments are important because if the same comment
    appears in both training and testing data, test performance
    can become artificially optimistic.
    """

    print("\n" + "=" * 70)
    print("DUPLICATE COMMENTS")
    print("=" * 70)

    duplicate_mask = df["comment_text"].duplicated(keep=False)

    duplicate_count = df["comment_text"].duplicated().sum()

    print("Duplicate comment rows:", duplicate_count)

    if duplicate_mask.sum() > 0:
        print("\nExamples of duplicated comments:")

        duplicates = (
            df.loc[duplicate_mask]
            .sort_values("comment_text")
        )

        print(duplicates.head(10))


# ============================================================
# 7. INSPECT ORIGINAL TOXICITY LABELS
# ============================================================

def inspect_original_labels(df):
    """
    Display the distribution of the six original Jigsaw labels.
    """

    print("\n" + "=" * 70)
    print("ORIGINAL JIGSAW LABEL DISTRIBUTION")
    print("=" * 70)

    for column in TOXICITY_COLUMNS:

        if column not in df.columns:
            print(f"WARNING: Column '{column}' not found.")
            continue

        count = df[column].sum()
        percentage = (count / len(df)) * 100

        print(
            f"{column:15s} : "
            f"{count:8,d} "
            f"({percentage:6.2f}%)"
        )


# ============================================================
# 8. CREATE BINARY TARGET
# ============================================================

def create_binary_target(df):
    """
    Convert the six original toxicity labels into one binary target.

    Rule:

        If ANY toxicity label is 1:
            target = 1 (Toxic)

        If ALL toxicity labels are 0:
            target = 0 (Non-toxic)
    """

    missing_columns = [
        column
        for column in TOXICITY_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "The following required toxicity columns are missing: "
            + ", ".join(missing_columns)
        )

    df = df.copy()

    # Sum all six toxicity indicators.
    toxicity_sum = df[TOXICITY_COLUMNS].sum(axis=1)

    # Convert the result to a binary target.
    df["target"] = (toxicity_sum > 0).astype(int)

    print("\n" + "=" * 70)
    print("BINARY TARGET CREATED")
    print("=" * 70)

    print("0 = Non-toxic")
    print("1 = Toxic")

    print("\nTarget distribution:")
    print(df["target"].value_counts())

    return df


# ============================================================
# 9. TARGET PERCENTAGES
# ============================================================

def show_target_distribution(df):
    """
    Display target counts and percentages.
    """

    print("\n" + "=" * 70)
    print("BINARY TARGET DISTRIBUTION")
    print("=" * 70)

    counts = df["target"].value_counts().sort_index()

    percentages = (
        df["target"]
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
    )

    distribution = pd.DataFrame({
        "Count": counts,
        "Percentage": percentages.round(2)
    })

    distribution.index = [
        "Non-toxic" if index == 0 else "Toxic"
        for index in distribution.index
    ]

    print(distribution)


# ============================================================
# 10. DISPLAY SAMPLE COMMENTS
# ============================================================

def show_sample_comments(df, number_of_samples=5):
    """
    Display toxic and non-toxic examples.
    """

    print("\n" + "=" * 70)
    print("NON-TOXIC COMMENT EXAMPLES")
    print("=" * 70)

    non_toxic = df[df["target"] == 0]["comment_text"]

    for comment in non_toxic.sample(
        min(number_of_samples, len(non_toxic)),
        random_state=42
    ):
        print("-", comment)

    print("\n" + "=" * 70)
    print("TOXIC COMMENT EXAMPLES")
    print("=" * 70)

    toxic = df[df["target"] == 1]["comment_text"]

    for comment in toxic.sample(
        min(number_of_samples, len(toxic)),
        random_state=42
    ):
        print("-", comment)


# ============================================================
# 11. VISUALIZE ORIGINAL LABEL DISTRIBUTION
# ============================================================

def plot_original_labels(df):
    """
    Plot the distribution of the six original toxicity labels.
    """

    counts = df[TOXICITY_COLUMNS].sum().sort_values(
        ascending=False
    )

    plt.figure(figsize=(10, 5))

    sns.barplot(
        x=counts.index,
        y=counts.values
    )

    plt.title("Distribution of Original Jigsaw Toxicity Labels")
    plt.xlabel("Toxicity Label")
    plt.ylabel("Number of Comments")

    plt.xticks(rotation=30)

    plt.tight_layout()
    plt.show()


# ============================================================
# 12. VISUALIZE BINARY TARGET
# ============================================================

def plot_binary_target(df):
    """
    Plot toxic vs non-toxic class distribution.
    """

    plot_data = df.copy()

    plot_data["target_label"] = plot_data["target"].map({
        0: "Non-toxic",
        1: "Toxic"
    })

    plt.figure(figsize=(7, 5))

    sns.countplot(
        data=plot_data,
        x="target_label"
    )

    plt.title("Binary Toxicity Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of Comments")

    plt.tight_layout()
    plt.show()


# ============================================================
# 13. MAIN PROGRAM
# ============================================================

def main():

    # Load dataset
    df = load_dataset(DATA_PATH)

    # Basic inspection
    inspect_dataset(df)

    # Missing values
    check_missing_values(df)

    # Empty comments
    check_empty_comments(df)

    # Duplicate comments
    check_duplicates(df)

    # Original six labels
    inspect_original_labels(df)

    # Create binary target
    df = create_binary_target(df)

    # Display binary distribution
    show_target_distribution(df)

    # Display example comments
    show_sample_comments(df)

    # Visualizations
    plot_original_labels(df)
    plot_binary_target(df)

    print("\n" + "=" * 70)
    print("PHASE 1 COMPLETED")
    print("=" * 70)

    print(
        "\nNext phase:"
        "\nData cleaning + duplicate handling + "
        "leakage-safe train/test split."
    )


if __name__ == "__main__":
    main()
