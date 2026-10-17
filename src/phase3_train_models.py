# ============================================================
# PHASE 4: MODEL TRAINING
# Toxic Comment Classification Project
#
# Models:
# 1. Multinomial Naive Bayes
# 2. Logistic Regression
# 3. Linear SVM
#
# IMPORTANT:
# - Only TRAINING data is used here.
# - TEST data is NOT used for model training.
# - Model evaluation will be done in Phase 5.
# ============================================================

import os
import joblib
import numpy as np

from scipy import sparse

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight


# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL_DIR = "models"

TRAIN_TFIDF_PATH = os.path.join(
    MODEL_DIR,
    "X_train_tfidf.npz"
)

TRAIN_TARGET_PATH = os.path.join(
    MODEL_DIR,
    "y_train.npy"
)


# Output model paths

NB_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "multinomial_nb.joblib"
)

LR_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "logistic_regression.joblib"
)

SVM_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "linear_svm.joblib"
)


# ============================================================
# 2. LOAD TRAINING DATA
# ============================================================

def load_training_data():
    """
    Load the TF-IDF training matrix and training target.

    Only training data is loaded because Phase 4 is
    responsible for model training.
    """

    print("=" * 70)
    print("LOADING TRAINING DATA")
    print("=" * 70)

    if not os.path.exists(TRAIN_TFIDF_PATH):
        raise FileNotFoundError(
            f"Training TF-IDF file not found:\n"
            f"{TRAIN_TFIDF_PATH}\n\n"
            "Run Phase 3 first."
        )

    if not os.path.exists(TRAIN_TARGET_PATH):
        raise FileNotFoundError(
            f"Training target file not found:\n"
            f"{TRAIN_TARGET_PATH}\n\n"
            "Run Phase 3 first."
        )

    X_train = sparse.load_npz(
        TRAIN_TFIDF_PATH
    )

    y_train = np.load(
        TRAIN_TARGET_PATH
    )

    print(
        f"Training TF-IDF shape: {X_train.shape}"
    )

    print(
        f"Training target shape: {y_train.shape}"
    )

    return X_train, y_train


# ============================================================
# 3. CHECK TRAINING CLASS DISTRIBUTION
# ============================================================

def inspect_training_classes(y_train):
    """
    Display the training class distribution.
    """

    print("\n" + "=" * 70)
    print("TRAINING CLASS DISTRIBUTION")
    print("=" * 70)

    unique_classes, counts = np.unique(
        y_train,
        return_counts=True
    )

    for class_label, count in zip(
        unique_classes,
        counts
    ):

        if class_label == 0:
            class_name = "Non-Toxic"
        else:
            class_name = "Toxic"

        percentage = (
            count / len(y_train)
        ) * 100

        print(
            f"{class_name:10s} ({class_label}): "
            f"{count:,} "
            f"({percentage:.2f}%)"
        )


# ============================================================
# 4. TRAIN MULTINOMIAL NAIVE BAYES
# ============================================================

def train_multinomial_nb(
    X_train,
    y_train
):
    """
    Train Multinomial Naive Bayes.

    MultinomialNB does not use class_weight.
    Therefore balanced sample weights are calculated
    from the training labels only.
    """

    print("\n" + "=" * 70)
    print("TRAINING MULTINOMIAL NAIVE BAYES")
    print("=" * 70)

    # Calculate balanced sample weights
    sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train
    )

    model = MultinomialNB(
        alpha=1.0
    )

    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights
    )

    print(
        "Multinomial Naive Bayes training completed."
    )

    return model


# ============================================================
# 5. TRAIN LOGISTIC REGRESSION
# ============================================================

def train_logistic_regression(
    X_train,
    y_train
):
    """
    Train Logistic Regression.

    class_weight="balanced" is used because
    the toxicity dataset is imbalanced.
    """

    print("\n" + "=" * 70)
    print("TRAINING LOGISTIC REGRESSION")
    print("=" * 70)

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="liblinear",
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    print(
        "Logistic Regression training completed."
    )

    return model


# ============================================================
# 6. TRAIN LINEAR SVM
# ============================================================

def train_linear_svm(
    X_train,
    y_train
):
    """
    Train Linear Support Vector Machine.

    LinearSVC is appropriate for large sparse
    TF-IDF text classification.
    """

    print("\n" + "=" * 70)
    print("TRAINING LINEAR SVM")
    print("=" * 70)

    model = LinearSVC(
        class_weight="balanced",
        C=1.0,
        max_iter=5000,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    print(
        "Linear SVM training completed."
    )

    return model


# ============================================================
# 7. SAVE TRAINED MODELS
# ============================================================

def save_models(
    nb_model,
    lr_model,
    svm_model
):
    """
    Save all three trained models using Joblib.
    """

    print("\n" + "=" * 70)
    print("SAVING TRAINED MODELS")
    print("=" * 70)

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    # Save Multinomial Naive Bayes
    joblib.dump(
        nb_model,
        NB_MODEL_PATH
    )

    print(
        f"Multinomial NB saved:\n"
        f"{NB_MODEL_PATH}"
    )

    # Save Logistic Regression
    joblib.dump(
        lr_model,
        LR_MODEL_PATH
    )

    print(
        f"\nLogistic Regression saved:\n"
        f"{LR_MODEL_PATH}"
    )

    # Save Linear SVM
    joblib.dump(
        svm_model,
        SVM_MODEL_PATH
    )

    print(
        f"\nLinear SVM saved:\n"
        f"{SVM_MODEL_PATH}"
    )


# ============================================================
# 8. VERIFY SAVED MODELS
# ============================================================

def verify_models():
    """
    Reload the saved models to make sure
    they were saved correctly.
    """

    print("\n" + "=" * 70)
    print("VERIFYING SAVED MODELS")
    print("=" * 70)

    model_paths = {
        "Multinomial Naive Bayes":
            NB_MODEL_PATH,

        "Logistic Regression":
            LR_MODEL_PATH,

        "Linear SVM":
            SVM_MODEL_PATH
    }

    for model_name, model_path in model_paths.items():

        if not os.path.exists(model_path):

            raise FileNotFoundError(
                f"Model was not saved correctly:\n"
                f"{model_path}"
            )

        loaded_model = joblib.load(
            model_path
        )

        print(
            f"{model_name}: "
            f"loaded successfully."
        )


# ============================================================
# 9. MAIN FUNCTION
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("PHASE 4: MODEL TRAINING")
    print("TOXIC COMMENT CLASSIFICATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load training data
    # --------------------------------------------------------

    X_train, y_train = load_training_data()

    # --------------------------------------------------------
    # Inspect class distribution
    # --------------------------------------------------------

    inspect_training_classes(
        y_train
    )

    # --------------------------------------------------------
    # Train Multinomial Naive Bayes
    # --------------------------------------------------------

    nb_model = train_multinomial_nb(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Train Logistic Regression
    # --------------------------------------------------------

    lr_model = train_logistic_regression(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Train Linear SVM
    # --------------------------------------------------------

    svm_model = train_linear_svm(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Save models
    # --------------------------------------------------------

    save_models(
        nb_model,
        lr_model,
        svm_model
    )

    # --------------------------------------------------------
    # Verify models
    # --------------------------------------------------------

    verify_models()

    # --------------------------------------------------------
    # Completion message
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("PHASE 4 COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print("\nModels trained:")
    print("1. Multinomial Naive Bayes")
    print("2. Logistic Regression")
    print("3. Linear SVM")

    print("\nModels saved in:")
    print("models/")

    print("\nNext phase:")
    print("Phase 5 - Model Evaluation")


# ============================================================
# 10. RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()