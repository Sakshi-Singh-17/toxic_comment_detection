# ============================================================
# PHASE 4: MODEL EVALUATION
# Toxic Comment Detection
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import sparse

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# 1. PATHS
# ============================================================

TEST_TFIDF_PATH = "models/X_test_tfidf.npz"
Y_TEST_PATH = "models/y_test.npy"

MODEL_DIR = "models"

RESULTS_PATH = "models/model_comparison.csv"


# ============================================================
# 2. LOAD TEST DATA
# ============================================================

print("=" * 70)
print("PHASE 4: MODEL EVALUATION")
print("=" * 70)

print("\nLoading test data...")

X_test = sparse.load_npz(TEST_TFIDF_PATH)
y_test = np.load(Y_TEST_PATH)

print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")


# ============================================================
# 3. LOAD TRAINED MODELS
# ============================================================

print("\n" + "=" * 70)
print("LOADING TRAINED MODELS")
print("=" * 70)

model_paths = {
    "Multinomial Naive Bayes":
        os.path.join(MODEL_DIR, "multinomial_nb.joblib"),

    "Logistic Regression":
        os.path.join(MODEL_DIR, "logistic_regression.joblib"),

    "Linear SVM":
        os.path.join(MODEL_DIR, "linear_svm.joblib")
}


models = {}

for model_name, model_path in model_paths.items():

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"\nModel file not found:\n{model_path}\n\n"
            "Run phase3_train_models.py first."
        )

    models[model_name] = joblib.load(model_path)

    print(f"Loaded: {model_name}")


# ============================================================
# 4. EVALUATE MODELS
# ============================================================

results = []


for model_name, model in models.items():

    print("\n")
    print("=" * 70)
    print(f"EVALUATING: {model_name}")
    print("=" * 70)

    # --------------------------------------------------------
    # Make predictions
    # --------------------------------------------------------

    y_pred = model.predict(X_test)

    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    toxic_precision = precision_score(
        y_test,
        y_pred,
        pos_label=1,
        zero_division=0
    )

    toxic_recall = recall_score(
        y_test,
        y_pred,
        pos_label=1,
        zero_division=0
    )

    toxic_f1 = f1_score(
        y_test,
        y_pred,
        pos_label=1,
        zero_division=0
    )

    non_toxic_f1 = f1_score(
        y_test,
        y_pred,
        pos_label=0,
        zero_division=0
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    results.append({
        "Model": model_name,
        "Toxic Precision": toxic_precision,
        "Toxic Recall": toxic_recall,
        "Toxic F1": toxic_f1,
        "Non-Toxic F1": non_toxic_f1,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1,
        "Accuracy": accuracy
    })

    # --------------------------------------------------------
    # Classification report
    # --------------------------------------------------------

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=[
                "Non-Toxic",
                "Toxic"
            ],
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    print("Confusion Matrix:")
    print(cm)

    tn, fp, fn, tp = cm.ravel()

    print("\nConfusion Matrix Interpretation:")
    print(f"TN - Correct Non-Toxic: {tn}")
    print(f"FP - Non-Toxic predicted as Toxic: {fp}")
    print(f"FN - Toxic predicted as Non-Toxic: {fn}")
    print(f"TP - Correct Toxic: {tp}")

    # --------------------------------------------------------
    # Plot confusion matrix
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(6, 5))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Non-Toxic",
            "Toxic"
        ]
    )

    display.plot(
        ax=ax,
        values_format="d"
    )

    ax.set_title(
        f"Confusion Matrix - {model_name}"
    )

    plt.tight_layout()

    safe_name = (
        model_name
        .lower()
        .replace(" ", "_")
    )

    plt.savefig(
        f"models/confusion_matrix_{safe_name}.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()


# ============================================================
# 5. CREATE COMPARISON TABLE
# ============================================================

print("\n")
print("=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

results_df = pd.DataFrame(results)


# Keep required column order

results_df = results_df[
    [
        "Model",
        "Toxic Precision",
        "Toxic Recall",
        "Toxic F1",
        "Non-Toxic F1",
        "Macro F1",
        "Weighted F1",
        "Accuracy"
    ]
]


# Sort by Macro F1

results_df = results_df.sort_values(
    by="Macro F1",
    ascending=False
)


# ============================================================
# 6. DISPLAY RESULTS
# ============================================================

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 7. SAVE RESULTS
# ============================================================

results_df.to_csv(
    RESULTS_PATH,
    index=False
)

print("\nComparison saved to:")
print(RESULTS_PATH)


# ============================================================
# 8. IDENTIFY BEST MODEL
# ============================================================

best_model = results_df.iloc[0]

print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

print(f"Model    : {best_model['Model']}")
print(f"Macro F1 : {best_model['Macro F1']:.4f}")


# ============================================================
# 9. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("PHASE 4 COMPLETED")
print("=" * 70)

print("\nEvaluation included:")
print("✓ Accuracy")
print("✓ Toxic Precision")
print("✓ Toxic Recall")
print("✓ Toxic F1")
print("✓ Non-Toxic F1")
print("✓ Macro F1")
print("✓ Weighted F1")
print("✓ Confusion Matrix")
print("✓ Model Comparison")

print("\nNo model was trained in Phase 4.")
print("Models were loaded from Phase 3.")