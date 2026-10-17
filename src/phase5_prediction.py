# ============================================================
# PHASE 5: TOXIC COMMENT PREDICTION
# Toxic Comment Classification Project
#
# This phase:
# 1. Loads the saved TF-IDF vectorizer
# 2. Loads the three trained models
# 3. Accepts a new comment
# 4. Converts the comment into TF-IDF features
# 5. Generates predictions from all three models
#
# IMPORTANT:
# - No model training happens here.
# - No test data is used.
# - The saved TF-IDF vectorizer from Phase 3 is used.
# ============================================================

import os
import joblib


# ============================================================
# 1. PATHS
# ============================================================

MODEL_DIR = "models"

VECTORIZER_PATH = os.path.join(
    MODEL_DIR,
    "tfidf_vectorizer.joblib"
)

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
# 2. LOAD SAVED OBJECTS
# ============================================================

def load_models():
    """
    Load the TF-IDF vectorizer and all three trained models.
    """

    print("=" * 70)
    print("LOADING SAVED MODELS")
    print("=" * 70)

    required_files = [
        VECTORIZER_PATH,
        NB_MODEL_PATH,
        LR_MODEL_PATH,
        SVM_MODEL_PATH
    ]

    for file_path in required_files:

        if not os.path.exists(file_path):

            raise FileNotFoundError(
                f"\nRequired file not found:\n"
                f"{file_path}\n\n"
                "Make sure Phase 3 and Phase 4 "
                "have been completed."
            )

    # Load TF-IDF vectorizer
    vectorizer = joblib.load(
        VECTORIZER_PATH
    )

    # Load trained models
    nb_model = joblib.load(
        NB_MODEL_PATH
    )

    lr_model = joblib.load(
        LR_MODEL_PATH
    )

    svm_model = joblib.load(
        SVM_MODEL_PATH
    )

    print("TF-IDF vectorizer loaded.")
    print("Multinomial Naive Bayes loaded.")
    print("Logistic Regression loaded.")
    print("Linear SVM loaded.")

    return (
        vectorizer,
        nb_model,
        lr_model,
        svm_model
    )


# ============================================================
# 3. PREPROCESS NEW COMMENT
# ============================================================

def transform_comment(
    comment,
    vectorizer
):
    """
    Transform a new comment using the SAME TF-IDF
    vectorizer that was fitted during Phase 3.

    The vectorizer is NOT fitted again.
    """

    if not isinstance(comment, str):

        raise ValueError(
            "Comment must be a string."
        )

    comment = comment.strip()

    if not comment:

        raise ValueError(
            "Comment cannot be empty."
        )

    # IMPORTANT:
    # transform() is used instead of fit_transform().
    #
    # This ensures that the vocabulary and IDF values
    # learned during Phase 3 remain unchanged.

    comment_tfidf = vectorizer.transform(
        [comment]
    )

    return comment_tfidf


# ============================================================
# 4. PREDICT WITH ALL MODELS
# ============================================================

def predict_comment(
    comment,
    vectorizer,
    nb_model,
    lr_model,
    svm_model
):
    """
    Generate predictions from all three models.
    """

    # Transform comment into TF-IDF
    comment_tfidf = transform_comment(
        comment,
        vectorizer
    )

    # --------------------------------------------------------
    # Multinomial Naive Bayes
    # --------------------------------------------------------

    nb_prediction = nb_model.predict(
        comment_tfidf
    )[0]

    nb_probability = nb_model.predict_proba(
        comment_tfidf
    )[0]

    # --------------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------------

    lr_prediction = lr_model.predict(
        comment_tfidf
    )[0]

    lr_probability = lr_model.predict_proba(
        comment_tfidf
    )[0]

    # --------------------------------------------------------
    # Linear SVM
    # --------------------------------------------------------

    svm_prediction = svm_model.predict(
        comment_tfidf
    )[0]

    # LinearSVC does NOT provide predict_proba().
    #
    # decision_function() gives a decision score.
    # It should NOT be described as a probability.

    svm_score = svm_model.decision_function(
        comment_tfidf
    )[0]

    return {
        "Multinomial Naive Bayes": {
            "prediction": int(nb_prediction),
            "toxic_probability": float(
                nb_probability[1]
            )
        },

        "Logistic Regression": {
            "prediction": int(lr_prediction),
            "toxic_probability": float(
                lr_probability[1]
            )
        },

        "Linear SVM": {
            "prediction": int(svm_prediction),
            "decision_score": float(
                svm_score
            )
        }
    }


# ============================================================
# 5. CONVERT LABEL TO TEXT
# ============================================================

def label_to_text(label):
    """
    Convert binary prediction to a readable label.

    0 = Non-Toxic
    1 = Toxic
    """

    if label == 1:
        return "TOXIC"

    return "NON-TOXIC"


# ============================================================
# 6. DISPLAY PREDICTIONS
# ============================================================

def display_predictions(
    comment,
    predictions
):
    """
    Display predictions in a readable format.
    """

    print("\n" + "=" * 70)
    print("PREDICTION RESULT")
    print("=" * 70)

    print("\nComment:")
    print(comment)

    # --------------------------------------------------------
    # Multinomial Naive Bayes
    # --------------------------------------------------------

    nb_result = predictions[
        "Multinomial Naive Bayes"
    ]

    print("\nMultinomial Naive Bayes:")
    print(
        f"Prediction: "
        f"{label_to_text(nb_result['prediction'])}"
    )

    print(
        f"Toxic probability: "
        f"{nb_result['toxic_probability']:.4f}"
    )

    # --------------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------------

    lr_result = predictions[
        "Logistic Regression"
    ]

    print("\nLogistic Regression:")
    print(
        f"Prediction: "
        f"{label_to_text(lr_result['prediction'])}"
    )

    print(
        f"Toxic probability: "
        f"{lr_result['toxic_probability']:.4f}"
    )

    # --------------------------------------------------------
    # Linear SVM
    # --------------------------------------------------------

    svm_result = predictions[
        "Linear SVM"
    ]

    print("\nLinear SVM:")
    print(
        f"Prediction: "
        f"{label_to_text(svm_result['prediction'])}"
    )

    print(
        f"Decision score: "
        f"{svm_result['decision_score']:.4f}"
    )

    print("\n" + "=" * 70)


# ============================================================
# 7. INTERACTIVE PREDICTION
# ============================================================

def interactive_prediction(
    vectorizer,
    nb_model,
    lr_model,
    svm_model
):
    """
    Continuously accept comments from the user.
    """

    print("\n" + "=" * 70)
    print("INTERACTIVE TOXIC COMMENT DETECTOR")
    print("=" * 70)

    print(
        "\nEnter a comment to classify."
    )

    print(
        "Type 'exit' or 'quit' to stop."
    )

    while True:

        print("\n" + "-" * 70)

        comment = input(
            "Enter comment: "
        ).strip()

        if comment.lower() in {
            "exit",
            "quit"
        }:

            print(
                "\nPrediction program closed."
            )

            break

        if not comment:

            print(
                "Please enter a comment."
            )

            continue

        try:

            predictions = predict_comment(
                comment,
                vectorizer,
                nb_model,
                lr_model,
                svm_model
            )

            display_predictions(
                comment,
                predictions
            )

        except Exception as error:

            print(
                f"\nPrediction error: {error}"
            )


# ============================================================
# 8. MAIN FUNCTION
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("PHASE 5: TOXIC COMMENT PREDICTION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load vectorizer and trained models
    # --------------------------------------------------------

    (
        vectorizer,
        nb_model,
        lr_model,
        svm_model
    ) = load_models()

    # --------------------------------------------------------
    # Start interactive prediction
    # --------------------------------------------------------

    interactive_prediction(
        vectorizer,
        nb_model,
        lr_model,
        svm_model
    )


# ============================================================
# 9. RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()