"""
Toxic Comment Detection System
Streamlit Application

Models:
- Multinomial Naive Bayes
- Logistic Regression
- Linear SVM

The app:
1. Loads the saved TF-IDF vectorizer.
2. Loads all three trained models.
3. Accepts a comment from the user.
4. Transforms the comment using the saved TF-IDF vectorizer.
5. Predicts whether the comment is toxic or non-toxic.
6. Displays predictions from all three models.
"""

import os
import joblib
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Toxic Comment Detector",
    page_icon="🛡️",
    layout="centered"
)


# ============================================================
# PATHS
# ============================================================

MODEL_DIR = "models/final_models"

VECTORIZER_PATH = os.path.join(
    MODEL_DIR,
    "tfidf_vectorizer.joblib"
)

NB_PATH = os.path.join(
    MODEL_DIR,
    "multinomial_nb.joblib"
)

LR_PATH = os.path.join(
    MODEL_DIR,
    "logistic_regression.joblib"
)

SVM_PATH = os.path.join(
    MODEL_DIR,
    "linear_svm.joblib"
)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    vectorizer = joblib.load(VECTORIZER_PATH)

    naive_bayes = joblib.load(NB_PATH)

    logistic_regression = joblib.load(LR_PATH)

    linear_svm = joblib.load(SVM_PATH)

    return (
        vectorizer,
        naive_bayes,
        logistic_regression,
        linear_svm
    )


# ============================================================
# CHECK MODEL FILES
# ============================================================

required_files = [
    VECTORIZER_PATH,
    NB_PATH,
    LR_PATH,
    SVM_PATH
]

missing_files = [
    file for file in required_files
    if not os.path.exists(file)
]


if missing_files:

    st.error("Required model files are missing.")

    st.write("Please run Phase 6 first:")

    st.code(
        "python src/phase6_save_models.py",
        language="bash"
    )

    st.write("Missing files:")

    for file in missing_files:
        st.write(f"- {file}")

    st.stop()


# ============================================================
# LOAD EVERYTHING
# ============================================================

try:

    (
        vectorizer,
        naive_bayes,
        logistic_regression,
        linear_svm
    ) = load_models()

except Exception as error:

    st.error("Unable to load the trained models.")

    st.exception(error)

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title("🛡️ Toxic Comment Detection System")

st.write(
    "Enter a comment below and the machine learning models "
    "will classify it as **Toxic** or **Non-Toxic**."
)


st.divider()


# ============================================================
# COMMENT INPUT
# ============================================================

comment = st.text_area(
    "Enter your comment:",
    height=150,
    placeholder="Type a comment here..."
)


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_comment(text):

    # Convert text into TF-IDF features
    text_tfidf = vectorizer.transform([text])

    # --------------------------------------------------------
    # Multinomial Naive Bayes
    # --------------------------------------------------------

    nb_prediction = naive_bayes.predict(text_tfidf)[0]

    nb_probability = naive_bayes.predict_proba(
        text_tfidf
    )[0][1]

    # --------------------------------------------------------
    # Logistic Regression
    # --------------------------------------------------------

    lr_prediction = logistic_regression.predict(
        text_tfidf
    )[0]

    lr_probability = logistic_regression.predict_proba(
        text_tfidf
    )[0][1]

    # --------------------------------------------------------
    # Linear SVM
    # --------------------------------------------------------

    svm_prediction = linear_svm.predict(
        text_tfidf
    )[0]

    svm_score = linear_svm.decision_function(
        text_tfidf
    )[0]

    return {
        "Naive Bayes": {
            "prediction": nb_prediction,
            "score": nb_probability
        },
        "Logistic Regression": {
            "prediction": lr_prediction,
            "score": lr_probability
        },
        "Linear SVM": {
            "prediction": svm_prediction,
            "score": svm_score
        }
    }


# ============================================================
# PREDICTION BUTTON
# ============================================================

if st.button(
    "🔍 Analyze Comment",
    type="primary",
    use_container_width=True
):

    if not comment.strip():

        st.warning("Please enter a comment first.")

    else:

        results = predict_comment(comment)

        st.subheader("Prediction Results")


        # ====================================================
        # NAIVE BAYES
        # ====================================================

        nb_result = results["Naive Bayes"]

        if nb_result["prediction"] == 1:

            st.error("Multinomial Naive Bayes: 🚨 TOXIC")

        else:

            st.success("Multinomial Naive Bayes: ✅ NON-TOXIC")

        st.write(
            f"Toxic probability: "
            f"{nb_result['score']:.2%}"
        )


        # ====================================================
        # LOGISTIC REGRESSION
        # ====================================================

        lr_result = results["Logistic Regression"]

        if lr_result["prediction"] == 1:

            st.error("Logistic Regression: 🚨 TOXIC")

        else:

            st.success("Logistic Regression: ✅ NON-TOXIC")

        st.write(
            f"Toxic probability: "
            f"{lr_result['score']:.2%}"
        )


        # ====================================================
        # LINEAR SVM
        # ====================================================

        svm_result = results["Linear SVM"]

        if svm_result["prediction"] == 1:

            st.error("Linear SVM: 🚨 TOXIC")

        else:

            st.success("Linear SVM: ✅ NON-TOXIC")

        st.write(
            f"Decision score: "
            f"{svm_result['score']:.4f}"
        )


        # ====================================================
        # MODEL AGREEMENT
        # ====================================================

        predictions = [
            nb_result["prediction"],
            lr_result["prediction"],
            svm_result["prediction"]
        ]

        toxic_votes = sum(predictions)

        st.divider()

        st.subheader("Model Agreement")

        st.write(
            f"**{toxic_votes} out of 3 models** "
            f"classified the comment as toxic."
        )

        if toxic_votes >= 2:

            st.warning(
                "⚠️ Majority of the models classify "
                "this comment as TOXIC."
            )

        else:

            st.success(
                "✅ Majority of the models classify "
                "this comment as NON-TOXIC."
            )


# ============================================================
# INFORMATION
# ============================================================

st.divider()

with st.expander("ℹ️ About this project"):

    st.write(
        """
        This Toxic Comment Detection System uses classical
        machine learning techniques for text classification.

        **Feature Engineering**
        - TF-IDF
        - Unigrams and bigrams

        **Machine Learning Models**
        - Multinomial Naive Bayes
        - Logistic Regression
        - Linear SVM

        **Classes**
        - 0 = Non-Toxic
        - 1 = Toxic

        The TF-IDF vectorizer used by the application is the
        same vectorizer fitted during the training phase.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "Toxic Comment Detection System | "
    "TF-IDF + Classical Machine Learning"
)