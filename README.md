# 🛡️ ToxiGuard — Toxic Comment Detection System

A machine learning-based **Toxic Comment Detection System** that classifies comments as **Toxic** or **Non-Toxic** using TF-IDF text features and three classical machine learning algorithms.

The project uses the **Jigsaw Toxic Comment Classification** dataset and provides an interactive Streamlit web application for testing new comments.

---

## 📌 Project Overview

Online platforms contain large amounts of user-generated content. Some comments may contain insults, threats, obscene language, or other forms of toxic behavior.

This project aims to automatically identify potentially toxic comments using traditional machine learning techniques.

The system converts text into numerical features using **TF-IDF** and then uses three different classifiers:

- Multinomial Naive Bayes
- Logistic Regression
- Linear Support Vector Machine (Linear SVM)

The predictions from all three models can be compared through the Streamlit interface.

---

## 🎯 Objectives

The main objectives of this project are:

1. Understand and preprocess the toxic comment dataset.
2. Convert text into numerical features using TF-IDF.
3. Train multiple classical machine learning models.
4. Handle class imbalance during model training.
5. Save trained models for later use.
6. Build an interactive web interface using Streamlit.
7. Allow users to enter comments and receive toxicity predictions.
8. Compare predictions from different machine learning models.

---

## 📂 Dataset

This project is based on the **Jigsaw Toxic Comment Classification Challenge** dataset.

The original dataset contains Wikipedia comments with six toxicity-related labels:

- `toxic`
- `severe_toxic`
- `obscene`
- `threat`
- `insult`
- `identity_hate`

For this project, these six labels are converted into a binary target.

### Binary Target

```text
0 = Non-Toxic
1 = Toxic
```

A comment is considered **Toxic** if at least one of the six toxicity labels is `1`.

Conceptually:

```python
target = (
    toxic
    + severe_toxic
    + obscene
    + threat
    + insult
    + identity_hate
) > 0
```

---

# 🏗️ Project Structure

```text
toxic-comment-classifier/
│
├── data/
│   ├── train.csv
│   ├── train_clean.csv
│   └── test_clean.csv
│
├── src/
│   ├── phase1_data_understanding.py
│   ├── phase2_preprocessing.py
│   ├── phase3_tfidf.py
│   ├── phase4_model_training.py
│   ├── phase5_prediction.py
│   └── phase6_save_models.py
│
├── models/
│   ├── tfidf_vectorizer.joblib
│   ├── multinomial_nb.joblib
│   ├── logistic_regression.joblib
│   ├── linear_svm.joblib
│   │
│   └── final_models/
│       ├── tfidf_vectorizer.joblib
│       ├── multinomial_nb.joblib
│       ├── logistic_regression.joblib
│       └── linear_svm.joblib
│
├── app.py
├── requirements.txt
└── README.md
```

---

# 🔄 Project Workflow

The project is divided into multiple phases.

```text
Dataset
   │
   ▼
Phase 1
Data Understanding
   │
   ▼
Phase 2
Preprocessing + Train/Test Split
   │
   ▼
Phase 3
TF-IDF Feature Engineering
   │
   ▼
Phase 4
Model Training
   │
   ├── Multinomial Naive Bayes
   ├── Logistic Regression
   └── Linear SVM
   │
   ▼
Phase 5
Prediction
   │
   ▼
Phase 6
Save & Verify Models
   │
   ▼
Streamlit Application
```

---

# 🔹 Phase 1 — Data Understanding

File:

```text
src/phase1_data_understanding.py
```

This phase performs initial exploration of the dataset.

It checks:

- Dataset shape
- Column names
- Data types
- Missing values
- Empty comments
- Duplicate comments
- Original toxicity labels
- Binary target distribution
- Example toxic comments
- Example non-toxic comments

The six original toxicity labels are also examined before creating the binary target.

Run:

```bash
python src/phase1_data_understanding.py
```

---

# 🔹 Phase 2 — Preprocessing

File:

```text
src/phase2_preprocessing.py
```

This phase prepares the data for machine learning.

### Processing steps

1. Load the dataset.
2. Validate required columns.
3. Remove missing comments.
4. Remove empty comments.
5. Remove exact duplicate comments.
6. Create the binary `target`.
7. Separate features and labels.
8. Perform an 80/20 train-test split.
9. Use stratification to preserve class proportions.
10. Save the cleaned datasets.

The split uses:

```python
random_state=42
```

and:

```python
stratify=y
```

Run:

```bash
python src/phase2_preprocessing.py
```

Output:

```text
data/train_clean.csv
data/test_clean.csv
```

---

# 🔹 Phase 3 — TF-IDF Feature Engineering

File:

```text
src/phase3_tfidf.py
```

This phase converts comments into numerical features using **TF-IDF (Term Frequency-Inverse Document Frequency)**.

The vectorizer uses:

```python
TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    max_features=100000
)
```

### Important data leakage prevention

The TF-IDF vectorizer is fitted **only on the training data**.

```python
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
```

The test data is never used to learn the vocabulary.

Run:

```bash
python src/phase3_tfidf.py
```

Generated files include:

```text
models/tfidf_vectorizer.joblib
models/X_train_tfidf.npz
models/X_test_tfidf.npz
models/y_train.npy
models/y_test.npy
```

---

# 🔹 Phase 4 — Model Training

File:

```text
src/phase4_model_training.py
```

Three machine learning models are trained.

## 1. Multinomial Naive Bayes

```python
MultinomialNB(alpha=1.0)
```

Balanced sample weights are used during training to account for class imbalance.

---

## 2. Logistic Regression

```python
LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    solver="liblinear",
    random_state=42
)
```

---

## 3. Linear SVM

```python
LinearSVC(
    class_weight="balanced",
    C=1.0,
    max_iter=5000,
    random_state=42
)
```

### Saved models

```text
models/multinomial_nb.joblib
models/logistic_regression.joblib
models/linear_svm.joblib
```

Run:

```bash
python src/phase4_model_training.py
```

---

# 🔹 Phase 5 — Prediction

File:

```text
src/phase5_prediction.py
```

This phase loads the saved vectorizer and trained models.

A new comment is transformed using:

```python
vectorizer.transform([comment])
```

The vectorizer is **not fitted again**.

The prediction system provides results from:

- Multinomial Naive Bayes
- Logistic Regression
- Linear SVM

Naive Bayes and Logistic Regression provide a toxic-class probability.

Linear SVM provides a decision score.

Run:

```bash
python src/phase5_prediction.py
```

---

# 🔹 Phase 6 — Save and Verify Models

File:

```text
src/phase6_save_models.py
```

This phase verifies that the trained models and TF-IDF vectorizer exist and can be loaded successfully.

It creates:

```text
models/final_models/
```

and stores the final model artifacts there.

Final files:

```text
models/final_models/
│
├── tfidf_vectorizer.joblib
├── multinomial_nb.joblib
├── logistic_regression.joblib
└── linear_svm.joblib
```

Run:

```bash
python src/phase6_save_models.py
```

---

# 🌐 Streamlit Application

The project includes an interactive web application:

```text
app.py
```

The application is designed as a user-friendly interface called **ToxiGuard**.

It allows users to:

- Enter a comment.
- Analyze the comment.
- View predictions from all three models.
- View toxic probability for Naive Bayes.
- View toxic probability for Logistic Regression.
- View the decision score for Linear SVM.
- Compare model predictions.
- View overall model agreement.
- Learn how the classification pipeline works.

---

# ▶️ Running the Streamlit Application

First install the required packages:

```bash
pip install -r requirements.txt
```

Then start the application using:

```bash
python -m streamlit run app.py
```

Alternatively:

```bash
streamlit run app.py
```

Do **not** start the Streamlit application using:

```bash
python app.py
```

because Streamlit requires its own application runner.

After starting the application, Streamlit will provide a local URL similar to:

```text
http://localhost:8501
```

Open that address in your browser.

---

# 📦 Requirements

The project requires:

```text
streamlit
pandas
numpy
scikit-learn
joblib
scipy
```

A `requirements.txt` file can contain:

```text
streamlit
pandas
numpy
scikit-learn
joblib
scipy
```

Install them using:

```bash
pip install -r requirements.txt
```

---

# 🧪 Example Prediction

You can enter a comment into the Streamlit application.

For example:

```text
This is a great discussion. I really enjoyed reading this.
```

The application will show the predictions from all three models.

For example, the interface may display:

```text
Multinomial Naive Bayes
✓ Non-Toxic

Logistic Regression
✓ Non-Toxic

Linear SVM
✓ Non-Toxic
```

The actual output depends on the trained models and should not be assumed in advance.

---

# ⚖️ Class Imbalance

Toxic comment datasets are typically imbalanced because non-toxic comments are more common than toxic comments.

This project therefore uses class balancing during model training.

For Logistic Regression:

```python
class_weight="balanced"
```

For Linear SVM:

```python
class_weight="balanced"
```

For Multinomial Naive Bayes, balanced sample weights are supplied during training.

This helps prevent the models from simply favoring the majority class.

---

# 🔐 Data Leakage Prevention

Data leakage is avoided throughout the project.

### Train/Test Split

The dataset is split before TF-IDF feature learning.

### TF-IDF

The vectorizer is fitted only on training comments:

```python
vectorizer.fit_transform(X_train)
```

The test data is only transformed:

```python
vectorizer.transform(X_test)
```

### Prediction

The saved vectorizer is reused for new comments:

```python
vectorizer.transform([comment])
```

It is never fitted again during prediction.

---

# 🤖 Machine Learning Models

| Model | Purpose |
|---|---|
| Multinomial Naive Bayes | Probabilistic text classifier |
| Logistic Regression | Linear classification with probabilities |
| Linear SVM | Linear margin-based classifier |

All three models use the same TF-IDF representation so their predictions can be compared fairly.

---

# 📊 Evaluation

The primary evaluation focus of the project should be **F1-score**, especially when the classes are imbalanced.

Important metrics include:

- Accuracy
- Precision
- Recall
- F1-score
- Macro F1
- Weighted F1
- Confusion Matrix

For the toxic class, precision and recall are particularly useful because a model can have high accuracy while still performing poorly on the minority toxic class.

---

# 🧩 Technologies Used

### Programming Language

- Python

### Machine Learning

- Scikit-learn

### Natural Language Processing

- TF-IDF
- Unigrams
- Bigrams

### Data Processing

- Pandas
- NumPy

### Model Persistence

- Joblib

### Web Application

- Streamlit

### Visualization

- Matplotlib

---

# 📁 Important Model Files

The application depends on these files:

```text
models/final_models/tfidf_vectorizer.joblib
models/final_models/multinomial_nb.joblib
models/final_models/logistic_regression.joblib
models/final_models/linear_svm.joblib
```

If these files are missing, run:

```bash
python src/phase4_model_training.py
```

followed by:

```bash
python src/phase6_save_models.py
```

---

# ⚠️ Limitations

This system is a machine learning classifier and should not be considered a perfect judge of human communication.

Possible limitations include:

- Context may be misunderstood.
- Sarcasm can be difficult to classify.
- Some non-toxic comments may be incorrectly classified.
- Some toxic comments may be missed.
- Language and cultural context can affect predictions.
- Model predictions depend on the training dataset.
- A decision score from Linear SVM should not be interpreted as a probability.
- The majority agreement shown in the application is a comparison of model outputs, not a separately trained ensemble model.

Therefore, predictions should be treated as automated assistance rather than definitive judgments.

---

# 🚀 Future Improvements

Possible future improvements include:

- Hyperparameter tuning
- Cross-validation
- More detailed model evaluation
- Threshold optimization
- Model explainability
- Feature importance analysis
- Better handling of sarcasm and context
- Multilingual toxicity detection
- Deployment to a cloud platform
- Monitoring model performance after deployment
- Adding a formally trained ensemble model

---

# 👩‍💻 Project Pipeline Summary

```text
Jigsaw Dataset
       ↓
Data Understanding
       ↓
Data Cleaning
       ↓
Duplicate Removal
       ↓
Binary Target Creation
       ↓
Stratified Train/Test Split
       ↓
TF-IDF
       ↓
┌───────────────────────────────┐
│                               │
│  Multinomial Naive Bayes      │
│  Logistic Regression          │
│  Linear SVM                   │
│                               │
└───────────────────────────────┘
       ↓
Model Saving
       ↓
Streamlit Application
       ↓
Toxic / Non-Toxic Prediction
```

---

# 📌 Quick Start

Clone/download the project and place the dataset at:

```text
data/train.csv
```

Then execute the phases in order:

```bash
python src/phase1_data_understanding.py
```

```bash
python src/phase2_preprocessing.py
```

```bash
python src/phase3_tfidf.py
```

```bash
python src/phase4_model_training.py
```

```bash
python src/phase5_prediction.py
```

```bash
python src/phase6_save_models.py
```

Finally start the web application:

```bash
python -m streamlit run app.py
```

---

# 🛡️ ToxiGuard

**ToxiGuard** is a classical machine-learning-based toxic comment detection system built using:

```text
TF-IDF
   +
Multinomial Naive Bayes
   +
Logistic Regression
   +
Linear SVM
   +
Streamlit
```

The project demonstrates a complete machine learning workflow from **raw text data → preprocessing → feature engineering → model training → model saving → interactive prediction**.