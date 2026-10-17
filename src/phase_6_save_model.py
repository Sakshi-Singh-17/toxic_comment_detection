"""
Phase 6: Save and Verify Trained Models

This script:
1. Loads the trained models from Phase 4.
2. Loads the TF-IDF vectorizer from Phase 3.
3. Verifies that all required model files exist and can be loaded.
4. Creates a final models directory if needed.
5. Saves/copies the final model artifacts into the final_models folder.
6. Prints a summary of the saved files.

Models:
- Multinomial Naive Bayes
- Logistic Regression
- Linear SVM

Important:
- No test data is used.
- No model is retrained.
- No evaluation is performed here.
"""

import os
import shutil
import joblib


# ============================================================
# 1. PROJECT PATHS
# ============================================================

MODELS_DIR = "models"
FINAL_MODELS_DIR = "models/final_models"


# ============================================================
# 2. MODEL FILES
# ============================================================

MODEL_FILES = {
    "TF-IDF Vectorizer": "tfidf_vectorizer.joblib",
    "Multinomial Naive Bayes": "multinomial_nb.joblib",
    "Logistic Regression": "logistic_regression.joblib",
    "Linear SVM": "linear_svm.joblib",
}


# ============================================================
# 3. CREATE FINAL MODEL DIRECTORY
# ============================================================

os.makedirs(FINAL_MODELS_DIR, exist_ok=True)

print("=" * 70)
print("PHASE 6: SAVE AND VERIFY MODELS")
print("=" * 70)


# ============================================================
# 4. CHECK REQUIRED FILES
# ============================================================

print("\nChecking required model files...")

missing_files = []

for model_name, filename in MODEL_FILES.items():

    source_path = os.path.join(MODELS_DIR, filename)

    if os.path.exists(source_path):
        print(f"[OK] {model_name}: {source_path}")
    else:
        print(f"[MISSING] {model_name}: {source_path}")
        missing_files.append(filename)


# ============================================================
# 5. STOP IF FILES ARE MISSING
# ============================================================

if missing_files:

    print("\nERROR: Some required files are missing.")

    print("\nMissing files:")
    for filename in missing_files:
        print(f" - {filename}")

    print("\nPlease run these phases first:")
    print("1. python src/phase3_tfidf.py")
    print("2. python src/phase4_model_training.py")

    raise FileNotFoundError(
        "Required model files are missing. "
        "Run Phase 3 and Phase 4 first."
    )


# ============================================================
# 6. VERIFY THAT MODELS CAN BE LOADED
# ============================================================

print("\nVerifying model files...")

loaded_objects = {}

for model_name, filename in MODEL_FILES.items():

    source_path = os.path.join(MODELS_DIR, filename)

    try:

        loaded_object = joblib.load(source_path)

        loaded_objects[model_name] = loaded_object

        print(f"[VERIFIED] {model_name}")

    except Exception as error:

        print(f"[ERROR] Could not load {model_name}")
        print(f"Reason: {error}")

        raise


# ============================================================
# 7. SAVE FINAL COPIES
# ============================================================

print("\nSaving final model artifacts...")

for model_name, filename in MODEL_FILES.items():

    source_path = os.path.join(MODELS_DIR, filename)
    destination_path = os.path.join(FINAL_MODELS_DIR, filename)

    shutil.copy2(source_path, destination_path)

    print(f"[SAVED] {model_name}")
    print(f"       -> {destination_path}")


# ============================================================
# 8. DISPLAY FINAL FILES
# ============================================================

print("\n" + "=" * 70)
print("FINAL MODEL FILES")
print("=" * 70)

for filename in sorted(os.listdir(FINAL_MODELS_DIR)):

    file_path = os.path.join(FINAL_MODELS_DIR, filename)

    if os.path.isfile(file_path):

        file_size = os.path.getsize(file_path)

        print(f"{filename:<35} {file_size:,} bytes")


# ============================================================
# 9. COMPLETION MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("PHASE 6 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nFinal artifacts are stored in:")
print(f"  {FINAL_MODELS_DIR}")

print("\nSaved artifacts:")

for model_name in MODEL_FILES:
    print(f"  - {model_name}")

print("\nThese files can now be used by the prediction application.")
print("=" * 70)
