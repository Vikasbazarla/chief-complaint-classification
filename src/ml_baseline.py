# src/ml_baseline.py
"""
ML Baseline Training for Chief Complaint Classification
----------------------------------------------------
Uses TF-IDF + Logistic Regression for:
- Multi-label classification (using Multi-label Classification column)
- Single-label primary classification

Goal: Establish strong, interpretable baseline before moving to transformer models.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, hamming_loss
import joblib
from datetime import datetime
from collections import Counter

# ---------------------- Config ----------------------
BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE_DIR, "data")
LABELED_FILE = os.path.join(DATA_DIR, "labeled_sample.csv")

OUTPUT_DIR = os.path.join(DATA_DIR, "ml_models")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE_FROM_TEMP = 0.5  # 10% val, 10% test after 80/20 split

# ---------------------- Load & Prepare Data ----------------------
print("Loading labeled dataset...")
df = pd.read_csv(LABELED_FILE)
df = df.dropna(subset=['processed', 'Multi-label Classification', 'Primary Classification'])

print(f"Total rows after cleaning: {len(df)}")

# Multi-label target preparation
df['multi_labels'] = df['Multi-label Classification'].str.split(r'\s*,\s*')
mlb = MultiLabelBinarizer()
y_multi = mlb.fit_transform(df['multi_labels'])

# Primary target (single label)
y_primary = df['Primary Classification']

# Features (processed text)
X = df['processed']

# ---------------------- Split (NO stratification) ----------------------
print("Creating splits (stratification disabled due to classes with count=1)...")

# First split: train vs temp (80/20)
X_train, X_temp, y_multi_train, y_multi_temp, y_primary_train, y_primary_temp = train_test_split(
    X, y_multi, y_primary,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

# Second split: val vs test from temp (50/50)
X_val, X_test, y_multi_val, y_multi_test, y_primary_val, y_primary_test = train_test_split(
    X_temp, y_multi_temp, y_primary_temp,
    test_size=VAL_SIZE_FROM_TEMP,
    random_state=RANDOM_STATE
)

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# Safety check: warn if any primary class missing from val/test
train_counts = Counter(y_primary_train)
val_counts = Counter(y_primary_val)
test_counts = Counter(y_primary_test)

missing_val = [cls for cls in train_counts if val_counts[cls] == 0]
missing_test = [cls for cls in train_counts if test_counts[cls] == 0]

if missing_val:
    print(f"Warning: Rare classes missing from validation: {missing_val}")
if missing_test:
    print(f"Warning: Rare classes missing from test: {missing_test}")

# ---------------------- Vectorization ----------------------
print("Fitting TF-IDF vectorizer on train set only...")
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    stop_words=None,           # already cleaned in preprocessing
    token_pattern=r'(?u)\b\w\w+\b'
)

X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec   = vectorizer.transform(X_val)
X_test_vec  = vectorizer.transform(X_test)

print(f"Vocabulary size: {len(vectorizer.get_feature_names_out())}")

# ---------------------- Model Training & Evaluation ----------------------
print("\nTraining models...")

# 1. Multi-label model
print("→ Multi-label classifier (OneVsRest Logistic Regression)")
multi_clf = OneVsRestClassifier(
    LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        C=0.5  # regularization to reduce overfitting on major classes
    )
)
multi_clf.fit(X_train_vec, y_multi_train)

# Validation evaluation
pred_val_multi = multi_clf.predict(X_val_vec)
print("\nMulti-label Validation Report:")
print(classification_report(y_multi_val, pred_val_multi, target_names=mlb.classes_, zero_division=0))
print(f"Hamming Loss: {hamming_loss(y_multi_val, pred_val_multi):.4f}")

# 2. Primary (single-label) model
print("\n→ Primary classifier (Logistic Regression)")
primary_clf = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',
    multi_class='multinomial',
    random_state=RANDOM_STATE,
    C=0.5  # regularization to reduce overfitting on major classes
)
primary_clf.fit(X_train_vec, y_primary_train)

pred_val_primary = primary_clf.predict(X_val_vec)
print("\nPrimary Validation Report:")
print(classification_report(y_primary_val, pred_val_primary, zero_division=0))

# ---------------------- Save Artifacts ----------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

joblib.dump(vectorizer,   os.path.join(OUTPUT_DIR, f"tfidf_vectorizer_{timestamp}.joblib"))
joblib.dump(multi_clf,    os.path.join(OUTPUT_DIR, f"multi_label_model_{timestamp}.joblib"))
joblib.dump(primary_clf,  os.path.join(OUTPUT_DIR, f"primary_model_{timestamp}.joblib"))
joblib.dump(mlb,          os.path.join(OUTPUT_DIR, f"multi_label_binarizer_{timestamp}.joblib"))

print(f"\nModels and artifacts saved to: {OUTPUT_DIR}")

# Save test set predictions for later analysis
test_results = pd.DataFrame({
    'processed': X_test,
    'true_primary': y_primary_test,
    'pred_primary': primary_clf.predict(X_test_vec)
})
test_results.to_csv(os.path.join(DATA_DIR, f"test_predictions_{timestamp}.csv"), index=False)