# src/clinicalbert_finetune.py
"""
Fine-tune Bio_ClinicalBERT on chief complaint primary classification
Uses your labeled_sample.csv

Last updated: Final version - stratification disabled, explicit final eval added
"""

import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, ClassLabel
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# Paths
BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(BASE_DIR, "data")
LABELED_FILE = os.path.join(DATA_DIR, "labeled_sample.csv")

# Model settings
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
MAX_LENGTH = 128
NUM_EPOCHS = 3
BATCH_SIZE = 16

print("Loading data...")
df = pd.read_csv(LABELED_FILE)
df = df.dropna(subset=['processed', 'Primary Classification'])

# Encode labels
le = LabelEncoder()
df['label'] = le.fit_transform(df['Primary Classification'])
num_labels = len(le.classes_)
print(f"Number of classes: {num_labels}")
print("Classes:", le.classes_)

# Show class distribution
print("\nClass distribution:")
print(df['Primary Classification'].value_counts())

# Create dataset
dataset = Dataset.from_pandas(df[['processed', 'label']])

# Convert label to ClassLabel (good practice)
class_names = le.classes_.tolist()
dataset = dataset.cast_column("label", ClassLabel(num_classes=num_labels, names=class_names))

# Split WITHOUT stratification (required due to classes with count=1)
print("Splitting dataset (stratification disabled due to classes with count=1)...")
dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = dataset['train']
eval_dataset = dataset['test']

print(f"Train size: {len(train_dataset)} | Eval size: {len(eval_dataset)}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(
        examples['processed'],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )

print("Tokenizing datasets...")
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_eval = eval_dataset.map(tokenize_function, batched=True)

# Set format for PyTorch
tokenized_train.set_format("torch", columns=["input_ids", "attention_mask", "label"])
tokenized_eval.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# Load model
print("Loading Bio_ClinicalBERT model...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    ignore_mismatched_sizes=True
)

# Training arguments
training_args = TrainingArguments(
    output_dir=os.path.join(DATA_DIR, "clinicalbert_results"),
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    fp16=True,                          # faster if GPU available
    logging_dir=os.path.join(DATA_DIR, "logs"),
    logging_steps=50,
    remove_unused_columns=False         # prevents column drop issues
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    return {"accuracy": acc, "f1_weighted": f1}

# Start training
print("Starting fine-tuning...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    compute_metrics=compute_metrics,
)

trainer.train()

# Explicit final evaluation
print("\nFinal evaluation on eval set:")
eval_results = trainer.evaluate(tokenized_eval)
print(eval_results)

# Save final model
final_save_path = os.path.join(DATA_DIR, "clinicalbert_final")
trainer.save_model(final_save_path)
tokenizer.save_pretrained(final_save_path)

print(f"Training complete. Final model saved to: {final_save_path}")