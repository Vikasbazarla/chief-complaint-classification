# 🏥 Chief Complaint Classification System  
### Hybrid NLP + Machine Learning Pipeline for Medical Text Normalization & Categorization

---

## 📌 Overview

This project is an AI-powered system designed to process and classify raw medical chief complaints into predefined clinical categories.

Medical complaints are often:

- Unstructured
- Full of abbreviations
- Contain spelling mistakes
- Include negations (e.g., "no fever")
- Contain multiple symptoms in one sentence

This system uses a **Hybrid NLP + Machine Learning approach** to handle real-world medical data at scale.

It is designed to scale to millions of healthcare records.

---

## 🚀 Key Features

✅ Abbreviation Expansion  
✅ Medical Spell Correction (SymSpell)  
✅ Synonym Normalization  
✅ Negation Detection  
✅ Rule-Based Classification (~81% coverage achieved)  
✅ Machine Learning Fallback (TF-IDF + Logistic Regression)  
✅ Multi-label Classification Support  
✅ ClinicalBERT Fine-tuning Capability  
✅ Streamlit Interactive Demo  

---

## 🧠 System Architecture

```
Raw Complaint Text
        ↓
Text Cleaning
        ↓
Abbreviation Expansion
        ↓
Spell Correction
        ↓
Normalization
        ↓
Negation Handling
        ↓
Rule-Based Labeling
        ↓ (if unmatched)
ML Model Prediction
        ↓
Final Categorized Output
```

---

## 🛠 Technology Stack

- Python 3.x
- spaCy
- SymSpell
- RapidFuzz
- scikit-learn
- Pandas
- Streamlit
- ClinicalBERT (Optional Deep Learning Enhancement)

---

## 📂 Project Structure

```
chief_complaint_classification/

│
├── data/
│   ├── abbreviations.json
│   ├── normalization.json
│   ├── rules.json
│   ├── stopwords_medical.txt
│   ├── medical_terms.txt
│   ├── raw_sample.csv
│   ├── processed_sample.csv
│   ├── labeled_sample.csv
│
├── src/
│   ├── preprocessing.py
│   ├── labeling.py
│   ├── ml_baseline.py
│   ├── clinicalbert_finetune.py
│   ├── process_dataset.py
│   ├── analyze_others.py
│   ├── demo_app.py
│
├── requirements.txt
└── README.md
```

---

## 🔍 Detailed Pipeline Explanation

### 1️⃣ Abbreviation Expansion

Medical text contains shortcuts like:

| Abbreviation | Expanded Form |
|--------------|--------------|
| c/o | complains of |
| sob | shortness of breath |
| bp | blood pressure |

These are expanded using `abbreviations.json`.

---

### 2️⃣ Spell Correction

Uses **SymSpell** with:

- English frequency dictionary
- Custom medical terms dictionary

Example:

| Incorrect | Correct |
|----------|----------|
| fevr | fever |
| diahrea | diarrhea |

---

### 3️⃣ Text Normalization

Maps synonyms to standard medical terminology.

Example:

| Raw Text | Normalized |
|----------|-----------|
| loose stools | diarrhea |
| burning micturition | urinary pain |

Normalization improves classification consistency.

---

### 4️⃣ Negation Detection

Handles patterns such as:

- no fever
- denies chest pain
- not vomiting

Ensures symptoms are not falsely classified.

---

### 5️⃣ Rule-Based Classification

High-precision keyword matching system.

Example rules:

- fever → Fever Category
- cough → Respiratory
- itching → Dermatology

Coverage achieved: **~81% of complaints**

---

### 6️⃣ Machine Learning Fallback

For unmatched complaints:

- TF-IDF Vectorization
- Logistic Regression Classifier
- Multi-label support

Ensures coverage for complex and unseen cases.

---

### 7️⃣ ClinicalBERT (Advanced Option)

Fine-tuning supported for:

- Context-aware classification
- Higher accuracy on large datasets
- Transformer-based modeling

Model weights are excluded due to GitHub size limits.

---

## 📊 Example

### Input:

```
Patient c/o fever and loose stools since 2 days
```

### After Processing:

```
Expanded: complains of fever and diarrhea
```

### Output Categories:

```
Fever
Gastrointestinal Disorder
```

---

## ▶️ How To Run

### Step 1: Clone Repository

```
git clone https://github.com/YOUR_USERNAME/chief-complaint-classification.git
cd chief-complaint-classification
```

---

### Step 2: Create Virtual Environment (Recommended)

Windows:

```
python -m venv venv
venv\Scripts\activate
```

Mac/Linux:

```
python3 -m venv venv
source venv/bin/activate
```

---

### Step 3: Install Dependencies

```
pip install -r requirements.txt
```

---

### Step 4: Run Streamlit Demo

```
cd src
streamlit run demo_app.py
```

Open browser at:

```
http://localhost:8501
```

---

## 📈 Performance

- Rule-Based Coverage: ~81%
- Remaining cases handled by ML fallback
- Designed to scale for 2M+ medical records
- Modular architecture for extensibility

---

## ⚠️ Large Model Notice

Trained ML models and ClinicalBERT checkpoints are not included in this repository due to GitHub size limitations.

To retrain models:

```
python src/clinicalbert_finetune.py
```

or

```
python src/ml_baseline.py
```

---

## 🏗 Future Enhancements

- FastAPI REST deployment
- Docker containerization
- Production inference pipeline
- Deep learning multi-label classification
- Real-time hospital EMR integration
- Model evaluation dashboard

---

## 🎯 Real-World Use Cases

- Hospital triage automation
- Electronic Medical Record standardization
- Healthcare analytics platforms
- Clinical NLP research
- Large-scale patient complaint analysis

---

## 👨‍💻 Author

**Vikas Bazarla**  
AI/ML Engineer | NLP Enthusiast  
Focused on scalable healthcare AI systems

---

## 📜 License

This project is intended for research, educational, and portfolio purposes.
