
# SomaliAccentClassification

**SomaliAccentClassification** is a machine learning project designed to detect Somali accents in spoken English using audio features extracted from `.wav` files. The model is trained primarily on Somali speakers and uses augmented data to handle the lack of non-Somali recordings.

---

## 📁 Project Structure

```

SomaliAccentClassification/
│
├── data/
│   ├── features/
│   │   ├── somali\_features.csv
│   │   └── nonsomali\_features.csv
│   ├── nonsomali\_augmented/
│   │   └── \[augmented non-Somali .wav files]
│   └── somali\_processed/
│       └── \[processed Somali .wav files]
│
├── models/
│   └── svm_model.joblib
│   └── scaler.joblib
│
├── scripts/
│   ├── augmentation\_scripts.py
│   ├── feature\_extraction.py
│   ├── preprocessing.py
│   ├── train\_model.py
│   └── predict.py
│
└── README.md

````

---

## 🧠 Project Goal

To build a machine learning model that classifies whether an English speaker has a Somali accent. This can help with linguistic studies, dialect recognition, or voice authentication in multilingual environments.

---

## 🛠️ Key Features

- **Data Augmentation** for limited non-Somali recordings.
- **Feature Extraction** using MFCCs.
- **SVM Classifier** for binary classification.
- **Prediction script** for real-time testing with single `.wav` files.

---

## 📜 How to Use

### 1. Preprocess and Extract Features

```bash
python scripts/feature_extraction.py
````

### 2. Train the Model

```bash
python scripts/train_model.py
```

### 3. Predict a Single File

```bash
python scripts/predict.py --file path/to/your/file.wav
```

---

## 🧪 Requirements

* Python 3.8+
* `librosa`
* `pandas`
* `numpy`
* `scikit-learn`
* `joblib`

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## 🔬 Author

* 👤 **Abdalla Jeylani**
* 📍 GitHub: [Jeylani-2526](https://github.com/Jeylani-2526)


