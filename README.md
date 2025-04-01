# Somali Accent Classification

## Overview
SomaliAccentClassification is a machine learning project aimed at distinguishing Somali speakers from non-Somali speakers based on their pronunciation of specific consonants. The project utilizes feature extraction techniques and ML-based classification models to analyze speech patterns and predict speaker origin.

## Features
- **Data Collection**: Voice recordings from Somali and non-Somali speakers.
- **Preprocessing**: Resampling, trimming, noise reduction, and normalization.
- **Feature Extraction**: MFCCs, chroma features, spectral centroid, spectral bandwidth, ZCR, and RMSE.
- **ML Classification**: Supervised learning models for speaker classification.
- **Evaluation Metrics**: Accuracy, precision, recall, and F1-score.

## Repository Structure
```
SomaliAccentClassification/
│-- data/                   # Processed voice recordings
│-- notebooks/              # Jupyter notebooks for data analysis & model training
│-- models/                 # Trained ML models
│-- scripts/                # Feature extraction & classification scripts
│-- README.md               # Project documentation
│-- requirements.txt        # Dependencies
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/SomaliAccentClassification.git
   ```
2. Navigate to the project directory:
   ```bash
   cd SomaliAccentClassification
   ```
3. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### Feature Extraction
Run the feature extraction script to generate the dataset for classification:
```bash
python scripts/extract_features.py
```

### Train the Model
To train the ML classifier, execute:
```bash
python scripts/train_model.py
```

### Predict Speaker Category
To classify a new audio file:
```bash
python scripts/predict.py --input path/to/audio.wav
```

## Future Enhancements
- Improve model accuracy with hyperparameter tuning
- Expand dataset with additional speaker samples
- Implement deep learning-based classification models
- Deploy as a web application

## Contributors
- **Abdalla** – Lead Developer & Researcher

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

