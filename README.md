# EireneML

A supervised natural language processing (NLP) model that classifies user input into predefined emotional intent categories.

---

## Overview
EireneML is a small-scale NLP project that implements an **intent classification chatbot**. The project focuses on demonstrating the
complete machine learning workflow, from text preprocessing and feature engineering to model training and interactive inference.

---

## Project Structure
```text
EireneML/
│
├── intents.json # Intent definitions and training data
├── training.py # Model training pipeline
├── eirene.py # Inference and interactive chatbot loop
├── words.pkl # Serialized vocabulary
├── classes.pkl # Serialized intent labels
├── eirene.h5 # Trained neural network model
└── README.md
```

---

## Data Format
Training data is stored in `intents.json`. Each intent consists of:

- a **tag** representing the intent label,Training


Example:
```json
{
  "tag": "greeting",
  "patterns": ["Hi", "Hello", "Good morning"],
  "responses": ["Hello!", "Hi there!", "Good to see you!"]
}
```
## Text Preprocessing
Text preprocessing is performed using NLTK and includes:
- tokenization,
- lowercasing,
- lemmatization,
- basic punctuation filtering.

The same preprocessing logic is applied during both training and inference to ensure vocabulary consistency.

## Feature Representation
Input sentences are converted into Bag-of-Words vectors:
each vocabulary term corresponds to a feature, a feature value of 1 indicates word presence, and a feature value of 0 indicates absence.


## Training
To train the model:

python training.py

This script processes the training data, builds the vocabulary and class labels, trains the neural network, and saves the trained model and preprocessing artifacts.

Note: The model can be trained on almost any modern computer since the dataset is very small. If training feels slow, consider reducing the number of epochs from 200 to 50–100,
and increasing the batch size from 5 to 8–32. This typically reduces training time with minimal impact on performance.
