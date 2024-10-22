# AI Music Generator

This repository contains an AI-based music generator that uses machine learning techniques to analyze and generate music based on input data. The model is designed to classify musical genres or generate new compositions by leveraging features extracted from audio tracks.

## Features
- **Music Feature Extraction**: Mel-frequency cepstral coefficients (MFCCs), chroma, spectral contrast, and more.
- **Model Training**: Trains a machine learning model to learn musical patterns.
- **Music Generation**: Generates new music sequences based on the patterns learned from the input data.
- **Genre Classification**: Classifies music into different genres based on input features.

## Technologies Used
- **Python**: For data processing and model implementation.
- **Librosa**: For feature extraction from music files.
- **Scikit-learn**: For implementing machine learning models.
- **Keras/TensorFlow** (Optional): If a deep learning approach is used for the model.
- **Streamlit**: For creating an interactive web application for user input and model predictions.

## How It Works
1. **Data Preprocessing**: The project first extracts essential features from the input music tracks. Features like MFCCs, chroma, and spectral contrast are processed to train the model.
   
2. **Model Training**: After feature extraction, a machine learning model (such as an SVM or a neural network) is trained on a labeled dataset of music tracks. The model learns to identify patterns that correspond to various genres or styles.

3. **Prediction/Generation**: Once trained, the model can either classify new music tracks by genre or generate new music sequences based on the input data.

## Installation
To get started, clone this repository and install the required dependencies:
```bash
git clone https://github.com/yashrith/ModelForge.git
cd ModelForge/AI-music-predictor
pip install -r requirements.txt

streamlit run Ai-music.py  # in the cmd of same root folder
