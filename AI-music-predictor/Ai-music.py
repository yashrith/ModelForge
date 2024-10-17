import streamlit as st
import numpy as np
import pandas as pd
import pickle
import librosa

def process_audio_file(file):
    # Load the audio file with librosa
    audio, sr = librosa.load(file, sr=None)  # sr=None preserves the original sampling rate
    duration = librosa.get_duration(y=audio, sr=sr)
    
    # Example: Extracting MFCC features
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)  # Take the mean of the MFCCs over time
    
    return duration, mfccs_mean

def main():
    st.title('Audio File Upload and Processing')
    st.subheader("'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock' only these music can be predicted")

# Upload audio file
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"])

# Check if the file is uploaded
    if uploaded_file is not None:
    # Process the uploaded audio file
        duration, mfccs_mean = process_audio_file(uploaded_file)

    # Display the results
        st.write(f"Audio duration: {duration:.2f} seconds")
        st.write("MFCCs (Mean of 13 coefficients):")
        st.subheader("Original Data")
        st.write(np.array(mfccs_mean))
        st.subheader("Scaled Data")
        scaler=pickle.load(open('music-predictor-scaler.pkl','rb'))
        scaled=scaler.transform(np.array(mfccs_mean).reshape(1, -1))
        st.write(scaled)
        st.subheader("Prediction")
        model = pickle.load(open("music-predictor.pkl", "rb"))
        predictions = model.predict(scaled)
        #predictions = np.expm1(predictions)
        st.write(predictions[0])

if __name__ == "__main__":
    main()