import streamlit as st
import numpy as np
import librosa
import matplotlib.pyplot as plt
import tensorflow as tf
import pywt
import plotly.express as px
from io import BytesIO
from pydub import AudioSegment

emotions = {1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad', 5: 'angry', 6: 'fearful', 7: 'disgust', 8: 'surprised'}

# Load the model architecture and weights
saved_model_path = 'lstm_model (2).json'
saved_weights_path = 'lstm_model_weights.weights (1).h5'

with open(saved_model_path, 'r') as json_file:
    json_savedModel = json_file.read()

# Load the model architecture and weights
model = tf.keras.models.model_from_json(json_savedModel)
model.load_weights(saved_weights_path)

def extract_mfcc(y, sr):
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfccs

def extract_wavelet_features(y):
    coeffs = pywt.wavedec(y, 'db4', level=5)
    cA = coeffs[0]
    cD = coeffs[1:]

    features = []
    features.append(np.mean(cA))
    features.append(np.std(cA))
    for detail in cD:
        features.append(np.mean(detail))
        features.append(np.std(detail))
    return features

def extract_features(wav_file_name):
    y, sr = librosa.load(wav_file_name)
    mfcc_features = extract_mfcc(y, sr)
    wavelet_features = extract_wavelet_features(y)
    combined_features = np.concatenate((mfcc_features, wavelet_features))
    return combined_features

def predict_emotion(wav_filepath):
    test_point = extract_features(wav_filepath)
    test_point = np.reshape(test_point, newshape=(1, 52, 1))
    predictions = model.predict(test_point)
    predicted_emotion = emotions[np.argmax(predictions[0]) + 1]
    return predicted_emotion

def plot_waveform(y, sr):
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr, color='b')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Audio Waveform')
    buf = BytesIO()
    plt.savefig(buf, format="png")
    st.image(buf)

def get_audio_length(file_path):
    # Load the audio file
    audio = AudioSegment.from_file(file_path)
    
    # Get the length in milliseconds
    length_in_milliseconds = len(audio)
    
    # Convert milliseconds to seconds
    length_in_seconds = length_in_milliseconds / 1000.0
    
    return length_in_seconds

def split_audio_and_predict(file_path, segment_length=7000):
    # Load the audio file
    audio = AudioSegment.from_file(file_path)
    
    # Initialize an array to store the results
    results = []
    
    # Calculate the number of segments
    total_length = len(audio)
    num_segments = (total_length + segment_length - 1) // segment_length
    
    # Split the audio into segments and predict emotion for each segment
    for i in range(num_segments):
        start_time = i * segment_length
        end_time = start_time + segment_length
        
        # Extract the segment
        segment = audio[start_time:end_time]
        
        # Save the segment as a temporary file
        segment.export("temp_audio.wav", format="wav")
        
        # Predict emotion for the segment and store the result
        emotion = predict_emotion("temp_audio.wav")
        results.append(emotion)
    
    return results

# Streamlit layout
st.title("Emotion Prediction from Audio")

uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    if get_audio_length("temp_audio.wav")<=7:

    # Load the audio file
     y, sr = librosa.load("temp_audio.wav", sr=None)

    # Display the waveform
     plot_waveform(y, sr)

    # Predict emotion
     predicted_emotion = predict_emotion("temp_audio.wav")
     st.write(f"*Predicted Emotion:* {predicted_emotion}")
    else:
     emotion_results = split_audio_and_predict("temp_audio.wav")
     time_intervals = [i*7 for i in range(len(emotion_results))]
     emotion_values = {'neutral': 0, 'calm': 1, 'happy': 2, 'sad': 3, 'angry': 4, 'fearful': 5, 'disgust': 6, 'suprised': 7}
     # Convert emotions to numerical values
     emotion_numerical = [emotion_values[emotion] for emotion in emotion_results]

      # Generate the line chart
     plt.figure(figsize=(10, 6))
     plt.plot(time_intervals, emotion_numerical, marker='o')

    # Customize the plot
     plt.yticks(list(emotion_values.values()), list(emotion_values.keys()))
     plt.xlabel('Time (s)')
     plt.ylabel('Emotion')
     plt.title('Emotion Variation Over Time')
     plt.grid(True)
     st.pyplot(plt)


    # Play audio
    st.audio(uploaded_file)
