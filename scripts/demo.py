import os
import numpy as np
import librosa
import sounddevice as sd
import speech_recognition as sr
import joblib
from tensorflow.keras.models import load_model

# Configuration
MODEL_PATH = os.path.join("data", "model.keras")
SCALER_PATH = os.path.join("data", "scaler.pkl")
LABEL_ENCODER_PATH = os.path.join("data", "label_encoder.pkl")

# Load the model and preprocessing objects
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)

# Function to extract features from audio
def extract_features(audio, sample_rate):
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean

# Function to preprocess audio
def preprocess_audio(audio, sample_rate):
    # Extract features
    features = extract_features(audio, sample_rate)
    # Normalize features
    features_scaled = scaler.transform([features])
    # Reshape for model input
    features_reshaped = np.expand_dims(features_scaled, axis=2)
    return features_reshaped

# Function to predict emotion
def predict_emotion(audio, sample_rate):
    # Preprocess audio
    features = preprocess_audio(audio, sample_rate)
    # Predict emotion
    prediction = model.predict(features)
    predicted_label = np.argmax(prediction, axis=1)
    predicted_emotion = label_encoder.inverse_transform(predicted_label)[0]
    return predicted_emotion

# Function to record audio
def record_audio(duration=5, sample_rate=22050):
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    print("Recording finished.")
    return audio.flatten(), sample_rate

# Function to transcribe audio to text
def transcribe_audio(audio, sample_rate):
    recognizer = sr.Recognizer()
    audio_data = sr.AudioData((audio * 32767).astype(np.int16), sample_rate, 2)  # Convert to AudioData format
    try:
        text = recognizer.recognize_google(audio_data)  # Use Google Web Speech API
        return text
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError:
        return "API unavailable"

# Main function
def run_demo():
    print("Starting real-time emotion prediction...")
    while True:
        # Record audio
        audio, sample_rate = record_audio()

        # Transcribe audio to text
        text = transcribe_audio(audio, sample_rate)
        print(f"You said: {text}")

        # Predict emotion
        emotion = predict_emotion(audio, sample_rate)
        print(f"Predicted Emotion: {emotion}")

        # Ask if the user wants to continue
        user_input = input("Press 'q' to quit or any other key to continue: ")
        if user_input.lower() == 'q':
            break

if __name__ == "__main__":
    run_demo()