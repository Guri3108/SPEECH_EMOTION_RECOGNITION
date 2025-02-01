import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import joblib
from tensorflow.keras.models import load_model

# Configuration
FEATURES_PATH = os.path.join("data", "features.csv")
MODEL_PATH = os.path.join("data", "model.keras")
SCALER_PATH = os.path.join("data", "scaler.pkl")
LABEL_ENCODER_PATH = os.path.join("data", "label_encoder.pkl")

# Load the data
def load_data():
    df = pd.read_csv(FEATURES_PATH)
    X = df.drop("label", axis=1).values

    # Map numeric labels to emotion strings
    emotion_map = {
        '1': 'neutral',
        '2': 'calm',
        '3': 'happy',
        '4': 'sad',
        '5': 'angry',
        '6': 'fearful',
        '7': 'disgust',
        '8': 'surprised'
    }
    y = np.array([emotion_map[label] for label in df["label"].astype(str)])  # Convert numeric labels to emotion strings

    return X, y

# Preprocess the data
def preprocess_data(X, y):
    # Load the scaler and label encoder
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)

    # Normalize features
    X_scaled = scaler.transform(X)

    # Reshape X for CNN input (samples, timesteps, features)
    X_reshaped = np.expand_dims(X_scaled, axis=2)

    # Debug: Print unique labels in the dataset
    print("Unique labels in the dataset:", np.unique(y))

    # Debug: Print label encoder classes
    print("Label Encoder Classes:", label_encoder.classes_)

    # Encode labels to integers
    try:
        y_encoded = label_encoder.transform(y)
    except ValueError as e:
        print(f"Error encoding labels: {e}")
        # Handle unexpected labels by mapping them to a default value (e.g., 'unknown')
        y = np.array(['unknown' if label not in label_encoder.classes_ else label for label in y])
        y_encoded = label_encoder.transform(y)

    return X_reshaped, y_encoded, label_encoder

# Plot emotion distribution
def plot_emotion_distribution(y, label_encoder):
    emotion_labels = label_encoder.inverse_transform(y)
    sns.countplot(x=emotion_labels)
    plt.title("Distribution of Emotions in the Dataset")
    plt.xlabel("Emotion")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.show()

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, label_encoder):
    emotion_labels = label_encoder.classes_
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=emotion_labels, yticklabels=emotion_labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Emotion")
    plt.ylabel("True Emotion")
    plt.show()

# Main function
def analyze_results():
    # Load and preprocess data
    X, y = load_data()
    X_reshaped, y_encoded, label_encoder = preprocess_data(X, y)

    # Load the model
    model = load_model(MODEL_PATH)

    # Make predictions
    y_pred = model.predict(X_reshaped)
    y_pred = np.argmax(y_pred, axis=1)

    # Plot emotion distribution
    plot_emotion_distribution(y_encoded, label_encoder)

    # Plot confusion matrix
    plot_confusion_matrix(y_encoded, y_pred, label_encoder)

    # Print classification report
    print("Classification Report:")
    print(classification_report(y_encoded, y_pred, target_names=label_encoder.classes_))

if __name__ == "__main__":
    analyze_results()