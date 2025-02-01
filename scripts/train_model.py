import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, GlobalAveragePooling1D
from tensorflow.keras.utils import to_categorical
import joblib

# Configuration
FEATURES_PATH = os.path.join("data", "features.csv")
MODEL_PATH = os.path.join("data", "model.keras")
NUM_CLASSES = 9  # Update this to match the number of unique labels

# Load the data
def load_data():
    df = pd.read_csv(FEATURES_PATH)
    X = df.drop("label", axis=1).values
    y = df["label"].values

    # Print unique labels in the dataset
    print("Unique labels in the dataset:", np.unique(y))

    return X, y

# Preprocess the data
def preprocess_data(X, y):
    # Map numeric labels to emotion strings
    emotion_map = {
        0: "neutral",
        1: "calm",
        2: "happy",
        3: "sad",
        4: "angry",
        5: "fearful",
        6: "disgust",
        7: "surprised",
        8: "unknown"  # Add this line to handle the new label
    }
    y = np.array([emotion_map[label] for label in y])  # Convert numeric labels to emotion strings

    # Encode labels to integers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)  # Fit with string labels
    y_onehot = to_categorical(y_encoded, num_classes=NUM_CLASSES)

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Reshape X for CNN input (samples, timesteps, features)
    X_reshaped = np.expand_dims(X_scaled, axis=2)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_onehot, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, scaler, label_encoder

# Build the CNN model
def build_model(input_shape):
    model = Sequential([
        Conv1D(64, kernel_size=3, activation="relu", input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        Conv1D(128, kernel_size=3, activation="relu"),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        GlobalAveragePooling1D(),
        Dense(256, activation="relu"),
        Dropout(0.3),
        Dense(NUM_CLASSES, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# Train the model
def train_model():
    # Load and preprocess data
    X, y = load_data()
    X_train, X_test, y_train, y_test, scaler, label_encoder = preprocess_data(X, y)

    # Build the model
    input_shape = (X_train.shape[1], 1)
    model = build_model(input_shape)

    # Train the model
    print("Training the model...")
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

    # Save the model and preprocessing objects
    model.save(MODEL_PATH)
    joblib.dump(scaler, os.path.join("data", "scaler.pkl"))
    joblib.dump(label_encoder, os.path.join("data", "label_encoder.pkl"))
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_model()