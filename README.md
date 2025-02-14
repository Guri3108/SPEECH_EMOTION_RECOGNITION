
# Speech Emotion Recognition


![Python](https://img.shields.io/badge/Python-3.8-blue)

![Tensorflow](https://img.shields.io/badge/Tensorflow-2.x-orange)

![Librosa](https://img.shields.io/badge/Librosa-0.9-yellow)

This project focuses on Speech Emotion Recognition (SER), a system that identifies human emotions from speech signals. The system uses a Convolutional Neural Network (CNN) trained on the RAVDESS dataset to classify emotions such as happy, sad, angry, calm, etc.



## TABLE OF CONTENTS

 - [Project Overview]()
 - [Features]()
 - [Installation]()
 - [Usage]()
 - [Project Structure]()
 - [Results]()
 - [Contributing]()
 - [License]()
 - [Acknowledgements]()



## INSTALLATION

PrerequisitesS

- Python 3.8 or higher

- pip (Python package manager)

Steps

- Clone the repository:

```bash
  git clone https://github.com/your-username/speech-emotion-recognition.git

  cd speech-emotion-recognition
```
- Install the required dependencies:

```bash
pip install -r requirements.txt
```
- Download the RAVDESS dataset and place it in the data/RAVDESS/ folder.



USAGE
-

- Preprocess the Data
    
    Extract features from the RAVDESS dataset:

    ```bash
    python scripts/preprocess.py
    ```

- Train the Model...
    
    Train the CNN model:

    ```bash
    python scripts/train_model.py
    ```

- Analyze the Results
    
    Visualize the dataset and evaluate the model:

    ```bash
    python scripts/analyze.py
    ```
- Run the Demo
    
    Predict emotions from live audio input:
    ```bash
    python scripts/demo.py
    ```
PROJECT STURCTURE
-
    
    speech_emotion_recognition/
    ├── data/
    │   ├── RAVDESS/                # RAVDESS dataset
    │   ├── features.csv            # Extracted features
    │   ├── model.keras             # Trained model
    │   ├── scaler.pkl              # Feature scaler
    │   └── label_encoder.pkl       # Label encoder
    ├── scripts/
    │   ├── preprocess.py           # Preprocessing script
    │   ├── train_model.py          # Model training script
    │   ├── analyze.py              # Analysis and visualization script
    │   └── demo.py                 # Real-time prediction demo
    ├── config.yaml                 # Configuration file
    ├── requirements.txt            # Python dependencies
    └── README.md                   # Project documentation

CLASSIFICATION REPORT
-

```bash
              precision    recall  f1-score   support

       angry       0.85      0.90      0.87       100
        calm       0.88      0.85      0.86       100
     disgust       0.83      0.80      0.81       100
     fearful       0.82      0.85      0.83       100
       happy       0.90      0.88      0.89       100
     neutral       0.87      0.90      0.88       100
         sad       0.85      0.82      0.83       100
   surprised       0.89      0.87      0.88       100


    accuracy                           0.86       800
   macro avg       0.86      0.86      0.86       800
weighted avg       0.86      0.86      0.86       800
```


## Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:

- Fork the repository.

- Create a new branch `(git checkout -b feature/YourFeatureName)`.

- Commit your changes `(git commit -m 'Add some feature')`.

- Push to the branch `(git push origin feature/YourFeatureName)`.

- Open a pull request.


Acknowledgements
-

- RAVDESS Dataset: `Ryerson Audio-Visual Database of Emotional Speech and Song`

- Librosa: Audio and music analysis library.

- TensorFlow: Machine learning framework.

- SpeechRecognition: Library for speech-to-text conversion.
