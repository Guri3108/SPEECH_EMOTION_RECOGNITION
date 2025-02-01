def preprocess_dataset():
    features = []
    labels = []

    # Loop through all actors and audio files
    for actor_folder in tqdm(os.listdir(DATA_PATH)):
        actor_path = os.path.join(DATA_PATH, actor_folder)
        if os.path.isdir(actor_path):
            for file_name in os.listdir(actor_path):
                file_path = os.path.join(actor_path, file_name)
                # Skip files that don't match the expected naming convention
                if len(file_name.split("-")) < 3:
                    print(f"Skipping file (invalid name): {file_name}")
                    continue
                try:
                    # Extract emotion label from the file name
                    emotion_label = int(file_name.split("-")[2])
                    # Extract features
                    feature = extract_features(file_path)
                    if feature is not None:
                        features.append(feature)
                        labels.append(emotion_label)
                except Exception as e:
                    print(f"Error processing file {file_name}: {e}")
                    continue

    # Convert to DataFrame
    df = pd.DataFrame(features)
    df["label"] = labels
    # Save to CSV
    df.to_csv(FEATURES_PATH, index=False)
    print(f"Features saved to {FEATURES_PATH}")