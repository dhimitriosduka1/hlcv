import os
import cv2
import numpy as np
import mediapipe as mp
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import logging
import joblib

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# MediaPipe hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

def load_data(base_path):
    X = []
    y = []
    
    for split in ['train', 'valid', 'test']:
        split_path = os.path.join(base_path, split)
        logging.info(f"Processing {split} data...")
        for chord_class in os.listdir(split_path):
            class_path = os.path.join(split_path, chord_class)
            for i, img_name in enumerate(os.listdir(class_path)):
                if i % 100 == 0:
                    logging.info(f"  Processing {split} - {chord_class}: image {i}")
                img_path = os.path.join(class_path, img_name)
                img = cv2.imread(img_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(img_rgb)
                
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                    X.append(landmarks)
                    y.append(chord_class)
        logging.info(f"Finished processing {split} data. Total samples: {len(X)}")
    
    return np.array(X), np.array(y)

def train_mlp(X_train, y_train, X_test, y_test, hidden_layer_sizes):
    logging.info("Training MLP model with different hidden_layer_sizes...")
    best_score = 0
    best_model = None
    results = []

    for hidden_layer_size in hidden_layer_sizes:
        logging.info(f"Training with hidden_layer_sizes={hidden_layer_size}...")
        mlp_clf = MLPClassifier(hidden_layer_sizes=hidden_layer_size, max_iter=4000, random_state=42)
        
        mlp_clf.fit(X_train, y_train)
        train_score = mlp_clf.score(X_train, y_train)
        logging.info(f"Train accuracy with hidden_layer_sizes={hidden_layer_size}: {train_score:.4f}")
        
        val_score = mlp_clf.score(X_test, y_test)
        results.append((hidden_layer_size, train_score, val_score))
        
        if val_score > best_score:
            best_score = val_score
            best_model = mlp_clf

    logging.info("MLP models trained successfully")
    for hidden_layer_size, train_score, val_score in results:
        logging.info(f"hidden_layer_sizes={hidden_layer_size}: Train accuracy = {train_score:.4f}, Validation accuracy = {val_score:.4f}")
    
    return best_model

def load_additional_test_data(test_path):
    logging.info(f"Loading additional test data from {test_path}")
    X_test = []
    y_test = []
    
    for chord_class in os.listdir(test_path):
        class_path = os.path.join(test_path, chord_class)
        for i, img_name in enumerate(os.listdir(class_path)):
            if i % 100 == 0:
                logging.info(f"  Processing test - {chord_class}: image {i}")
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                X_test.append(landmarks)
                y_test.append(chord_class)
                
    logging.info(f"Finished processing additional test data. Total samples: {len(X_test)}")
    
    return np.array(X_test), np.array(y_test)

def save_model(model, le, filename='mlp_model.joblib'):
    logging.info(f"Saving model to {filename}")
    joblib.dump({'model': model, 'label_encoder': le}, filename)
    logging.info("Model saved successfully")

def load_model(filename='mlp_model.joblib'):
    logging.info(f"Loading model from {filename}")
    loaded = joblib.load(filename)
    logging.info("Model loaded successfully")
    return loaded['model'], loaded['label_encoder']

def evaluate_model(model, X, y, le, split_name):
    logging.info(f"Evaluating model on {split_name} set...")
    y_pred = model.predict(X)
    
    accuracy = accuracy_score(y, y_pred)
    logging.info(f"Accuracy on {split_name} set: {accuracy:.4f}")
    
    print(f"\nMLP Classification Report - {split_name} Set:")
    print(classification_report(y, y_pred, target_names=le.classes_))
    
    return accuracy

def main():
    logging.info("Starting chord classification script")
    
    # Load and preprocess data
    base_path = '/home/dhimitrios-duka/Documents/UdS/SoSe 2024/High-Level Computer Vision/Assignments/hlcv/Project/src/hand-pose-ess/merged_ds'
    logging.info(f"Loading data from {base_path}")
    X, y = load_data(base_path)

    # Encode labels
    logging.info("Encoding labels")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    logging.info(f"Total number of classes: {len(le.classes_)}")

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Train the MLP model with different hidden_layer_sizes configurations
    hidden_layer_sizes = [(100,), (100, 256), (100, 256, 100)]
    best_mlp_model = train_mlp(X_train, y_train, X_test, y_test, hidden_layer_sizes)

    # Save the trained model
    save_model(best_mlp_model, le)

    # Evaluate the model on training and test sets
    # train_accuracy = evaluate_model(best_mlp_model, X_train, y_train, le, "Training")
    test_accuracy = evaluate_model(best_mlp_model, X_test, y_test, le, "Test")

    # logging.info(f"Final Training Accuracy: {train_accuracy:.4f}")
    logging.info(f"Final Test Accuracy: {test_accuracy:.4f}")

    # Load additional test dataset and evaluate
    additional_test_path = 'guitar-chords-tiny/test'
    X_additional_test, y_additional_test = load_additional_test_data(additional_test_path)
    y_additional_test_encoded = le.transform(y_additional_test)
    additional_test_accuracy = evaluate_model(best_mlp_model, X_additional_test, y_additional_test_encoded, le, "guitar-chords-tiny")
    logging.info(f"Additional Test Accuracy: {additional_test_accuracy:.4f}")

    print("-" * 100)
    print("-" * 100)
    print("-" * 100)

    # Load additional test dataset and evaluate
    additional_test_path = 'guitar-chords-ours-A-G/test'
    X_additional_test, y_additional_test = load_additional_test_data(additional_test_path)
    y_additional_test_encoded = le.transform(y_additional_test)
    additional_test_accuracy = evaluate_model(best_mlp_model, X_additional_test, y_additional_test_encoded, le, "guitar-chords-ours-A-G")
    logging.info(f"Additional Test Accuracy: {additional_test_accuracy:.4f}")


if __name__ == "__main__":
    main()
