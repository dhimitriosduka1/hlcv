import os
import cv2
import numpy as np
import mediapipe as mp
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import logging
import time
import joblib


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# MediaPipe hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)

def process_image(img_path, chord_class):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
        return landmarks, chord_class
    return None, None

def load_data(base_path):
    datasets = {'train': [], 'test': []}
    labels = {'train': [], 'test': []}

    number_of_skipped_images = 0
    
    for split in ['train', 'valid', 'test']:
        split_path = os.path.join(base_path, split)
        logging.info(f"Processing {split} data...")
        target_split = 'train' if split != 'test' else 'test'
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
                    datasets[target_split].append(landmarks)
                    labels[target_split].append(chord_class)
                else:
                    number_of_skipped_images += 1
        logging.info(f"Finished processing {split} data. Total samples in {target_split}: {len(datasets[target_split])}")
        logging.info(f"Number of skipped images: {number_of_skipped_images}")

    return (
        np.array(datasets['train']), 
        np.array(datasets['test']),
        np.array(labels['train']), 
        np.array(labels['test'])
    )

def load_additional_test_data(test_path):
    logging.info(f"Loading additional test data from {test_path}")
    X_test = []
    y_test = []
    
    number_of_skipped_images = 0

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
            else:
                number_of_skipped_images += 1
                
    logging.info(f"Finished processing additional test data. Total samples: {len(X_test)}")
    logging.info(f"Number of skipped images: {number_of_skipped_images}")

    return np.array(X_test), np.array(y_test)

def train_svm(X_train, y_train, C_values):
    logging.info("Training SVM models with different C values...")
    best_score = 0
    best_model = None
    results = []

    for C in C_values:
        svm_clf = SVC(kernel='rbf', C=C, random_state=42)
        
        start_time = time.time()
        scores = cross_val_score(svm_clf, X_train, y_train, cv=5, scoring='accuracy')
        end_time = time.time()
        
        mean_score = np.mean(scores)
        results.append((C, mean_score))
        
        logging.info(f"C={C}: Mean accuracy = {mean_score:.4f}, Time: {end_time - start_time:.2f} seconds")
        
        if mean_score > best_score:
            best_score = mean_score
            best_C = C

    # Train the final model with the best C value
    best_model = SVC(kernel='rbf', C=best_C, random_state=42)
    best_model.fit(X_train, y_train)
    
    logging.info(f"Best C value: {best_C}")
    logging.info(f"Best cross-validation score: {best_score:.4f}")
    
    return best_model, results

def save_model(model, le, filename='svm_model.joblib'):
    logging.info(f"Saving model to {filename}")
    joblib.dump({'model': model, 'label_encoder': le}, filename)
    logging.info("Model saved successfully")

def load_model(filename='svm_model.joblib'):
    logging.info(f"Loading model from {filename}")
    loaded = joblib.load(filename)
    logging.info("Model loaded successfully")
    return loaded['model'], loaded['label_encoder']

def evaluate_model(model, X, y, le, split_name):
    logging.info(f"Evaluating model on {split_name} set...")
    start_time = time.time()
    y_pred = model.predict(X)
    end_time = time.time()
    logging.info(f"Evaluation on {split_name} set completed in {end_time - start_time:.2f} seconds")
    
    accuracy = accuracy_score(y, y_pred)
    logging.info(f"Accuracy on {split_name} set: {accuracy:.4f}")
    
    print(f"\nSVM Classification Report - {split_name} Set:")
    # print(classification_report(y, y_pred, target_names=le.classes_))
    
    return accuracy

def classify_chord(image_path, model, le):
    logging.info(f"Classifying chord in image: {image_path}")
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
        
        prediction = model.predict([landmarks])
        result = le.inverse_transform(prediction)[0]
        
        logging.info(f"Predicted chord: {result}")
    else:
        logging.warning("No hand detected in the image.")

def plot_embeddings(X, y, le):
    # PCA
    logging.info("Computing PCA projection")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # t-SNE
    logging.info("Computing t-SNE projection")
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)

    # Plotting PCA
    plt.figure(figsize=(7, 7))
    for label in np.unique(y):
        plt.scatter(X_pca[y == label, 0], X_pca[y == label, 1], label=le.inverse_transform([label])[0], alpha=0.6)
    plt.title('PCA Projection')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.savefig('pca_projection.png')
    plt.close()
    
    # Plotting t-SNE
    plt.figure(figsize=(7, 7))
    for label in np.unique(y):
        plt.scatter(X_tsne[y == label, 0], X_tsne[y == label, 1], label=le.inverse_transform([label])[0], alpha=0.6)
    plt.title('t-SNE Projection')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.savefig('tsne_projection.png')
    plt.close()


def main():
    logging.info("Starting chord classification script")
    
    # Load and preprocess data
    base_path = 'merged_ds'
    logging.info(f"Loading data from {base_path}")
    X_train, X_test, y_train, y_test = load_data(base_path)

    # Encode labels
    logging.info("Encoding labels")
    le = LabelEncoder()
    le.fit(np.concatenate((y_train, y_test)))
    y_train_encoded = le.transform(y_train)
    y_test_encoded = le.transform(y_test)
    logging.info(f"Total number of classes: {len(le.classes_)}")

    # Train the SVM model with different C values
    C_values = [0.1, 1, 10, 100, 200, 300]
    best_svm_model, results = train_svm(X_train, y_train_encoded, C_values)

    # Print results for each C value
    print("\nResults for each C value:")
    for C, score in results:
        print(f"C = {C}: Mean accuracy = {score:.4f}")

    # Save the best trained model
    save_model(best_svm_model, le)

    # Evaluate the model on training and test sets
    train_accuracy = evaluate_model(best_svm_model, X_train, y_train_encoded, le, "Training")
    test_accuracy = evaluate_model(best_svm_model, X_test, y_test_encoded, le, "Test")

    logging.info(f"Final Training Accuracy: {train_accuracy:.4f}")
    logging.info(f"Final Test Accuracy: {test_accuracy:.4f}")

    print("-" * 100)
    print("-" * 100)
    print("-" * 100)

    # Load additional test dataset and evaluate
    additional_test_path = 'guitar-chords-tiny/test'
    X_additional_test, y_additional_test = load_additional_test_data(additional_test_path)
    y_additional_test_encoded = le.transform(y_additional_test)
    additional_test_accuracy = evaluate_model(best_svm_model, X_additional_test, y_additional_test_encoded, le, "guitar-chords-tiny")
    logging.info(f"Additional Test Accuracy: {additional_test_accuracy:.4f}")

    print("-" * 100)
    print("-" * 100)
    print("-" * 100)

    # Load additional test dataset and evaluate
    additional_test_path = 'guitar-chords-ours-A-G/test'
    X_additional_test, y_additional_test = load_additional_test_data(additional_test_path)
    y_additional_test_encoded = le.transform(y_additional_test)
    additional_test_accuracy = evaluate_model(best_svm_model, X_additional_test, y_additional_test_encoded, le, "guitar-chords-ours-A-G")
    logging.info(f"Additional Test Accuracy: {additional_test_accuracy:.4f}")

    # Plot the embeddings
    plot_embeddings(X_train, y_train_encoded, le)

    # # Example of loading the saved model and classifying a new image
    # loaded_model, loaded_le = load_model()
    # classify_chord('Screenshot from 2024-07-30 20-47-11.png', loaded_model, loaded_le)

if __name__ == "__main__":
    main()