import cv2
import torch
from collections import deque, Counter
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
from ultralytics import YOLO

def load_yolo(model_path):
    model = YOLO(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model


def process_video(video_path, model_name="facebook/dinov2-large", feature_extractor="facebook/dinov2-large"):
    # Load the pre-trained model and feature extractor
    feature_extractor = AutoImageProcessor.from_pretrained(feature_extractor)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    model.eval()

    yolo = load_yolo("yolov9c_trained_with_head.pt")
    yolo.eval()
    print(yolo)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    print(f"Video FPS: {fps}")

    recent_classifications = deque(maxlen=fps)
    
    current_frame = 0
    while False:
        ret, frame = cap.read()
        if not ret:
            break
        
        current_frame += 1

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_frame)
        
        # Preprocess the image
        inputs = feature_extractor(images=pil_image, return_tensors="pt")

        # Perform inference
        with torch.no_grad():
            outputs = model(**inputs)

        # Get the predicted class
        predicted_class_idx = outputs.logits.argmax(-1).item()
        predicted_class = model.config.id2label[predicted_class_idx]

        # Add the prediction to recent classifications
        recent_classifications.append(predicted_class)

        # If we have collected enough frames, determine the most common classification
        if len(recent_classifications) == fps:
            print(recent_classifications)
            most_common_class = Counter(recent_classifications).most_common(1)[0][0]
            print(f"Frame {current_frame}: Most common classification in last {fps} frames: {most_common_class}")
            recent_classifications.clear()
        
        # Optional: Print progress
        if current_frame % 100 == 0:
            print(f"Processed {current_frame}/{current_frame} frames")
    
    cap.release()

if __name__ == "__main__":
    video_path = "/home/dhimitriosduka/Documents/UdS/SoSe 2024/High-Level Computer Vision/Assignments/Datasets/Videos/data/G.mp4"

    process_video(
        video_path, 
        model_name="/home/dhimitriosduka/Documents/dinov2-small",
        feature_extractor="facebook/dinov2-small"
    )