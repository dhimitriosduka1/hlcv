import os
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoConfig, AutoModelForImageClassification
from ultralytics import YOLO


def safe_image_load(image_input: Any) -> np.ndarray:
    """
    Load an image from a file path to a numpy array.

    Parameters
    ----------
    image_input : str | np.ndarray
        Path to the image file or numpy array of the image.

    Returns
    -------
    numpy.ndarray
        Image as a numpy array in BGR format.

    Raises
    ------
    ValueError
        If `image_input` is not a file path (str) or a numpy array (np.ndarray).
    """
    # Handle input: convert to numpy array if it's a file path
    if isinstance(image_input, str):
        image_np = load_image_to_numpy(image_input)
    elif isinstance(image_input, np.ndarray):
        image_np = image_input
    else:
        raise ValueError("image_input must be either a file path (str) or a numpy array")
    return image_np


def extract_box_object(
    model: YOLO,
    image_input: str | np.ndarray,
    class_name="fretboard",
    conf=0.25,
    expand_percent=0,
    as_PIL=False,
) -> Image.Image | np.ndarray | None:
    """
    Extract a specified object from an image using a YOLO model.

    Parameters
    ----------
    model : YOLO
        YOLO model for object detection.
    image_input : str | np.ndarray
        Path to image file or numpy array of the image.
    class_name : str, optional
        Name of the class to extract (default: 'fretboard').
    conf : float, optional
        Confidence threshold for detection (default: 0.25).
    expand_percent : int, optional
        Percentage to expand the bounding box (default: 0).
    as_PIL : bool, optional
        Whether to return the extracted object as a PIL Image. If False, returns a numpy array
        (default: False).

    Returns
    -------
    PIL.Image.Image | np.ndarray | None
        Extracted object as a PIL Image or numpy array. Returns None if no object is detected.
    """
    image_np = safe_image_load(image_input)

    # Predict using the model
    results = model(image_np)[0]

    # Get the original image dimensions
    img_height, img_width = image_np.shape[:2]

    # Find the box with the highest confidence for the specified class
    target_box = None
    max_conf = 0

    for box in results.boxes:
        if results.names[int(box.cls)] == class_name and box.conf > conf:
            if box.conf > max_conf:
                target_box = box
                max_conf = box.conf

    if target_box is None:
        print(f"No {class_name} detected with confidence > {conf}")
        return None

    # Extract coordinates
    x1, y1, x2, y2 = map(int, target_box.xyxy[0])

    # Expand the box
    if expand_percent > 0:
        width = x2 - x1
        height = y2 - y1
        expand_x = int(width * expand_percent / 100)
        expand_y = int(height * expand_percent / 100)

        x1 = max(0, x1 - expand_x)
        y1 = max(0, y1 - expand_y)
        x2 = min(img_width, x2 + expand_x)
        y2 = min(img_height, y2 + expand_y)

    # Crop the image
    cropped_img = image_np[y1:y2, x1:x2]

    # Convert BGR to RGB
    cropped_img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)

    return Image.fromarray(cropped_img_rgb) if as_PIL else cropped_img_rgb


def predict_chord(model: AutoModelForImageClassification, image_input: str | np.ndarray) -> str:
    """
    Predict the chord from an image using an image classification model.

    Parameters
    ----------
    model : AutoModelForImageClassification
        Image classification model.
    image_input : str | np.ndarray
        Path to image file or numpy array of the image.
    
    Returns
    -------
    str
        Predicted chord.
    """
    # Define image transformations
    transform = transforms.Compose(
        [
            # transforms.Resize((518, 518)),  # Resize to match the model's expected input size
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load and preprocess the image
    image = safe_image_load(image_input)
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Move image to the same device as the model
    device = next(model.parameters()).device
    image = image.to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(image)

    # Get the predicted class
    predicted_class = outputs.logits.argmax(-1).item()

    # Map the predicted class to a label
    id2label = model.config.id2label
    predicted_label = id2label[predicted_class]

    return predicted_label


def load_image_to_numpy(image_path) -> np.ndarray:
    """
    Load an image from a file path to a numpy array.

    Parameters
    ----------
    image_path : str
        Path to the image file.

    Returns
    -------
    numpy.ndarray
        Image as a numpy array in BGR format.
    """
    return cv2.imread(image_path)


def find_files(root_path, extensions, find_all=False) -> list | str | None:
    """
    Find files with specific extensions in a given directory.

    Parameters
    ----------
    root_path : str
        The root directory to search in.
    extensions : list
        List of file extensions to search for (e.g., ['.safetensors', '.pt']).
    find_all : bool, optional
        If True, find all matching files. If False, return the first match.

    Returns
    -------
    list or str or None
        List of file paths if find_all is True, otherwise the first file path found. Returns None
        if no matching files are found.
    """
    root_path = os.path.abspath(root_path)  # Ensure root_path is absolute
    found_files = []

    for dirpath, _, filenames in os.walk(root_path):
        for filename in filenames:
            if any(filename.endswith(ext) for ext in extensions):
                full_path = os.path.join(dirpath, filename)
                relative_path = os.path.relpath(full_path, root_path)
                found_files.append(os.path.join(root_path, relative_path))
                if not find_all:
                    return found_files[0]  # Return the first file found

    if find_all:
        return found_files if found_files else None
    else:
        return None  # No files found


def ensure_files_exist(*files, names=None) -> None:
    """
    Ensure that the specified `files` exist.

    Parameters
    ----------
    files : str
        List of file paths to check.
    names : list, optional
        List of names corresponding to each file path in `files`. If provided, the names will be
        used in the output messages. If not provided, the file paths will be used as names.
    """
    for name, path in zip(names if names is not None else files, files):
        if (path and not os.path.exists(path)) or not path:
            print(f"{name} not found" + f"{f'at {path}' if path else ''}")
        else:
            print(f"{name} found at {path}")


def load_model(model_path, config_path=None, custom_class=None):
    """
    Load a model from a given path.

    Parameters
    ----------
    model_path : str
        Path to the model.
    config_path : str, optional
        Path to the model configuration file. If provided, the model will be loaded with the
        configuration.
    custom_class : class, optional
        Custom class to load the model with. If provided, the model will be loaded with the custom
        class. If `config_path` is provided, this argument will be ignored. If `config_path` is not
        provided, this argument must be provided, so that the model can be loaded with the custom
        class.

    Returns
    -------
    torch.nn.Module
        The loaded model.

    Raises
    ------
    ValueError
        If neither `config_path` nor `custom_class` is provided.
    """
    if config_path:
        # Load with config
        config = AutoConfig.from_pretrained(config_path)
        model = AutoModelForImageClassification.from_pretrained(
            model_path, config=config, local_files_only=True
        )
    elif custom_class:
        # Load with custom class (e.g., YOLO)
        model = custom_class(model_path)
    else:
        raise ValueError("Either config_path or custom_class must be provided")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model


def show_model_output(images, name="output", dpi=80, savefig=False, save_dir=".") -> None:
    """
    Display the output images.

    Parameters
    ----------
    images : list
        List of images to display.
    name, dpi, savefig, save_dir : str, int, bool, str, optional
        Name of the output image, DPI of the output image, whether to save the output image, and
        the directory to save the output image, respectively.
    """
    image = np.vstack(images)
    fig = plt.figure(dpi=dpi)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    fig.set_size_inches((5, 15))
    ax.imshow(image[..., ::-1])
    fig.tight_layout()

    # If name already exists, add an index to the name
    name = f"{name}.jpg"
    i = 1
    while os.path.exists(os.path.join(save_dir, name)):
        name = name[:-4]
        name = f"{name}_{i}.jpg"
        i += 1
    if savefig:
        name_path = os.path.join(save_dir, name)
        plt.savefig(
            os.path.join(save_dir, name_path),
            bbox_inches="tight",
            pad_inches=0.05,
            dpi=dpi,
        )
    plt.show()
    plt.close()
