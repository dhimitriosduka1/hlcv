import os
from email.mime import image
from gc import collect as garbage_collect
from typing import Any
from warnings import warn

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.cuda import empty_cache as cuda_empty_cache
from torch.cuda import mem_get_info
from torchvision import transforms
from torchvision.transforms import Resize
from transformers import AutoConfig, AutoModelForImageClassification, Dinov2Config, Dinov2ForImageClassification
from ultralytics import YOLO


class ClassificationModelWrapper(torch.nn.Module):
    """A wrapper class to use an image classification model, where the forward method returns the logits/probabilities."""

    def __init__(self, model, return_logits=True) -> None:
        """
        Parameters
        ----------
        model : torch.nn.Module
            Image classification model.
        return_logits : bool, optional
            Whether to return the logits instead of the probabilities (default: True).
        """
        super().__init__()
        self.model = model
        self.return_logits = return_logits

    def forward(self, x):
        outputs = self.model(x)
        return outputs.logits if self.return_logits else outputs.probs


def safe_image_load(
    image_input: Any, as_tensor=False, resize_shape: tuple[int, int] | None = None, add_batch_dim=False
) -> np.ndarray | torch.Tensor:
    """
    Load an image from a file path to a `np.ndarray` or `torch.Tensor` if `as_tensor` is True.

    Parameters
    ----------
    image_input : str | np.ndarray | torch.Tensor
        Path to the image file or numpy array / tensor of the image.
    as_tensor : bool, optional
        Whether to return the image as a torch.Tensor (default: False).
    resize_shape : tuple[int, int] | None, optional
        Shape to resize the image to (default: None). Only used if `as_tensor` is True with the `Resize` transform
        from `torchvision.tranforms`.
    add_batch_dim : bool, optional
        Whether to add a batch dimension to the image (default: False).

    Returns
    -------
    numpy.ndarray | torch.Tensor
        Image as a numpy array or tensor in BGR format.

    Raises
    ------
    ValueError
        If `image_input` is not a file path (str) or a numpy array (np.ndarray) or a torch.Tensor.
    """
    # Handle input: convert to numpy array if it's a file path
    if isinstance(image_input, str):
        image_np = load_image_to_numpy(image_input)
    elif isinstance(image_input, np.ndarray):
        image_np = image_input if not as_tensor else torch.from_numpy(image_input)
    elif isinstance(image_input, torch.Tensor):
        image_np = image_input.detach().numpy() if not as_tensor else image_input
    else:
        raise ValueError("image_input must be either a file path (str), numpy array, or torch.Tensor")

    if as_tensor and resize_shape:
        # If the image has 3 dimensions, assume it's in HWC format and convert to CHW
        image_np = HWC_to_CHW(image_np, add_batch_dim=add_batch_dim, normalize=False)
        resize_transform = Resize(resize_shape)
        image_np = resize_transform(image_np)

    return image_np


def HWC_to_CHW(image: Any, add_batch_dim=False, normalize=True) -> torch.Tensor:
    """Convert an image from HWC to CHW format or BCHW format if `add_batch_dim` is True.

    Parameters
    ----------
    image : Any
        Same as in `safe_image_load()`.
    add_batch_dim : bool, optional
        Whether to add a batch dimension to the image (default: False).
    normalize : bool, optional
        Whether to normalize the image to [0, 1] (default: True).

    Returns
    -------
    torch.Tensor
        Image in CHW format or BCHW format if `add_batch_dim` is True.
    """
    image_tensor = safe_image_load(image, as_tensor=True)
    image_tensor = image_tensor.permute(2, 0, 1)

    # Ensure the tensor is in the correct format (BCHW)
    if add_batch_dim and image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.float()  # Convert to float

    if normalize and image_tensor.max() > 1.0:
        image_tensor /= 255.0  # Normalize to [0, 1]

    return image_tensor


def extract_box_object(
    image_input: str | np.ndarray | torch.Tensor,
    model: YOLO,
    class_name="fretboard",
    conf=0.25,
    expand_percent=0,
    as_bounding_box=False,
    as_PIL=False,
) -> Image.Image | np.ndarray | torch.Tensor | None:
    """
    Extract a specified object from an image using a YOLO model.

    Parameters
    ----------
    image_input : str | np.ndarray | torch.Tensor
        Path to image file or array / tensor of the image.
    model : YOLO
        YOLO model for object detection.
    class_name : str, optional
        Name of the class to extract (default: 'fretboard').
    conf : float, optional
        Confidence threshold for detection (default: 0.25).
    as_bounding_box : bool, optional
        Whether to return the bounding box coordinates instead of the extracted object (default: False). If True,
        arguments `as_PIL` and `expand_percent` will be ignored.
    expand_percent : int, optional
        Percentage to expand the bounding box (default: 0).
    as_PIL : bool, optional
        Whether to return the extracted object as a PIL Image. If False, returns a numpy array
        (default: False).

    Returns
    -------
    PIL.Image.Image | np.ndarray | torch.Tensor | None
        Extracted object as a PIL Image or numpy array if `as_bounding_box` is False, otherwise the bounding box
        coordinates as a torch.Tensor. Returns None if no object is detected.
    """
    # Get the required input shape from the model
    if hasattr(model, "model") and hasattr(model.model, "args"):
        required_shape = model.model.args.get("imgsz", (640, 640))
    else:
        # Default to (640, 640) if we can't determine the shape from the model
        warn("Could not determine the required input shape from the given model. Defaulting to (640, 640)")
        required_shape = (640, 640)

    # Ensure required_shape is a tuple of two integers
    if isinstance(required_shape, int):
        required_shape = (required_shape, required_shape)

    # Load the image as tensor for the model
    image_tensor = safe_image_load(image_input, as_tensor=True, resize_shape=required_shape, add_batch_dim=True)

    # Load the original image as numpy array for the final extraction
    image_np = safe_image_load(image_input, as_tensor=False)

    # Predict using the model
    results = model(image_tensor)[0]

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

    # Return the bounding box coordinates
    if as_bounding_box:
        return target_box.xyxy.squeeze()

    # Extract, scale, and expand the bounding box
    x1, y1, x2, y2 = map(int, target_box.xyxy[0])
    x1, y1, x2, y2 = scale_coordinates([x1, y1, x2, y2], (img_height, img_width), required_shape)
    x1, y1, x2, y2 = expand_coordinates([x1, y1, x2, y2], (img_height, img_width), expand_percent)

    # Crop the original image
    cropped_img = image_np[y1:y2, x1:x2]

    # Convert BGR to RGB
    cropped_img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)

    return Image.fromarray(cropped_img_rgb) if as_PIL else cropped_img_rgb


def scale_coordinates(box: list[int], original_shape: tuple[int, int], resized_shape: tuple[int, int]) -> list[int]:
    """Scale coordinates from the resized image back to the original image.

    Parameters
    ----------
    box : list[int]
        [x1, y1, x2, y2] coordinates on the resized image.
    original_shape : tuple[int, int]
        (height, width) of the original image
    resized_shape : tuple[int, int]
        (height, width) of the resized image

    Returns
    -------
    list[int]
        Scaled [x1, y1, x2, y2] coordinates for the original image.
    """
    orig_h, orig_w = original_shape
    resized_h, resized_w = resized_shape

    # Calculate scaling factors
    scale_x = orig_w / resized_w
    scale_y = orig_h / resized_h

    # Scale coordinates
    x1, y1, x2, y2 = box
    x1 = int(x1 * scale_x)
    y1 = int(y1 * scale_y)
    x2 = int(x2 * scale_x)
    y2 = int(y2 * scale_y)

    return [x1, y1, x2, y2]


def expand_coordinates(box: list[int], original_shape: tuple[int, int], expand_percent: float) -> list[int]:
    """Expand bounding box coordinates by a percentage.

    Parameters
    ----------
    box : list[int]
        [x1, y1, x2, y2] coordinates of the bounding box.
    original_shape : tuple[int, int]
        (height, width) of the original image.
    expand_percent : float
        Percentage to expand the bounding box.

    Returns
    -------
    list[int]
        Expanded [x1, y1, x2, y2] coordinates.
    """
    x1, y1, x2, y2 = box
    if expand_percent > 0:
        width = x2 - x1
        height = y2 - y1
        expand_x = int(width * expand_percent / 100)
        expand_y = int(height * expand_percent / 100)
        img_height, img_width = original_shape
        x1 = max(0, x1 - expand_x)
        y1 = max(0, y1 - expand_y)
        x2 = min(img_width, x2 + expand_x)
        y2 = min(img_height, y2 + expand_y)
    else:
        raise ValueError("expand_percent must be a non-negative integer")

    return [x1, y1, x2, y2]


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


def invert_dict(d: dict) -> dict:
    """Invert the keys and values of a dictionary, so that the values become the keys and vice versa."""
    return {v: k for k, v in d.items()}


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
        model = AutoModelForImageClassification.from_pretrained(model_path, config=config, local_files_only=True)
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


def clean_cache():
    """Cleans the GPU memory cache."""
    garbage_collect()
    cuda_empty_cache()
    mem_info = mem_get_info()
    print(f"Freeing GPU Memory\nFree: %d MB\tTotal: %d MB" % (mem_info[0] // 1024**2, mem_info[1] // 1024**2))


def initialize_untrained_dinov2(num_labels, image_size=518):
    # Define the configuration
    config = Dinov2Config(
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=4096,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-6,
        image_size=image_size,
        patch_size=14,
        num_channels=3,
        qkv_bias=True,
        num_labels=num_labels,
    )

    # Initialize the model
    model = Dinov2ForImageClassification(config)

    return model
