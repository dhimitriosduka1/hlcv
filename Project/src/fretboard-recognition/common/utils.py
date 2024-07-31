import colorsys
import os
import shutil
import time
from datetime import datetime
from gc import collect as garbage_collect
from typing import Any
from warnings import warn

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import yaml
from datasets import load_dataset
from dotenv import load_dotenv
from matplotlib import font_manager as fm
from matplotlib import pyplot as plt
from roboflow import Roboflow
from thop import profile
from torch.cuda import empty_cache as cuda_empty_cache
from torch.cuda import mem_get_info
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.ops import nms
from transformers import Trainer

load_dotenv()


def load_config(config_path: str) -> dict:
    """Loads a YAML config file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def clean_cache():
    """Cleans the GPU memory cache."""
    garbage_collect()
    cuda_empty_cache()
    mem_info = mem_get_info()
    print(
        f"Freeing GPU Memory\nFree: %d MB\tTotal: %d MB"
        % (mem_info[0] // 1024**2, mem_info[1] // 1024**2)
    )


def ensure_directory_exists(directory: str, make=True, raise_error=False) -> None:
    """Ensure that a directory exists. If it doesn't, this function will create it if `make` is True.
    If `raise_error` is True, it will raise an error if the directory doesn't exist."""
    if make:
        os.makedirs(directory, exist_ok=not raise_error)
    else:
        if not os.path.exists(directory):
            if raise_error:
                raise FileNotFoundError(f"The directory {directory} does not exist.")
            else:
                print(f"The directory {directory} does not exist.")
        else:
            print(f"The directory {directory} exists.")


def rename_folder(old_name, new_name):
    try:
        os.rename(old_name, new_name)
        print(f"Folder renamed successfully from '{old_name}' to '{new_name}'")
    except FileNotFoundError:
        print(f"Error: The folder '{old_name}' does not exist.")
    except PermissionError:
        print(f"Error: Permission denied. Unable to rename the folder.")
    except OSError as e:
        print(f"Error: An OS error occurred: {e}")


def available_device(verbose=False) -> torch.device:
    """Returns the available device for training. If a GPU is available, it will return that,
    otherwise, it will return the CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            print("Using GPU for training")
    else:
        device = torch.device("cpu")
        if verbose:
            print("Using CPU for training")

    return device


def download_from(config: dict, location: str) -> None:
    """Downloads a dataset using the loaded `config`. It must have the following structure:

    ```
    data:
        dataset: e.g., "guitar-necks-detector" or "dduka/guitar-chords" # The name of the dataset
        load:
            interface: "roboflow" or "datasets"
            # (These must be available only if interface is "roboflow":)
            workspace: "..."
            project-version: "1"
            version-download: "..."
    ```
    """
    if config["data"]["load"]["interface"] == "roboflow":
        # Test if a ROBOFLOW_API_KEY is available
        if not os.getenv("ROBOFLOW_API_KEY"):
            from importlib.util import find_spec

            if find_spec("google"):
                from google.colab import userdata

                if userdata.get("ROBOFLOW_API_KEY"):
                    os.environ["ROBOFLOW_API_KEY"] = userdata.get("ROBOFLOW_API_KEY")
            else:
                raise ValueError(
                    "ROBOFLOW_API_KEY is not available in the environment variables. "
                    + "Create a .env file in this directory with the key or in Google "
                    + "Colab, add it to secret keys."
                )

        # Initialize Roboflow
        rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))

        # Access the workspace and project
        project = rf.workspace(config["data"]["load"]["workspace"]).project(
            config["data"]["dataset"]
        )
        version = project.version(config["data"]["load"]["project-version"])
        dataset_path = os.path.join(location, config["data"]["dataset"])
        ds = version.download(config["data"]["load"]["version-download"], location=dataset_path)
    elif config["data"]["load"]["interface"] == "datasets":
        dataset_path = os.path.join(location, config["data"]["dataset"])
        ds = load_dataset(config["data"]["dataset"], cache_dir=dataset_path)

    return ds, dataset_path


def safe_save(model: torch.nn.Module | Trainer, final_model_path: str) -> None:
    """Saves a model to a file, ensuring that the file does not already exist. If it does, it
    renames the existing file."""
    # Ensure the directory exists
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)

    if os.path.exists(final_model_path):
        # Get current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H_%M_%S")

        # Split the path into directory, filename, and extension
        directory, filename = os.path.split(final_model_path)
        name, ext = os.path.splitext(filename)

        # Create new filename with timestamp
        new_filename = f"{name}_{timestamp}{ext}"
        new_path = os.path.join(directory, new_filename)

        warn(f"{final_model_path} already exists. Renaming existing file to: {new_filename}")

        # Rename the existing file
        shutil.move(final_model_path, new_path)

    # Save the new model
    if isinstance(model, Trainer):
        model.save_model(final_model_path)
    else:
        model.save(final_model_path)
    print(f"New model saved as: {final_model_path}")


def show(imgs: list, titles=[]) -> None:
    """Show images using matplotlib."""
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        if titles:
            axs[0, i].set_title(titles[i])
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    fig.tight_layout()
    plt.show()
    plt.close()


def predict(
    image: torch.Tensor | np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    detection_threshold=0.7,
    iou_threshold=0.5,
    model_transforms=None,
    names=None,
):
    """Predict the output of an image after forward pass throughthe model and return the bounding
    boxes, class names, and class labels.

    Parameters
    ----------
    image : torch.Tensor | np.ndarray
        The image to predict on.
    model : torch.nn.Module
        The model to use for prediction. It has to be in `model.eval()` mode.
    device : torch.device
        The device to use for prediction.
    detection_threshold : float, optional
        The threshold for detection, by default 0.7.
    iou_threshold : float, optional
        The threshold for non-maximum suppression, by default 0.5.
    model_transforms : torchvision.transforms, optional
        The transforms to apply to the image before prediction, by default None.
    names : list, optional
        The list of class names, by default None.

    Returns
    -------
    tuple
        A tuple containing the bounding boxes, class names, and class labels.
    """
    # Transform the image to tensor
    if not isinstance(image, torch.Tensor):
        image = transforms.ToTensor(image)
    if model_transforms is not None:
        image = model_transforms(image)
    image = image.to(device)
    model = model.to(device)

    # Get the predictions on the image
    with torch.no_grad():
        outputs = model(image.unsqueeze(0))

    pred_scores = outputs[0]["scores"]
    pred_bboxes = outputs[0]["boxes"]
    labels = outputs[0]["labels"]

    # Get all the predicited class names
    if names is not None:
        pred_classes = [names[i] for i in labels.detach().cpu().numpy()]
    else:
        pred_classes = None

    # Apply NMS
    keep_nms = nms(pred_bboxes, pred_scores, iou_threshold)
    boxes = pred_bboxes[keep_nms].detach().cpu()
    scores = pred_scores[keep_nms].detach().cpu()
    labels = labels[keep_nms].detach().cpu()
    classes = np.asarray(pred_classes)[keep_nms.detach().cpu().numpy()] if pred_classes else None

    # Get boxes above the threshold score
    keep_score = scores >= detection_threshold
    boxes = boxes[keep_score]
    scores = scores[keep_score]
    labels = labels[keep_score]
    classes = classes[keep_score.cpu().numpy()] if classes is not None else None

    return boxes, classes, labels


def safe_item(data: np.ndarray | torch.Tensor | float) -> Any:
    """Safely gets the value of the data, whether it is a numpy array, torch tensor, or float."""
    if isinstance(data, torch.Tensor) | isinstance(data, np.ndarray):
        return data.item()
    elif isinstance(data, float):
        return data
    else:
        raise ValueError(f"Data type not supported: {type(data)} for {data}")


def as_255(img: torch.Tensor | np.ndarray, astorch=True) -> torch.Tensor:
    """Converts an image to a 255 scale."""
    if astorch:
        astype = torch.uint8
    else:
        astype = np.uint8

    if astorch:
        return torch.asarray((img * 255).type(astype))
    else:
        return (img * 255).astype(astype)


def preds_or_target_to_tensor(data: list[dict]) -> list[dict]:
    """Converts the boxes, scores, and labels in the data to torch tensors."""
    for i in range(len(data)):
        if "boxes" in data[i]:
            data[i]["boxes"] = torch.as_tensor(data[i]["boxes"], dtype=torch.float32)
        if "scores" in data[i]:
            data[i]["scores"] = torch.as_tensor(data[i]["scores"], dtype=torch.float32)
        if "labels" in data[i]:
            data[i]["labels"] = torch.as_tensor(data[i]["labels"], dtype=torch.int64)

    return data


def get_coco_names() -> list[str]:
    """Returns the COCO names."""
    return FasterRCNN_ResNet50_FPN_Weights.COCO_V1.meta["categories"].copy()


def get_custom_coco(
    new_class_name: str, coco_names: list[str] | None = None
) -> tuple[list[str], int]:
    """Creates a new list of COCO names, by looking at the next "N/A" and replacing it with
    `new_class_name`.

    Parameters
    ----------
    new_class_name : str
        The new class name to add.
    coco_names : list[str] | None
        If specified, it will use this list of COCO names, by default None.

    Returns
    -------
    tuple[list[str], int]
        A tuple containing the new list of COCO names and the index of the new class name.

    Raises
    ------
    ValueError
        If no "N/A" class is found in the COCO names.
    """
    if coco_names is None:
        custom_coco_names = get_coco_names()
    else:
        custom_coco_names = coco_names.copy()

    for i in range(len(custom_coco_names)):
        if custom_coco_names[i] == "N/A":
            custom_coco_names[i] = new_class_name
            return custom_coco_names, i
    else:
        raise ValueError("No 'N/A' class found in COCO names.")


def find_font(with_strs_in_name: list[str], case_insensitive=True) -> str:
    """Finds a font file in the system with the specified strings in the name."""
    for font in fm.findSystemFonts(fontpaths=None, fontext="ttf"):
        is_found = False
        for str_ in with_strs_in_name:
            if case_insensitive:
                if str_.lower() in fm.FontProperties(fname=font).get_name().lower():
                    is_found = True
            else:
                if str_ in fm.FontProperties(fname=font).get_name():
                    is_found = True
        if is_found:
            return font
    return None


def generate_distinct_colors(
    n: int, s: float = 0.8, v: float = 0.9, as_255=True
) -> list[tuple[float, float, float]]:
    """Generate `n` distinct colors that are perceptually different.

    Parameters
    ----------
    n : int
        Number of colors to generate
    s : float, optional
        Saturation value (0-1), by default 0.8
    v : float, optional
        Value value (0-1), by default 0.9
    as_255 : bool, optional
        If True, the colors will be in the range [0, 255], by default True

    Returns
    -------
    list[tuple[float, float, float]]
        List of RGB colors in the range [0, 1]
    """
    golden_ratio_conjugate = 0.618033988749895
    hue = 0
    colors = []

    for _ in range(n):
        hue += golden_ratio_conjugate
        hue %= 1

        # Convert HSV to RGB
        r, g, b = colorsys.hsv_to_rgb(hue, s, v)

        colors.append((int(r * 255), int(g * 255), int(b * 255) if as_255 else (r, g, b)))

    return colors


def create_color_map(class_names: list[str], as_255=True) -> dict:
    """Create a color map for the given class names `class_names` using perceptually distinct
    colors generated with `generate_distinct_colors()`."""
    n_colors = len(class_names)
    colors = generate_distinct_colors(n_colors, as_255=as_255)
    return dict(zip(class_names, colors))


def normalize_color_map(color_map: dict) -> dict:
    """Normalize the color map to be in the range [0, 1]."""
    for key in color_map:
        color_map[key] = tuple([c / 255 for c in color_map[key]])
    return color_map


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of parameters in the `model`."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_gflops(model: torch.nn.Module, input_size=(1, 3, 640, 640)) -> float:
    """Estimate the GFLOPs for the given `model` using the given `input_size`."""
    input = torch.randn(input_size).to(next(model.parameters()).device)
    flops, _ = profile(model, inputs=(input,))
    return flops / 1e9  # Convert to GFLOPs


def measure_inference_speed(model, input_size=(1, 3, 640, 640), num_iterations=100):
    """Measure the inference speed of the `model` using the given `input_size` and `num_iterations`."""
    input = torch.randn(input_size).to(next(model.parameters()).device)

    # Warm-up
    for _ in range(10):
        _ = model(input)

    # Measure
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_iterations):
        _ = model(input)
    torch.cuda.synchronize()
    end_time = time.time()

    return (end_time - start_time) / num_iterations * 1000  # ms
