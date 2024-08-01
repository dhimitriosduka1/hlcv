import os
import subprocess

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)

# Specify your variables here
data_path = os.path.join(CURRENT_DIR, "data", "coco-custom.yaml")
img_size = 640
batch_size = 32
conf = 0.001
iou = 0.7
device = 0
weights_path = os.path.join(
    PARENT_DIR, "final_models", "fasterrcnn_mobilenet_v3_large_fpn_finetuned.pt"
)
save_dir = "fasterrcnn_mobilenet_v3_large_fpn_finetuned"

# Construct the command
command = [
    "python",
    os.path.join(CURRENT_DIR, "val.py"),
    "--data",
    data_path,
    "--img",
    str(img_size),
    "--batch",
    str(batch_size),
    "--conf",
    str(conf),
    "--iou",
    str(iou),
    "--device",
    str(device),
    "--weights",
    weights_path,
    "--verbose",
    "--save-json",
    "--name",
    save_dir,
]

# Run the command
try:
    subprocess.run(command, check=True)
    print("Validation completed successfully.")
except subprocess.CalledProcessError as e:
    print(f"An error occurred while running the validation: {e}")
