import os

from utils.general import check_dataset, ensure_directory_exists

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(CURRENT_DIR, "data")


def main() -> None:
    """Download the COCO dataset."""
    ensure_directory_exists(DATASETS_DIR)

    if os.path.exists(os.path.join(DATASETS_DIR, "coco")):
        print(f"COCO dataset already exists in {os.path.join(DATASETS_DIR, 'coco')}.")
    else:
        _ = check_dataset(os.path.join(DATASETS_DIR, "coco.yaml"))


if __name__ == "__main__":
    main()
