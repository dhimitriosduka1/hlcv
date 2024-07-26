import json
import os
import shutil

from common.utils import download_from

# Dataset name
CUSTOM_DATASET_NAME = "guitar-necks-detector"

# Constants
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(CURRENT_DIR, "data")
COCO_DIR = os.path.join(DATASETS_DIR, "coco")
CUSTOM_DATASETS_DIR = os.path.join(DATASETS_DIR, CUSTOM_DATASET_NAME)
CUSTOM_COCO_DIR = os.path.join(DATASETS_DIR, "coco_custom")

# Category ID and Class ID for the new category
NEW_CATEGORY_ID = 91  # This is the ID of the new category in the COCO dataset (JSON annotations)
NEW_CLASS_ID = 80  # This is the ID of the new class in the label files (TXT files)


def short(full_path: str, parts_to_keep=2) -> str:
    """Returns the last n parts of a file path.

    :param full_path: The full file path.
    :param parts_to_keep: Number of path parts to keep, starting from the end. Default is 2.
    :return: A string with the last n parts of the path."""
    path_parts = full_path.split(os.sep)
    relative_parts = path_parts[-parts_to_keep:]
    relative_path = os.path.join(*relative_parts)
    return relative_path


def download_custom_dataset() -> None:
    """Download the custom dataset."""
    if os.path.exists(os.path.join(DATASETS_DIR, CUSTOM_DATASET_NAME)):
        print(f"The directory {CUSTOM_DATASET_NAME} already exists. Skipping download.")
        return

    config = {
        "data": {
            "dataset": CUSTOM_DATASET_NAME,
            "load": {
                "interface": "roboflow",
                "workspace": "hubert-drapeau-qt6ae",
                "project-version": "1",
                "version-download": "yolov9",
            },
        }
    }

    # Download first the YOLOv9 version of the dataset
    download_from(config, os.path.join(DATASETS_DIR, "yolov9"))

    # Then, download the COCO version of the dataset
    config["data"]["load"]["version-download"] = "coco"
    download_from(config, os.path.join(DATASETS_DIR, "coco"))


def combine_custom_datasets() -> None:
    if os.path.exists(os.path.join(DATASETS_DIR, CUSTOM_DATASET_NAME)):
        print(f"The directory {CUSTOM_DATASET_NAME} already exists. Skipping combining.")
        return

    # This will copy the .json files from the COCO version to the YOLOv9 version
    # and then delete the COCO version
    for set_ in ["train", "valid", "test"]:
        for file_ in os.listdir(os.path.join(DATASETS_DIR, "coco", CUSTOM_DATASET_NAME, set_)):
            if file_.endswith(".json"):
                coco_json = os.path.join(DATASETS_DIR, "coco", CUSTOM_DATASET_NAME, set_, file_)
                yolo_json = os.path.join(
                    DATASETS_DIR, "yolov9", CUSTOM_DATASET_NAME, set_, "labels", file_
                )
                shutil.copy(coco_json, yolo_json)

    # Delete the COCO version
    shutil.rmtree(os.path.join(DATASETS_DIR, "coco", CUSTOM_DATASET_NAME))

    # If os.path.join(DATASETS_DIR, CUSTOM_DATASET_NAME) exists, delete it
    if os.path.exists(os.path.join(DATASETS_DIR, CUSTOM_DATASET_NAME)):
        shutil.rmtree(os.path.join(DATASETS_DIR, CUSTOM_DATASET_NAME))

    # Move the YOLOv9 version to the main directory
    shutil.move(
        os.path.join(DATASETS_DIR, "yolov9", CUSTOM_DATASET_NAME),
        os.path.join(DATASETS_DIR, CUSTOM_DATASET_NAME),
    )

    # Delete the YOLOv9 version
    shutil.rmtree(os.path.join(DATASETS_DIR, "yolov9"))


def copy_coco_folder() -> bool:
    """Copy the COCO dataset in the same folder."""
    # Make sure data/coco exists
    if not os.path.exists(COCO_DIR):
        from validate.download_coco import main as download_coco

        download_coco()

    if not os.path.exists(CUSTOM_COCO_DIR):
        shutil.copytree(COCO_DIR, CUSTOM_COCO_DIR)
        print(f"Copied {COCO_DIR} to {CUSTOM_COCO_DIR}")
        return True
    else:
        print(f"{CUSTOM_COCO_DIR} already exists. Skipping copy.")
        return False


def merge_annotations(
    subset: tuple[str, str],
    new_category_id=NEW_CATEGORY_ID,
    new_category_name="fretboard",
    new_category_supercategory="guitar",
) -> bool:
    """Merge the annotations of the custom dataset with the COCO dataset."""
    if not os.path.exists(CUSTOM_DATASETS_DIR):
        print(f"The directory {CUSTOM_DATASETS_DIR} to merge with COCO does not exist.")
        return

    coco_subset = subset[0]
    custom_subset = subset[1]

    coco_json_path = os.path.join(
        CUSTOM_COCO_DIR, "annotations", f"instances_{coco_subset}2017.json"
    )
    custom_json_path = os.path.join(
        CUSTOM_DATASETS_DIR, custom_subset, "labels", "_annotations.coco.json"
    )

    if os.path.exists(coco_json_path) and os.path.exists(custom_json_path):
        with open(coco_json_path, "r") as f:
            coco_data = json.load(f)
        with open(custom_json_path, "r") as f:
            custom_data = json.load(f)

        print(
            f"\tLoaded coco_data with {len(coco_data['images'])} images, "
            + f"{len(coco_data['annotations'])} annotations, and "
            + f"{len(coco_data['categories'])} categories."
        )
        print(
            f"\tLoaded custom_data with {len(custom_data['images'])} images "
            + f"and {len(custom_data['annotations'])} annotations."
        )

        # Merge images
        max_image_id = max(img["id"] for img in coco_data["images"])
        for img in custom_data["images"]:
            img["id"] += max_image_id
            coco_data["images"].append(img)

        # Merge annotations
        max_ann_id = max(ann["id"] for ann in coco_data["annotations"])
        for ann in custom_data["annotations"]:
            ann["id"] += max_ann_id
            ann["image_id"] += max_image_id
            ann["category_id"] = new_category_id
            coco_data["annotations"].append(ann)

        # Add new category if not present
        if not any(cat["id"] == new_category_id for cat in coco_data["categories"]):
            coco_data["categories"].append(
                {
                    "id": new_category_id,
                    "name": new_category_name,
                    "supercategory": new_category_supercategory,
                }
            )

        print(
            f"\tFinal custom coco_data with {len(coco_data['images'])} images and "
            f"{len(coco_data['annotations'])} annotations, saved to {short(coco_json_path)}"
        )

        # Save merged JSON
        with open(coco_json_path, "w") as f:
            json.dump(coco_data, f, indent=4)

        print(f"\tMerged annotations for {subset}")
        return coco_data
    else:
        print(f"\tEither {short(coco_json_path)} or {short(custom_json_path)} does not exist.")
        return None


def copy_and_modify_files(subset: str, new_class_id=NEW_CLASS_ID) -> bool:
    """Copy and modify the image and label files for the custom dataset."""

    def replace_first_number(line: str) -> str:
        """Replace the first number in a line with the `new_class_id`."""
        parts = line.split()
        if parts:
            parts[0] = str(new_class_id)
        return " ".join(parts)

    coco_subset = subset[0]
    custom_subset = subset[1]

    src_img_dir = os.path.join(CUSTOM_DATASETS_DIR, custom_subset, "images")
    src_label_dir = os.path.join(CUSTOM_DATASETS_DIR, custom_subset, "labels")
    dst_img_dir = os.path.join(CUSTOM_COCO_DIR, "images", f"{coco_subset}2017")
    dst_label_dir = os.path.join(CUSTOM_COCO_DIR, "labels", f"{coco_subset}2017")

    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_label_dir, exist_ok=True)

    already_copied_files = True
    for filename in os.listdir(src_img_dir):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp")):
            # Copy image file
            shutil.copy(os.path.join(src_img_dir, filename), dst_img_dir)

            # Add image to the {coco_subset}2017.txt file
            with open(os.path.join(CUSTOM_COCO_DIR, f"{coco_subset}2017.txt"), "a") as f:
                f.write(f"./images/{coco_subset}2017/{filename}\n")

            # Process corresponding label file
            label_filename = os.path.splitext(filename)[0] + ".txt"
            src_label_path = os.path.join(src_label_dir, label_filename)
            dst_label_path = os.path.join(dst_label_dir, label_filename)

            previous_category_id = set()
            if os.path.exists(src_label_path):
                with open(src_label_path, "r") as f:
                    lines = f.readlines()

                # Replace the first number in each line
                previous_category_id.add(int(lines[0].split(" ")[0]))
                if len(previous_category_id) > 1:
                    print(f"Multiple categories found in {src_label_path}. Not implemented.")
                    exit(1)

                # Replace the first number in each line
                new_lines = [replace_first_number(line) for line in lines]

                # Write the modified content back to the file
                with open(dst_label_path, "w") as f:
                    f.writelines(new_lines)
            else:
                print(f"\tLabel file {src_label_path} does not exist!!!")
                already_copied_files = False

    print(f"\tCopied images for {subset}")
    if already_copied_files:
        print(f"\tLabels already copied for {subset}")
    return already_copied_files


def delete_cache_file(file_="val2017.cache") -> None:
    """Delete the cache file to force the program to recompute it."""
    cache_file = os.path.join(CUSTOM_COCO_DIR, file_)
    if os.path.exists(cache_file):
        os.remove(cache_file)
        print(f"Deleted cache file: {cache_file}")


def main() -> None:
    download_custom_dataset()
    combine_custom_datasets()
    success = copy_coco_folder()

    if success:
        for subset in [("train", "train"), ("val", "valid"), ("test", "test")]:
            print(f"Processing {subset}... ")
            _ = merge_annotations(subset)
            _ = copy_and_modify_files(subset)
            print("... Done.")
        delete_cache_file()


if __name__ == "__main__":
    main()
