from warnings import warn
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from torchvision.utils import draw_bounding_boxes

from common.utils import as_255, create_color_map, find_font, normalize_color_map


def draw_colored_boxes(
    image: torch.Tensor | np.ndarray,
    boxes: torch.Tensor | np.ndarray,
    classes: list[int | str] | None = None,
    class_names: list[str] = None,
    width=2,
    font_size=10,
    font="/usr/share/fonts/google-noto-vf/NotoSansMono[wght].ttf",
    annotate_classes=True,
    show_colorbar=True,
    legend_fontsize_ratio=1,
    only_existing_classes_in_colorbar=True,
    show=True,
) -> torch.Tensor:
    if annotate_classes and not font:
        font = find_font(("Noto", "Sans", "Mono"))
        if not font:
            warn("Unable to find a suitable font for labels. Please, specify a font path.")

    if class_names is None and classes is not None:
        class_names = np.unique(classes).tolist()

    if class_names is not None:
        color_map = create_color_map(class_names)

    if classes is not None and isinstance(classes[0], int):
        classes = [class_names[cls] for cls in classes]

    if classes is not None:
        box_colors = [color_map[cls] for cls in classes]
        rgb_colors = [(r, g, b) for r, g, b in box_colors]
    else:
        rgb_colors = "red"

    # Test if the image is in the range [0, 1] and convert it to [0, 255] if necessary
    if image.min() >= 0 and image.max() <= 1:
        image = as_255(image)

    # Draw bounding boxes
    drawn_boxes = draw_bounding_boxes(
        torch.as_tensor(image),
        torch.as_tensor(boxes),
        labels=classes if annotate_classes else None,
        colors=rgb_colors,
        width=width,
        font_size=font_size,
        font=font,
    )

    # Convert drawn_boxes to numpy array for matplotlib
    drawn_boxes_np = drawn_boxes.permute(1, 2, 0).numpy()

    # Plot the image with bounding boxes
    if show:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(drawn_boxes_np)
        ax.axis("off")

        # Add a legend for the classes
        if class_names is not None and show_colorbar and classes is not None:
            show_classes = class_names
            if only_existing_classes_in_colorbar:
                show_classes = np.unique(classes).tolist()

            # Create a list of patches to display in the legend
            color_map = normalize_color_map(color_map)
            legend_patches = [
                Patch(color=tuple(color_map[class_name]), label=class_name)
                for class_name in show_classes
            ]

            # Add the legend to the plot
            ax.legend(
                handles=legend_patches,
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                ncol=2 if len(show_classes) > 40 else 1,
                fontsize=font_size * legend_fontsize_ratio,
                frameon=False,
            )

        fig.tight_layout()
        plt.show()

    return drawn_boxes
