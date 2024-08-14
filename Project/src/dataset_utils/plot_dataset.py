import os
import numpy as np
import scienceplots
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from PIL import Image

plt.style.use(["science", "ieee", "no-latex", "std-colors"])

def load_and_flatten_image(image_path):
    img = Image.open(image_path).resize((32, 32))  # Resize to a manageable size
    img_array = np.array(img).flatten()
    return img_array / 255.0  # Normalize to [0, 1]

# ds_name = "merged_ds"
# ds_name = "guitar-chords-tiny"
ds_name = "guitar-chords-ours-A-G"

path = f"/home/dhimitriosduka/Documents/UdS/SoSe 2024/High-Level Computer Vision/Assignments/Datasets/Used Datasets/{ds_name}/"

# Directories
base_dirs = [f'{path}train', f'{path}test', f'{path}valid']
class_dirs = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

# Collect data and labels
data = []
labels = []

for base_dir in base_dirs:
    for class_dir in class_dirs:
        dir_path = os.path.join(base_dir, class_dir)
        for filename in os.listdir(dir_path):
            if filename.endswith('.jpg'):
                image_path = os.path.join(dir_path, filename)
                img_data = load_and_flatten_image(image_path)
                data.append(img_data)
                labels.append(class_dir)

data = np.array(data)
labels = np.array(labels)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(data)

# Apply PCA
pca = PCA(n_components=2, random_state=42)
pca_result = pca.fit_transform(data)

# dataset_real_name = "Guitar-chords (Merged)"
# dataset_real_name = "Guitar-chords-tiny"
dataset_real_name = "Guitar-chords-ours"

# Plot and save t-SNE
plt.figure(figsize=(6, 4))
for class_dir in class_dirs:
    mask = labels == class_dir
    plt.scatter(tsne_result[mask, 0], tsne_result[mask, 1], label=class_dir, s=5)
plt.legend()
plt.title(f'{dataset_real_name} t-SNE Visualization')
plt.savefig(f'src/dataset_utils/images/{dataset_real_name}_tsne_plot.png', dpi=300)
plt.close()

# Plot and save PCA
plt.figure(figsize=(6, 4))
for class_dir in class_dirs:
    mask = labels == class_dir
    plt.scatter(pca_result[mask, 0], pca_result[mask, 1], label=class_dir, s=5)
plt.legend()
plt.title(f'{dataset_real_name} PCA Visualization')
plt.savefig(f'src/dataset_utils/images/{dataset_real_name}_pca_plot.png', dpi=300)
plt.close()

print("Plots have been saved as 'tsne_plot.png' and 'pca_plot.png'")

