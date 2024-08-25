import os
import numpy as np
import scienceplots
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap import UMAP
from PIL import Image

plt.style.use(["science", "ieee", "no-latex", "std-colors"])

def load_and_flatten_image(image_path):
    img = Image.open(image_path).resize((32, 32))  # Resize to a manageable size
    img_array = np.array(img).flatten()
    return img_array / 255.0  # Normalize to [0, 1]

ds_name = "merged_ds"
# ds_name = "guitar-chords-tiny"
# ds_name = "guitar-chords-ours-A-G"

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
print("Applying t-SNE...")
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(data)

# Apply PCA
print("Applying PCA...")
pca = PCA(n_components=2, random_state=42)
pca_result = pca.fit_transform(data)

# Apply UMAP
print("Applying UMAP...")
umap = UMAP(n_components=2, random_state=42)
umap_result = umap.fit_transform(data)

dataset_real_name = "Guitar-chords (Merged)"
# dataset_real_name = "Guitar-chords-tiny"
# dataset_real_name = "Guitar-chords-ours"

# Function to plot and save results
def plot_and_save(result, method_name):
    plt.figure(figsize=(6, 4))
    for class_dir in class_dirs:
        mask = labels == class_dir
        plt.scatter(result[mask, 0], result[mask, 1], label=class_dir, s=5)
    plt.legend()
    plt.title(f'{dataset_real_name} {method_name} Visualization')
    plt.savefig(f'{dataset_real_name}_{method_name.lower()}_plot.png', dpi=300)
    plt.close()

# Plot and save t-SNE, PCA, and UMAP results
plot_and_save(tsne_result, 't-SNE')
plot_and_save(pca_result, 'PCA')
plot_and_save(umap_result, 'UMAP')

print("Plots have been saved as 't-sne_plot.png', 'pca_plot.png', and 'umap_plot.png'")