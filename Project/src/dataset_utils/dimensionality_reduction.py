from gc import collect as collect_garbage
from importlib.util import find_spec as importlib_find_spec

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from umap import UMAP

OPENTSNE_INSTALLED = importlib_find_spec("opentsne") is not None
TSNE_CUDA_INSTALLED = importlib_find_spec("tsnecuda") is not None

# Prefer using opentsne instead of sklearn.manifold.TSNE because it is faster by using the FFT method
# Install with: conda install --channel conda-forge opentsne
# See: https://github.com/pavlin-policar/openTSNE/
if OPENTSNE_INSTALLED:
    from openTSNE import TSNE
else:
    from sklearn.manifold import TSNE

if TSNE_CUDA_INSTALLED:
    # If `use_tsnecuda` is specified when calling `apply_tsne`, then we use the CUDA version of t-SNE
    # See: https://github.com/CannyLab/tsne-cuda/blob/main/INSTALL.md
    # Install with: conda install tsnecuda -c conda-forge
    from tsnecuda import TSNE as TSNE_CUDA  # Import the CUDA version of t-SNE


def apply_pca(
    X: np.ndarray | pd.DataFrame,
    n_components: int = 2,
    random_state: int = 42,
    standardize_first=True,
) -> np.ndarray:
    """Apply PCA to reduce the dimensionality of the given high-dimensional array."""
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()

    pipeline = PCA(n_components=n_components, random_state=random_state)
    if standardize_first:
        pipeline = make_pipeline(StandardScaler(), pipeline)

    return pipeline.fit_transform(X)


def apply_tsne(
    X: np.ndarray | pd.DataFrame,
    n_components: int = 2,
    perplexity: float = None,
    learning_rate: float = None,
    random_state: int = 42,
    use_tsnecuda: bool = False,
    verbose: bool = True,
) -> np.ndarray:
    """Apply t-SNE to reduce the dimensionality of the given high-dimensional array.

    The following parameters are set based on the number of samples in the data and the
    recommendations cited from various sources:
    * Uncertain Choices in Method Comparisons: An Illustration with t-SNE and UMAP (2023)
      See: https://epub.ub.uni-muenchen.de/107259/1/BA_Weber_Philipp.pdf
    * New guidance for using t-SNE: Alternative defaults, hyperparameter selection automation, and comparative
      evaluation (2022)
      See: https://www.sciencedirect.com/science/article/pii/S2468502X22000201
    """
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()

    n = X.shape[0]
    learning_rate = max(200, int(n / 12)) if learning_rate is None else learning_rate
    perplexity = max(30, int(n / 100)) if perplexity is None else perplexity

    if verbose:
        print(
            f"Applying t-SNE (perplexity={perplexity}, learning_rate={learning_rate}) using ",
            end="",
        )

    collect_garbage()  # Collect garbage

    if use_tsnecuda and TSNE_CUDA_INSTALLED:
        print("tsnecuda")
        tsne_embedded = TSNE_CUDA(
            n_components=n_components,
            perplexity=perplexity,
            learning_rate=learning_rate,
            random_seed=random_state,
        ).fit_transform(X)
    else:
        print("opentsne")
        tsne_embedded = TSNE(
            n_components=n_components,
            n_jobs=32,
            perplexity=perplexity,
            learning_rate=learning_rate,
            random_state=random_state,
        ).fit_transform(X)

    return tsne_embedded


def apply_lda(
    X: np.ndarray | pd.DataFrame,
    labels: np.ndarray,
    n_components: int = 2,
    labels_are_indices: bool = False,
    standardize_first: bool = True,
) -> np.ndarray:
    """
    Apply LDA to reduce the dimensionality of the given high-dimensional array.

    Parameters:
    -----------
    X : np.ndarray | pd.DataFrame
        The input data to be transformed.
    indices : list
        A list of boolean arrays or index arrays indicating different groups/clusters.
    n_components : int, optional
        Number of components to keep. Default is 2.
    labels_are_indices : bool, optional
        Whether the labels are indices. This is useful if the labels are constructed by applying a boolean mask to the
        data. For example, 

        ```
        >>> X = np.random.rand(100, 10)
        >>> indices = [X[:, 0] > 0.5, X[:, 0] <= 0.5]
        ```
        By default, False.
    standardize_first : bool, optional
        Whether to standardize the data before applying LDA. Default is True.

    Returns:
    --------
    np.ndarray
        The transformed data.
    """
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()

    # Create labels from indices
    if labels_are_indices:
        indices = labels
        labels = np.zeros(X.shape[0], dtype=int)
        for i, idx in enumerate(indices, 1):
            labels[idx] = i

        # Encode labels
        le = LabelEncoder()
        y = le.fit_transform(labels)
    else:
        y = labels

    pipeline = LinearDiscriminantAnalysis(n_components=n_components)
    if standardize_first:
        pipeline = make_pipeline(StandardScaler(), pipeline)

    return pipeline.fit_transform(X, y)


def apply_ica(
    X: np.ndarray | pd.DataFrame,
    n_components: int = 2,
    random_state: int = 42,
    standardize_first=True,
) -> np.ndarray:
    """Apply ICA to reduce the dimensionality of the given high-dimensional array."""

    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()

    if standardize_first:
        pipeline = make_pipeline(StandardScaler(), FastICA(n_components=n_components, random_state=random_state))
    else:
        pipeline = FastICA(n_components=n_components, random_state=random_state)
    return pipeline.fit_transform(X)


def apply_umap(
    X: np.ndarray | pd.DataFrame,
    n_components: int = 2,
    standardize_first: bool = True,
    **kwargs,
) -> np.ndarray:
    """Apply UMAP to reduce the dimensionality of the given high-dimensional array.

    Parameters:
    -----------
    X : np.ndarray | pd.DataFrame
        The input data to be transformed.
    n_components : int, optional
        Number of components to keep. Default is 2.
    standardize_first : bool, optional
        Whether to standardize the data before applying UMAP. Default is True.
    **kwargs : dict
        Additional keyword arguments for UMAP.

    Returns:
    --------
    np.ndarray
        The transformed data.
    """
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()

    umap = UMAP(
        n_components=n_components,
        **kwargs,
    )

    if standardize_first:
        pipeline = make_pipeline(StandardScaler(), umap)
    else:
        pipeline = umap

    return pipeline.fit_transform(X)
