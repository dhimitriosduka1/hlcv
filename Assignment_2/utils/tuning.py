"""
A module for classes and functions for hyperparameter tuning.
"""

# pylint: disable=C0103, W0718

import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from os import cpu_count

import numpy as np
from skimage.feature import hog
from sklearn.decomposition import PCA
from sympy import pretty_print
from tqdm import tqdm  # conda install conda-forge::tqdm

from models.twolayernet.model import TwoLayerNetv1
from utils.utils import pretty_dict


class NetGridSearchCV:
    """
    Class that mimics scikit-learn's GridSearchCV to perform grid search over a parameter grid to
    find the best hyperparameters for a given Neural Network using train-validation split.
    """

    def __init__(self, net: TwoLayerNetv1, param_grid: dict) -> None:
        """
        Initializes the NetGridSearchCV with the Neural Network `net` and parameter grid.

        Parameters
        ----------
        net : TwoLayerNetv1
            The TwoLayerNet to be optimized (includes the nets that inherit from this one:
            TwoLayerNetv1, TwoLayerNetv2, TwoLayerNetv3, TwoLayerNetv4)
        param_grid : dict
            A dictionary where keys are hyperparameters and values are lists of values to try.
            Example:
            ```
            param_grid = {
                "num_iters": [1000, 2000],
                "batch_size": [200, 100],
                "learning_rate": [1e-4, 1e-3],
                "learning_rate_decay": [0.95, 0.9],
                "reg": [0.25, 0.5],
            }
            ```
        """
        self.net = net
        self.param_grid = param_grid
        self.best_params_ = None
        self.best_score_ = None
        self.cv_results_ = []

    def _evaluate_combination(
        self, combination, X_train, y_train, X_val, y_val, input_size, hidden_size, output_size
    ):
        """
        Helper function to train and evaluate the model for a given combination of hyperparameters.
        """
        params = dict(zip(self.param_grid.keys(), combination))

        model = self.net(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
        stats = model.train(
            X_train,
            y_train,
            X_val,
            y_val,
            num_iters=params["num_iters"],
            batch_size=params["batch_size"],
            learning_rate=params["learning_rate"],
            learning_rate_decay=params["learning_rate_decay"],
            reg=params["reg"],
            verbose=False,
        )

        val_acc = stats["val_acc_history"][-1]
        return params, val_acc

    def _process_combination(self, args):
        """Helper function to unpack `args` and call the `_evaluate_combination` function."""
        return self._evaluate_combination(*args)

    def fit(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        input_size,
        hidden_size,
        output_size,
        verbose=False,
        n_jobs=-1,
    ):
        """
        Performs grid search over the parameter grid and finds the best hyperparameters.

        Parameters
        ----------
        X_train, y_train, X_val, y_val : np.ndarray
            Training data, training labels, validation data, and validation labels.
        input_size, hidden_size, output_size : int
            Sizes for input, hidden, and output layers of the neural network.
        verbose : bool, optional (default=False)
            If True, it will print the results of each combination in a line.
        n_jobs : int, optional (default=-1)
            The number of jobs to run in parallel. -1 means using all processors.
        """
        param_values = list(self.param_grid.values())
        all_combinations = list(product(*param_values))

        if n_jobs == -1:
            n_jobs = cpu_count()
        elif n_jobs < 0:
            n_jobs = max(1, cpu_count() + n_jobs)

        # Use ProcessPoolExecutor for parallel execution
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {
                executor.submit(
                    self._evaluate_combination,
                    combination,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    input_size,
                    hidden_size,
                    output_size,
                ): combination
                for combination in all_combinations
            }

            progress_bar = tqdm(
                total=len(all_combinations), desc="Grid Search Progress", file=sys.stdout
            )

            best_val_acc = -np.inf
            best_params = None

            for k, future in enumerate(as_completed(futures)):
                params, val_acc = future.result()
                self.cv_results_.append((params, val_acc))

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_params = params

                metrics_description = f"val_acc={val_acc:.3f}, best_val_acc={best_val_acc:.3f}"
                progress_bar.set_postfix_str(metrics_description)
                if verbose:
                    current_description = (
                        f"Combination ({k+1}/{len(all_combinations)}): {pretty_dict(params)}, "
                        + metrics_description
                    )
                    tqdm.write(current_description)

                progress_bar.update(1)
            progress_bar.close()

            self.best_params_ = best_params
            self.best_score_ = best_val_acc

            if verbose:
                tqdm.write(
                    f"\nBest parameters := {pretty_print(self.best_params_)}, best_val_acc={self.best_score_:.3f}"
                )

    def get_best_params(self):
        """Returns the best parameters found by the grid search."""
        if self.best_params_ is None:
            raise ValueError("You must call `fit` before `get_best_params`.")
        return self.best_params_


def apply_pca(X_train, X_val, X_test, variance=0.95) -> tuple:
    pca = PCA(n_components=variance, svd_solver="full")
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)
    return X_train_pca, X_val_pca, X_test_pca


def compute_color_histograms(X):
    num_bins = 32
    histograms = []
    for i in range(X.shape[0]):
        img = X[i].reshape(32, 32, 3)
        hist = np.concatenate(
            [np.histogram(img[:, :, j], bins=num_bins, range=(0, 256))[0] for j in range(3)]
        )
        histograms.append(hist)
    return np.array(histograms)


def rgb_to_grayscale(X):
    return np.dot(X[..., :3], [0.299, 0.587, 0.114])


def compute_hog_features(X):
    hog_features = []
    for i in range(X.shape[0]):
        img = X[i].reshape(32, 32, 3)
        gray = rgb_to_grayscale(img)
        hog_feature = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
        hog_features.append(hog_feature)
    return np.array(hog_features)


def add_features(X_train, X_val, X_test) -> tuple:
    # Add more features to the input data
    color_histograms_train = compute_color_histograms(X_train)
    color_histograms_val = compute_color_histograms(X_val)
    color_histograms_test = compute_color_histograms(X_test)

    hog_train = compute_hog_features(X_train)
    hog_val = compute_hog_features(X_val)
    hog_test = compute_hog_features(X_test)

    # Concatenate the original features with the color histograms
    X_train = np.hstack((X_train, color_histograms_train, hog_train))
    X_val = np.hstack((X_val, color_histograms_val, hog_val))
    X_test = np.hstack((X_test, color_histograms_test, hog_test))

    return X_train, X_val, X_test
