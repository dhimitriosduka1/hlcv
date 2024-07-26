# Global configuration file
import logging
import multiprocessing
from importlib.util import find_spec as importlib_find_spec
from os import environ as os_environ
from os import getcwd as os_getcwd
from os.path import join as path_join
from re import search as re_search
from warnings import simplefilter as warnings_simplefilter
from warnings import warn

from pandas import options as pd_options

from common.utils import ensure_directory_exists, find_font

CPU_COUNT = multiprocessing.cpu_count()

CUML_INSTALLED = importlib_find_spec("cuml") is not None

if not CUML_INSTALLED:
    warn(
        "cuML is not installed, so we cannot use GPU-acceleration. Using sklearn's version instead."
    )

SKOPT_INSTALLED = importlib_find_spec("skopt") is not None

if not SKOPT_INSTALLED:
    warn("scikit-optimize is not installed. Using sklearn's GridSearchCV instead.")

# Suppress Numba CUDA driver debug and info messages
logging.getLogger("numba.cuda.cudadrv.driver").setLevel(logging.WARNING)

# Set the logging level for matplotlib to WARNING to suppress debug messages
logging.getLogger("matplotlib").setLevel(logging.WARNING)

pd_options.mode.chained_assignment = None  # default='warn'
warnings_simplefilter("ignore", RuntimeWarning)  # Ignore repeated instances of the same warning


class Config:
    """General configuration class for the project."""

    DATA_DIR = path_join(os_getcwd(), "data")
    ORIGINAL_MODELS_DIR = path_join(os_getcwd(), "original_models")
    FINAL_MODELS_DIR = path_join(os_getcwd(), "final_models")
    PLOTS_DIR = path_join(os_getcwd(), "plots")
    CONFIGS_DIR = path_join(os_getcwd(), "configs")
    IMAGES_DIR = path_join(os_getcwd(), "images")
    FONT = find_font(("Noto", "Sans", "Mono"))
    
    # Functions
    NUMERICAL_ORDERING_FUNC = lambda x: int(re_search(r"\d+", x).group())

    @classmethod
    def all_paths(cls):
        """Return all the paths in the configuration."""
        return cls.DATA_DIR, cls.ORIGINAL_MODELS_DIR, cls.FINAL_MODELS_DIR, cls.PLOTS_DIR

    @classmethod
    def ensure_directories_exist(cls):
        """Ensure that all the directories in the configuration exist."""
        for directory in cls.all_paths():
            ensure_directory_exists(directory)

    @classmethod
    def set_debugger_timeout(cls, t: float, verbose: bool = True):
        """Set a default timeout value for the current session."""
        timeout_value = t
        os_environ["PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT"] = str(timeout_value)
        if verbose:
            print(f"Set PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT to {timeout_value} seconds")


Config.ensure_directories_exist()
