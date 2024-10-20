from functools import partial

import torch
import torch.nn as nn

from src.data_loaders.data_modules import CIFAR10DataModule
from src.trainers.vgg_trainer import VGGTrainer
from src.models.cnn.metric import TopKAccuracy
from src.models.cnn.vgg11_bn import VGG11_bn
from copy import deepcopy

q4_dict = dict(
    name="CIFAR10_VGG",

    model_arch = VGG11_bn,
    model_args = dict(
        num_classes = 10,

        fine_tune=False,
        weights="IMAGENET1K_V1",

        # The following configs are not for the backbone (that's always a VGG). Rather,
        # they are the configuration for the MLP head that's applied afterwards!
        layer_config = [512, 256], 
        activation = nn.ReLU,
        norm_layer = nn.BatchNorm1d, # Not 2D! Because these are gonna be 1D features.

    ),

    datamodule = CIFAR10DataModule,
    data_args=dict(
        data_dir="data/exercise-2", # You may need to change this for Colab.
        transform_preset="CIFAR10_VGG",
        batch_size=64,
        shuffle=True,
        heldout_split=0.1,
        num_workers=6,
        training=True,
    ),
    
    optimizer=partial(
        torch.optim.Adam,
            lr=1e-3, weight_decay=1e-5,
    ),
    lr_scheduler=partial(
        torch.optim.lr_scheduler.ExponentialLR,
        gamma=0.99, last_epoch=-1, verbose=False,
    ),

    criterion = nn.CrossEntropyLoss,
    criterion_args = dict(),

    metrics=dict(
        top1 = TopKAccuracy(k=1),
        top5 = TopKAccuracy(k=5),
    ),

    trainer_module = VGGTrainer,
    trainer_config = dict(
        n_gpu = 1,
        epochs = 30,
        eval_period = 1,
        save_dir = "Saved",
        save_period = 10,
        monitor = "max eval_top1",
        early_stop = 5,
        log_step = 100,
        tensorboard = False,
        wandb = False
    ),
)

q4_dict_ft = deepcopy(q4_dict)
q4_dict_ft["model_args"]["fine_tune"] = True
q4_dict_ft["name"] = "CIFAR10_VGG_FineTune"

q4_dict_ft_nw = deepcopy(q4_dict)
q4_dict_ft_nw["model_args"]["fine_tune"] = True
q4_dict_ft_nw["model_args"]["weights"] = None
q4_dict_ft_nw["name"] = "CIFAR10_VGG_FineTune_NoWeights"

q4_dict_ft_list = []
for tp in ["CIFAR10_VGG_HF", "CIFAR10_VGG_ROT", "CIFAR10_VGG_PERSP", "CIFAR10_VGG_GEO_COMBINED", "CIFAR10_VGG_GEO_COL_COMBINED"]:
    config = deepcopy(q4_dict)
    config["data_args"]["transform_preset"]=tp
    config["model_args"]["fine_tune"] = True
    config["name"] = f"CIFAR10_VGG_FineTune{tp.split("_VGG")[-1]}"
    q4_dict_ft_list.append(config)

q4_dict_ft_nw_list = []
for tp in ["CIFAR10_VGG_HF", "CIFAR10_VGG_ROT", "CIFAR10_VGG_PERSP", "CIFAR10_VGG_GEO_COMBINED", "CIFAR10_VGG_GEO_COL_COMBINED"]:
    config = deepcopy(q4_dict)
    config["data_args"]["transform_preset"]=tp
    config["model_args"]["fine_tune"] = True
    config["model_args"]["weights"] = None
    config["name"] = f"CIFAR10_VGG_FineTune_NoWeights{tp.split('_VGG')[-1]}"
    q4_dict_ft_nw_list.append(config)