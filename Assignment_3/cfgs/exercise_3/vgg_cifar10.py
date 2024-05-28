from functools import partial

import torch
import torch.nn as nn

from src.data_loaders.data_modules import CIFAR10DataModule
from src.trainers.vgg_trainer import VGGTrainer
from src.models.cnn.metric import TopKAccuracy
from src.models.cnn.vgg11_bn import VGG11_bn

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
        epochs = 15,
        eval_period = 1,
        save_dir = "Saved",
        save_period = 10,
        monitor = "max eval_top1",
        early_stop = 5,

        log_step = 100,
        tensorboard = False,
        wandb = True
    ),
)
