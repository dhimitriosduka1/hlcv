from copy import deepcopy
from functools import partial

import torch
import torch.nn as nn
from numpy import linspace
from src.data_loaders.data_modules import CIFAR10DataModule
from src.models.cnn.metric import TopKAccuracy
from src.models.cnn.model import ConvNet
from src.trainers.cnn_trainer import CNNTrainer

q1_experiment = dict(
    name="CIFAR10_CNN",
    model_arch=ConvNet,
    model_args=dict(
        input_size=3,
        num_classes=10,
        hidden_layers=[128, 512, 512, 512, 512, 512],
        activation=nn.ReLU,
        norm_layer=nn.Identity,
        drop_prob=0.0,
    ),
    datamodule=CIFAR10DataModule,
    data_args=dict(
        data_dir="data/exercise-2",  # You may need to change this for Colab.
        transform_preset="CIFAR10",
        batch_size=200,
        shuffle=True,
        heldout_split=0.1,
        num_workers=6,
    ),
    optimizer=partial(
        torch.optim.Adam,
        lr=0.002,
        weight_decay=0.001,
        amsgrad=True,
    ),
    lr_scheduler=partial(torch.optim.lr_scheduler.StepLR, step_size=5, gamma=0.8),
    criterion=nn.CrossEntropyLoss,
    criterion_args=dict(),
    metrics=dict(
        top1=TopKAccuracy(k=1),
        top5=TopKAccuracy(k=5),
    ),
    trainer_module=CNNTrainer,
    trainer_config=dict(
        n_gpu=1,
        epochs=50,
        eval_period=1,
        save_dir="Saved",
        save_period=10,
        monitor="max eval_top1", # Template: "monitor_mode monitor_metric"
        early_stop=0,
        log_step=100,
        tensorboard=True,
        wandb=True,
    ),
)


#########  TODO #####################################################
#  You would need to create the following config dictionaries       #
#  to use them for different parts of Q2 and Q3.                    #
#  Feel free to define more config files and dictionaries if needed.#
#  But make sure you have a separate config for every question so   #
#  that we can use them for grading the assignment.                 #
#####################################################################
q2a_normalization_experiment = deepcopy(q1_experiment)
q2a_normalization_experiment["name"] = "CIFAR10_CNN_BN"
q2a_normalization_experiment["model_args"]["norm_layer"] = nn.BatchNorm2d
q2a_normalization_experiment["trainer_config"]["epochs"] = 50

q2c_earlystop_experiment = ()

q3a_aug1_experiment = deepcopy(q2a_normalization_experiment)
q3a_aug1_experiment["name"] = "CIFAR10_CNN_Geo_Aug"
q3a_aug1_experiment["data_args"]["transform_preset"] = "CIFAR10_geo_aug"
q3a_aug1_experiment["trainer_config"]["epochs"] = 30

q3a_aug2_experiment = deepcopy(q2a_normalization_experiment)
q3a_aug2_experiment["name"] = "CIFAR10_CNN_Col_Aug"
q3a_aug2_experiment["data_args"]["transform_preset"] = "CIFAR10_col_aug"
q3a_aug2_experiment["trainer_config"]["epochs"] = 30

q3a_aug3_experiment = deepcopy(q2a_normalization_experiment)
q3a_aug3_experiment["name"] = "CIFAR10_CNN_Geo_Col_Aug"
q3a_aug3_experiment["data_args"]["transform_preset"] = "CIFAR10_geo_col_aug"
q3a_aug3_experiment["trainer_config"]["epochs"] = 30

q3b_dropout_experiment = deepcopy(q2a_normalization_experiment)
q3b_dropout_experiment["name"] = "cnn_q3b_dropout"
q3b_dropout_experiment["model_args"]["drop_prob"] = linspace(0.1, 0.9, 9).tolist()
q3b_dropout_experiment["trainer_config"]["epochs"] = 30
q3b_dropout_experiment["trainer_config"]["wandb"] = True
