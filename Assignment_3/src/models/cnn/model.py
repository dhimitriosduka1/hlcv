import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

from ..base_model import BaseModel


class ConvNet(BaseModel):
    def __init__(
        self,
        input_size,
        hidden_layers,
        num_classes,
        activation,
        norm_layer,
        drop_prob=0.0,
    ):
        super(ConvNet, self).__init__()

        ############## TODO ###############################################
        # Initialize the different model parameters from the config file  #
        # (basically store them in self)                                  #
        ###################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.num_classes = num_classes
        self.activation = activation
        self.norm_layer = norm_layer
        self.drop_prob = drop_prob

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self._build_model()

    def _build_model(self):

        #################################################################################
        # TODO: Initialize the modules required to implement the convolutional layer    #
        # described in the exercise.                                                    #
        # For Q1.a make use of conv2d and relu layers from the torch.nn module.         #
        # For Q2.a make use of BatchNorm2d layer from the torch.nn module.              #
        # For Q3.b Use Dropout layer from the torch.nn module if drop_prob > 0          #
        # Do NOT add any softmax layers.                                                #
        #################################################################################
        layers = []
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        CONV_KERNEL_SIZE = 3
        CONV_STRIDE = 1
        CONV_PADDING = 1

        POOLING_KERNEL_SIZE = 2
        POOLING_STRIDE = 2

        input_size = self.input_size

        # Features
        for i in range(len(self.hidden_layers) - 1):  # -1 ignores the FC layer
            layers.append(
                nn.Conv2d(
                    in_channels=input_size,
                    out_channels=self.hidden_layers[i],
                    kernel_size=CONV_KERNEL_SIZE,
                    stride=CONV_STRIDE,
                    padding=CONV_PADDING,
                )
            )

            # Add normalization
            if self.norm_layer == nn.BatchNorm2d:
                layers.append(self.norm_layer(self.hidden_layers[i]))
            elif self.norm_layer == nn.Identity:
                layers.append(self.norm_layer())
            else:
                raise("Norm type not supported!")

            # Add MaxPool with kernel size and stride of 2
            layers.append(
                nn.MaxPool2d(kernel_size=POOLING_KERNEL_SIZE, stride=POOLING_STRIDE)
            )

            # Add activation function
            layers.append(self.activation())

            # Add dropout if drop_prob is provided
            if self.drop_prob > 0:
                layers.append(nn.Dropout(p=self.drop_prob))

            input_size = self.hidden_layers[i]

        # Classification
        layers.append(
            nn.Flatten()
        )  # self.hidden_layers[-1]x1x1 -> self.hidden_layers[-1]
        layers.append(nn.Linear(self.hidden_layers[-1], self.num_classes))

        self.layers = nn.Sequential(*layers)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def _normalize(self, img):
        """
        Helper method to be used for VisualizeFilter.
        This is not given to be used for Forward pass! The normalization of Input for forward pass
        must be done in the transform presets.
        """
        max = np.max(img)
        min = np.min(img)
        return (img - min) / (max - min)

    def VisualizeFilter(self):
        ################################################################################
        # TODO: Implement the functiont to visualize the weights in the first conv layer#
        # in the model. Visualize them as a single image fo stacked filters.            #
        # You can use matlplotlib.imshow to visualize an image in python                #
        #################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ROWS = 8
        COLUMS = 16

        fig, axes = plt.subplots(ROWS, COLUMS, figsize=(COLUMS, ROWS))

        for i in range(self.hidden_layers[0]):
            k = i // COLUMS
            j = i % COLUMS

            normalized = self._normalize(
                self.layers[0].weight[i].cpu().detach().numpy()
            )

            axes[k, j].imshow(normalized)
            axes[k, j].axis("off")

        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.show()

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def forward(self, x):
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        out = self.layers(x)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return out
