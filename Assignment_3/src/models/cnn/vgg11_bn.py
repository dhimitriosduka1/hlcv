from torchvision.models import vgg11_bn
import torch
import torch.nn as nn

from ..base_model import BaseModel


class VGG11_bn(BaseModel):
    def __init__(self, layer_config, num_classes, activation, norm_layer, fine_tune, weights=None):
        super(VGG11_bn, self).__init__()

        ####### TODO ###########################################################
        # Initialize the different model parameters from the config file       #
        # Basically store them in `self`                                       #
        # Note that this config is for the MLP head (and not the VGG backbone) #
        ########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        self.layer_config = layer_config
        self.num_classes = num_classes
        self.activation = activation
        self.norm_layer = norm_layer
        self.fine_tune = fine_tune
        self.weights = weights

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self._build_model()

    def _build_model(self):
        #################################################################################
        # TODO: Build the classification network described in Q4 using the              #
        # models.vgg11_bn network from torchvision model zoo as the feature extraction  #
        # layers and two linear layers on top for classification. You can load the      #
        # pretrained ImageNet weights based on the weights variable. Set it to None if  #
        # you want to train from scratch, 'DEFAULT'/'IMAGENET1K_V1' if you want to use  #
        # pretrained weights. You can either write it here manually or in config file   #
        # You can enable and disable training the feature extraction layers based on    # 
        # the fine_tune flag.                                                           #
        #################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        vgg = vgg11_bn(weights=self.weights)

        # Freeze the feature extraction layers if fine_tune is False
        if not self.fine_tune:
            for param in vgg.parameters():
                param.requires_grad = False

        self.features = vgg.features
        self.avgpool = vgg.avgpool
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(vgg.classifier[0].in_features, self.layer_config[0]),
            self.norm_layer(self.layer_config[0]),
            self.activation(),
            nn.Linear(self.layer_config[0], self.layer_config[1]),
            self.norm_layer(self.layer_config[1]),
            self.activation(),
            nn.Linear(self.layer_config[1], self.num_classes)
        )

        self.layers = nn.Sequential(
            self.features,
            self.avgpool,
            self.flatten,
            self.classifier
        )

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def forward(self, x):
        #################################################################################
        # TODO: Implement the forward pass computation.                                 #
        # Do not apply any softmax on the output                                        #
        #################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        out = self.layers(x)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return out