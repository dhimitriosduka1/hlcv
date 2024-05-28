import torch.nn as nn
import numpy as np
from abc import abstractmethod


class BaseModel(nn.Module):
    """
    Base class for all models
    """
    @abstractmethod # To be implemented by child classes.
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """

        ret_str = super().__str__()
    
        #### TODO #######################################
        # Print the number of **trainable** parameters  #
        # by appending them to ret_str                  #
        #################################################
        
        
        return ret_str