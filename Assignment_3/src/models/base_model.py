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
        total = 0
        for i, (name, param) in enumerate(self.named_parameters()):
            ret_str += f'\nNumber of trainable parameters of {name}:\t{np.prod(param.size())}'
            total += np.prod(param.size())

        ret_str += f'\nTotal number of trainable parameters:\t{total}'

        return ret_str