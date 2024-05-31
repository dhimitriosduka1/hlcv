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
        
        max_name_length = 0
        max_number_length = 0
        total_trainable_params = 0

        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue

            if len(name) > max_name_length:
                max_name_length = len(name)

            if len(str(parameter.numel())) > max_number_length:
                max_number_length = len(str(parameter.numel()))

        header = '| {:<{name_width}} | {:<{number_width}} |\n'.format(
            'Layer Name', 
            'Params', 
            name_width=max_name_length, 
            number_width=max_number_length
        )

        separator = '| {} | {} |\n'.format('-' * max_name_length, '-' * max_number_length)

        ret_str = header + separator
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
                
            number_of_params = parameter.numel()
            total_trainable_params += number_of_params 

            ret_str += '| {:<{name_width}} | {:<{number_width}} |\n'.format(
                name, 
                number_of_params, 
                name_width=max_name_length, 
                number_width=max_number_length
            )

        ret_str += f'\nTotal number of trainable parameters: {total_trainable_params}\n'        
        return ret_str