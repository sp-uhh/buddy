import torch.nn as nn
import abc
### Abstract class

class Operator(nn.Module):

    @abc.abstractmethod
    def degradation(self, *args, **kwargs):
        '''
        Forward Pass for degradation with given parameters
        '''
        pass

    @abc.abstractmethod
    def update_params(self, *args, **kwargs):
        '''
        Method for updating parameteres in blind scenarios or when loading new settings with same class
        '''
        pass

    def prepare_optimization(self, x_den, y):
        """
        Some preprocessing for optimizing the parameters. Empty by default
        """
        return x_den, y

    def constrain_params(self):
        pass
