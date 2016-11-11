'''
Created on 10.08.2016

@author: fgieseke
'''
import numpy as np
import lasagne
from nolearn.lasagne import NeuralNet, objective, TrainSplit

from lasagne.nonlinearities import softmax
from lasagne.layers import InputLayer
from lasagne.layers import Conv2DLayer
from lasagne.layers import MaxPool2DLayer
from lasagne.layers import DropoutLayer
from lasagne.layers import DenseLayer
        
def regularization_objective(layers, lambda1=0., lambda2=0., *args, **kwargs):
    """
    """
    
    # default loss function
    losses = objective(layers, *args, **kwargs)
    
    # get the layers' weights, but only those that should be regularized
    # (i.e. not the biases)
    weights = lasagne.layers.get_all_params(layers[-1], regularizable=True)
    
    # sum of absolute weights for L1
    sum_abs_weights = sum([abs(w).sum() for w in weights])
    
    # sum of squared weights for L2
    sum_squared_weights = sum([(w ** 2).sum() for w in weights])
    
    # add weights to regular loss
    losses += lambda1 * sum_abs_weights + lambda2 * sum_squared_weights
    
    return losses
                         

class AstroNet(NeuralNet):
    """Class that extends the nolearn NeuralNet class.
    Provides a simple way of loading a pre-defined
    network architecture for use with AstroWrapper.
        
    Parameters
    ----------

    net_type : string
        Name of the pre-defined network to be created.
    input_shape : tuple or list
        Shape of the input of the form
        (depth, width, height).
    output_size : integer
        Number of output nodes. Should be equal to the
        number of classes to predict.
    regression : boolean, default False
        Should be set to True when the labels are real
        values (regression) instead of distinct classes
        (classification).
    epochs : integer, default 100
        Number of times all input patterns are fed to
        the network during training.
    learning_rate : float, default 0.0002
        Parameter governing the speed of learning of 
        the network. Careful tuning is often necessary
        to reach a good performance. Value >0.
    verbose : integer, default 0
        Verbosity level.
    """

    def __init__(self, net_type, input_shape, output_size,
                     regression=False,
                     epochs=100, 
                     learning_rate=0.0002,
                     verbose=1):

        layers = self.get_layers(net_type, input_shape, output_size)
        NeuralNet.__init__(self,
                           layers=layers,
                           max_epochs=epochs,
                           regression=regression,
                           update=lasagne.updates.adam,
                           update_learning_rate=learning_rate,
                           objective_l2=0.0025,
                           train_split=TrainSplit(eval_size=0.05),
                           verbose=verbose,
                          )     
             

    def get_layers(self, net_type, shape, output_size):
        """ Method to select a pre-defined architecture.
            
        Parameters
        ----------
        net_type : string
            Name of the pre-defined network to be created.
        shape : tuple or list
            Shape of the input of the form
            (depth, width, height).
        output_size : integer
            Number of output nodes. Should be equal to the
            number of classes to predict.

        Returns
        -------
        layers : list
            The architecture of the selected pre-defined network.
        """
        if net_type == "ShallowNet":
            layers = [
                (InputLayer, {'shape': (None, shape[0], shape[1], shape[2])}),
            
                (Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3), 'pad': 0}),
                (MaxPool2DLayer, {'pool_size': (2, 2)}),
                (DropoutLayer, {'p':0.1}),
                                            
                (DenseLayer, {'num_units': 64}),
                (DropoutLayer, {'p':0.5}),
                (DenseLayer, {'num_units': 64}),
            
                (DenseLayer, {'num_units': output_size, 'nonlinearity': softmax}),
            ]

        elif net_type == "DeepNet":
            layers = [
                (InputLayer, {'shape': (None, shape[0], shape[1], shape[2])}),
            
                (Conv2DLayer, {'num_filters': 16, 'filter_size': (3, 3), 'pad': 0}),
                (MaxPool2DLayer, {'pool_size': (2, 2)}),
                (DropoutLayer, {'p':0.1}),

                (Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3), 'pad': 0}),
                (MaxPool2DLayer, {'pool_size': (2, 2)}),
                (DropoutLayer, {'p':0.3}),
                
                (Conv2DLayer, {'num_filters': 64, 'filter_size': (3, 3), 'pad': 0}),
                (MaxPool2DLayer, {'pool_size': (2, 2)}),
                (DropoutLayer, {'p':0.5}),                
            
                (DenseLayer, {'num_units': 1000}),
                (DropoutLayer, {'p':0.5}),
                (DenseLayer, {'num_units': 1000}),
            
                (DenseLayer, {'num_units': output_size, 'nonlinearity': softmax}),
            ]

        elif net_type == "SmallGZNet":

            layers = [
                (InputLayer, {'shape': (None, shape[0], shape[1], shape[2])}),
            
                (Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3), 'pad': 0}),
                (MaxPool2DLayer, {'pool_size': (2, 2)}),
                (DropoutLayer, {'p':0.1}),
                                            
                (DenseLayer, {'num_units': 64}),
                (DropoutLayer, {'p':0.5}),
                (DenseLayer, {'num_units': 64}),
            
                (DenseLayer, {'num_units': output_size, 'nonlinearity': None}),
            ]

        elif net_type == "Shallow1DNet":

            layers = [
                (InputLayer, {'shape': (None, shape[0], shape[1], shape[2])}),
            
                (Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 1), 'pad': 0}),
                (MaxPool2DLayer, {'pool_size': (2, 1)}),
                (DropoutLayer, {'p':0.1}),
                                            
                (DenseLayer, {'num_units': 64}),
                (DropoutLayer, {'p':0.5}),
                (DenseLayer, {'num_units': 64}),
            
                (DenseLayer, {'num_units': output_size, 'nonlinearity': softmax}),
            ]

        else:
            raise Exception("Unknown network type: %s" % str(net_type))

        return layers       
