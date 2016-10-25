'''
Created on 10.08.2016

@author: fgieseke
'''
import numpy as np
import lasagne
from nolearn.lasagne import NeuralNet, objective, TrainSplit, BatchIterator

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

class NetGenerator(object):
    
    def __init__(self):
        
        pass
    
    def get_instance(self, net_type, shape, output_size,
                     regression=False,
                     epochs=100, 
                     learning_rate=0.0002,
                     batch_iterator_train=BatchIterator(128),
                     batch_iterator_test=BatchIterator(128),
                     verbose=0):

        if net_type == "astronet1":
                
            net = NeuralNet(
                        layers=[ 
                            ('input', lasagne.layers.InputLayer),
                            ('plotConv1', lasagne.layers.Conv2DLayer),
                            ('pool1', lasagne.layers.MaxPool2DLayer),
                            ('dropout1', lasagne.layers.DropoutLayer),
                            ('conv2', lasagne.layers.Conv2DLayer),
                            ('pool2', lasagne.layers.MaxPool2DLayer),
                            ('dropout2', lasagne.layers.DropoutLayer),
                            ('conv3', lasagne.layers.Conv2DLayer),
                            ('pool3', lasagne.layers.MaxPool2DLayer),
                            ('dropout3', lasagne.layers.DropoutLayer),
                            ('hidden4', lasagne.layers.DenseLayer),
                            ('dropout4', lasagne.layers.DropoutLayer),
                            ('hidden5', lasagne.layers.DenseLayer),
                            ('output', lasagne.layers.DenseLayer),
                            ],
                        input_shape=(None, shape[0], shape[1], shape[2]),
                        plotConv1_num_filters=16, plotConv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
                        dropout1_p=0.1,  # !
                        conv2_num_filters=32, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
                        dropout2_p=0.3,  # !
                        conv3_num_filters=64, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
                        dropout3_p=0.5,  # !
                        hidden4_num_units=1000,
                        dropout4_p=0.5,  # !
                        hidden5_num_units=1000,
                        output_num_units=output_size, output_nonlinearity=lasagne.nonlinearities.softmax,
                        objective=regularization_objective,
                        objective_lambda2=0.0025,
                        update=lasagne.updates.adam,
                        update_learning_rate=learning_rate,
                        max_epochs=epochs,
                        verbose=verbose,
                        train_split=TrainSplit(eval_size=0.05),
                        batch_iterator_train=batch_iterator_train,
                        batch_iterator_test =batch_iterator_test,
                        )
            return net
        
        elif net_type == "astronet2":

            layers2 = [
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

            net = NeuralNet(
                layers=layers2,
                max_epochs=epochs,
                update=lasagne.updates.adam,
                update_learning_rate=learning_rate,
                objective_l2=0.0025,
                batch_iterator_train=batch_iterator_train,
                batch_iterator_test =batch_iterator_test,
                train_split=TrainSplit(eval_size=0.05),
                verbose=verbose,
            )            
            
            return net
        
        elif net_type == "astronet3":

            layers3 = [
                (InputLayer, {'shape': (None, shape[0], shape[1], shape[2])}),
            
                (Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3), 'pad': 0}),
                (Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3), 'pad': 0}),
                (Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3), 'pad': 0}),
                (Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3), 'pad': 0}),
                (Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3), 'pad': 0}),
                (Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3), 'pad': 0}),
                (Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3), 'pad': 0}),
                (MaxPool2DLayer, {'pool_size': (2, 2)}),
            
                (Conv2DLayer, {'num_filters': 64, 'filter_size': (3, 3), 'pad': 0}),
                (Conv2DLayer, {'num_filters': 64, 'filter_size': (3, 3), 'pad': 0}),
                (Conv2DLayer, {'num_filters': 64, 'filter_size': (3, 3), 'pad': 0}),
                (MaxPool2DLayer, {'pool_size': (2, 2)}),
            
                (DenseLayer, {'num_units': 64}),
                (DropoutLayer, {'p':0.5}),
                (DenseLayer, {'num_units': 64}),
            
                (DenseLayer, {'num_units': output_size, 'nonlinearity': softmax}),
            ]

            net = NeuralNet(
                layers=layers3,
                max_epochs=epochs,
                update=lasagne.updates.adam,
                update_learning_rate=learning_rate,
                objective_l2=0.0025,
                train_split=TrainSplit(eval_size=0.05),
                batch_iterator_train=batch_iterator_train,
                batch_iterator_test =batch_iterator_test,
                verbose=verbose,
            )         
            
            return net   

        elif net_type == "astronet4":

            layers4 = [
                (InputLayer, {'shape': (None, shape[0], shape[1], shape[2])}),
            
                (Conv2DLayer, {'num_filters': 8, 'filter_size': (3, 3), 'pad': 0}),
                (Conv2DLayer, {'num_filters': 16, 'filter_size': (3, 3), 'pad': 0}),
                (MaxPool2DLayer, {'pool_size': (2, 2)}),
                
                (Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3), 'pad': 0}),
                (Conv2DLayer, {'num_filters': 64, 'filter_size': (3, 3), 'pad': 0}),
                (MaxPool2DLayer, {'pool_size': (2, 2)}),
            
                (DenseLayer, {'num_units': 128}),
                (DropoutLayer, {'p':0.5}),
                (DenseLayer, {'num_units': 128}),
            
                (DenseLayer, {'num_units': output_size, 'nonlinearity': softmax}),
            ]

            net = NeuralNet(
                layers=layers4,
                max_epochs=epochs,
                update=lasagne.updates.adam,
                update_learning_rate=learning_rate,
                objective_l2=0.0025,
                batch_iterator_train=batch_iterator_train,
                batch_iterator_test =batch_iterator_test,
                train_split=TrainSplit(eval_size=0.05),
                verbose=verbose,
            )     
            
            return net
            
        elif net_type == "astronet5":

            layers5 = [
                (InputLayer, {'shape': (None, shape[0], shape[1], shape[2])}),
            
                (Conv2DLayer, {'num_filters': 64, 'filter_size': (3, 3), 'pad': 0}),
                (MaxPool2DLayer, {'pool_size': (2, 2)}),
                (DropoutLayer, {'p':0.1}),
                
                (Conv2DLayer, {'num_filters': 128, 'filter_size': (3, 3), 'pad': 0}),
                (MaxPool2DLayer, {'pool_size': (2, 2)}),
                (DropoutLayer, {'p':0.1}),
                
                (Conv2DLayer, {'num_filters': 256, 'filter_size': (3, 3), 'pad': 0}),
                (MaxPool2DLayer, {'pool_size': (2, 2)}),
                (DropoutLayer, {'p':0.1}),                
            
                (DenseLayer, {'num_units': 128}),
                (DropoutLayer, {'p':0.5}),
                (DenseLayer, {'num_units': 128}),
            
                (DenseLayer, {'num_units': output_size, 'nonlinearity': softmax}),
            ]

            net = NeuralNet(
                layers=layers5,
                max_epochs=epochs,
                update=lasagne.updates.adam,
                update_learning_rate=learning_rate,
                objective_l2=0.0025,
                batch_iterator_train=batch_iterator_train,
                batch_iterator_test =batch_iterator_test,
                train_split=TrainSplit(eval_size=0.05),
                verbose=verbose,
            )     
            
            return net              

        elif net_type == "astronet6":

            layers6 = [
                (InputLayer, {'shape': (None, shape[0], shape[1], shape[2])}),
            
                (Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3), 'pad': 0}),
                (MaxPool2DLayer, {'pool_size': (2, 2)}),
                (DropoutLayer, {'p':0.1}),

                (Conv2DLayer, {'num_filters': 128, 'filter_size': (3, 3), 'pad': 0}),
                (MaxPool2DLayer, {'pool_size': (2, 2)}),
                (DropoutLayer, {'p':0.1}),
                                                            
                (DenseLayer, {'num_units': 512}),
                (DropoutLayer, {'p':0.5}),
                (DenseLayer, {'num_units': 512}),
            
                (DenseLayer, {'num_units': output_size, 'nonlinearity': softmax}),
            ]

            net = NeuralNet(
                layers=layers6,
                max_epochs=epochs,
                update=lasagne.updates.adam,
                update_learning_rate=learning_rate,
                objective_l2=0.0025,
                train_split=TrainSplit(eval_size=0.05),
                verbose=verbose,
                batch_iterator_train=batch_iterator_train,
                batch_iterator_test =batch_iterator_test,
            )     
            
            return net    
        
        elif net_type == "astronet7_128_256":

            layers7 = [
                (InputLayer, {'shape': (None, shape[0], shape[1], shape[2])}),
            
                (Conv2DLayer, {'num_filters': 128, 'filter_size': (3, 3), 'pad': 0}),
                (MaxPool2DLayer, {'pool_size': (2, 2)}),
                (DropoutLayer, {'p':0.1}),
                                            
                (DenseLayer, {'num_units': 256}),
                (DropoutLayer, {'p':0.5}),
                (DenseLayer, {'num_units': 256}),
            
                (DenseLayer, {'num_units': output_size, 'nonlinearity': softmax}),
            ]

            net = NeuralNet(
                layers=layers7,
                max_epochs=epochs,
                update=lasagne.updates.adam,
                update_learning_rate=learning_rate,
                objective_l2=0.0025,
                batch_iterator_train=batch_iterator_train,
                batch_iterator_test =batch_iterator_test,
                train_split=TrainSplit(eval_size=0.05),
                verbose=verbose,
            )     
            
            return net    
                
        elif net_type == "astronet7_64_128":

            layers7 = [
                (InputLayer, {'shape': (None, shape[0], shape[1], shape[2])}),
            
                (Conv2DLayer, {'num_filters': 64, 'filter_size': (3, 3), 'pad': 0}),
                (MaxPool2DLayer, {'pool_size': (2, 2)}),
                (DropoutLayer, {'p':0.1}),
                                            
                (DenseLayer, {'num_units': 128}),
                (DropoutLayer, {'p':0.5}),
                (DenseLayer, {'num_units': 128}),
            
                (DenseLayer, {'num_units': output_size, 'nonlinearity': softmax}),
            ]

            net = NeuralNet(
                layers=layers7,
                max_epochs=epochs,
                update=lasagne.updates.adam,
                update_learning_rate=learning_rate,
                objective_l2=0.0025,
                batch_iterator_train=batch_iterator_train,
                batch_iterator_test =batch_iterator_test,
                train_split=TrainSplit(eval_size=0.05),
                verbose=verbose,
            )     
            
            return net   
        
        elif net_type == "astronet7_32_64":

            layers7 = [
                (InputLayer, {'shape': (None, shape[0], shape[1], shape[2])}),
            
                (Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3), 'pad': 0}),
                (MaxPool2DLayer, {'pool_size': (2, 2)}),
                (DropoutLayer, {'p':0.1}),
                                            
                (DenseLayer, {'num_units': 64}),
                (DropoutLayer, {'p':0.5}),
                (DenseLayer, {'num_units': 64}),
            
                (DenseLayer, {'num_units': output_size, 'nonlinearity': softmax}),
            ]

            net = NeuralNet(
                layers=layers7,
                max_epochs=epochs,
                update=lasagne.updates.adam,
                update_learning_rate=learning_rate,
                objective_l2=0.0025,
                batch_iterator_train=batch_iterator_train,
                batch_iterator_test =batch_iterator_test,
                train_split=TrainSplit(eval_size=0.05),
                verbose=verbose,
            )     
            
            return net            

        elif net_type == "galaxyzoo_small":

            layers7 = [
                (InputLayer, {'shape': (None, shape[0], shape[1], shape[2])}),
            
                (Conv2DLayer, {'num_filters': 32, 'filter_size': (3, 3), 'pad': 0}),
                (MaxPool2DLayer, {'pool_size': (2, 2)}),
                (DropoutLayer, {'p':0.1}),
                                            
                (DenseLayer, {'num_units': 64}),
                (DropoutLayer, {'p':0.5}),
                (DenseLayer, {'num_units': 64}),
            
                (DenseLayer, {'num_units': output_size, 'nonlinearity': None}),
            ]

            net = NeuralNet(
                layers=layers7,
                max_epochs=epochs,
                regression=regression,
                update=lasagne.updates.adam,
                update_learning_rate=learning_rate,
                objective_l2=0.0025,
                batch_iterator_train=batch_iterator_train,
                batch_iterator_test =batch_iterator_test,
                train_split=TrainSplit(eval_size=0.05),
                verbose=verbose,
            )     
            
            return net                 
                                
        else:
            raise Exception("Unknown network type: %s" % str(net_type))
        
