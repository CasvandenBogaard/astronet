'''
Created on 29.05.2016

@author: Fabian Gieseke, Cas van den Bogaard
'''

import os
import copy
import numpy as np

from astronet.augmentations import AugmentationBatchIterator


class AstroWrapper(object):
    """ Wrapper that extends the functionality of the
    'nolearn.lasagne.NeuralNet' class. Added functionality
    includes on-the-fly data augmentation, loading data in
    batches and balancing classes.

    'astronet' is designed for use with astrophysical data.
    It can be used on before-after-diff images for transient
    detection, timeseries such as lightcurve data or any other
    type of data in image or timeseries form. The augmentations
    implemented in 'astronet' include both standard augmentations,
    like normalization, flipping and rotation, and augmentations
    designed for astrophysical data, like adding foreground stars,
    or modelling CCD errors.
    
    Parameters
    ----------

    net : nolearn.lasagne.NeuralNet
        The network that will be used in training and testing.
    regression : boolean, default False
        Specifies if training labels are classes or continuous.
    batch_size : int, default 128
        Number of input samples per batch.
    balanced : boolean, default False
        If true, ensures that each batch contains an equal amount 
        of samples from all classes. Only implemented for binary
        classification.
    augments : list, default []
        Augmentation steps that shall be applied
        to the data before training the model. The
        ordering does matter. Each element is a tuple
        with two elements, where the first element is
        a string specifying the method and the second 
        is a dictionary with the parameters for the 
        augmentation method. 
    transforms : list, default []
        Transformations that will be applied to 
        the images/patterns before training or
        applying the model. This list has the same
        form as the list with augmentations.
    seed : int, default 0
        The seed to be used
    verbose : int, default 1
        Verbosity level      

    Examples
    --------

    With the following list of augments, all training data will be
    flipped with the default probability (0.5), and then a random
    value between -10 and 10 is added to each input.

    >>> from astronet import AstroNet
    >>> 
    >>> augments = [
    >>>             ('flipud', {}),
    >>>             ('add_const', {'range':[-10,10]})
    >>>            ]  
    >>>
    >>> net1 = AstroNet('ShallowNet')
    >>> model1 = AstroWrapper(net, augments=augments)           

    After the AstroWrapper has been created, one can still add 
    augmentations using the 'add_augment' method. The following
    results in an equivalent model.

    >>> net2 = AstroNet('ShallowNet')
    >>> model2 = AstroWrapper(net)
    >>> model2.add_method('flipud', {})
    >>> model2.add_method('add_const', {'range':[-10,10]})
    """
    
    def __init__(self, net,
                 regression=False,
                 batch_size=128,
                 balanced=False,
                 augments=[],
                 transforms=[],
                 seed=0,
                 verbose=1,
                 ):
        
        self.net = net
        self.regression = regression
        self.batch_size = batch_size
        self.balanced = balanced
        self.augments = augments
        self.transforms = transforms
        self.seed = seed
        self.verbose = verbose

        self.ABI = AugmentationBatchIterator(self.batch_size, balanced=self.balanced, augments=self.augments+self.transforms)
        self.ABI_test = AugmentationBatchIterator(self.batch_size, augments=self.transforms)

        self.net.batch_iterator_train = self.ABI
        self.net.batch_iterator_test = self.ABI_test

    def get_params(self):
        """ Returns the current model parameters.
            
        Returns
        -------
        dictionary
            Parameters of the model.
        """
        return {"net": self.net,
                "regression": self.regression,
                "batch_size": self.batch_size,
                "augments": self.augments,
                "transforms": self.transforms,
                "seed": self.seed,
                "verbose": self.verbose,
                }
    
    def set_params(self, **parameters):
        """ Sets model parameters and returns self. Also recreates AugmentationBatchIterators with new model parameters.
            
        Parameters
        -------
        **parameters : dictionary
            Parameters of the model.

        Returns
        -------
        self : AstroWrapper
            AstroWrapper with the changed values and new
            AugmentationBatchIterators for these values.
        """        
        for parameter, value in parameters.items():
            self.setattr(parameter, value)

        self.ABI = AugmentationBatchIterator(self.batch_size, balanced=self.balanced, augments=self.augments+self.transforms)
        self.ABI_test = AugmentationBatchIterator(self.batch_size, augments=self.transforms)

        self.net.batch_iterator_train = self.ABI
        self.net.batch_iterator_test = self.ABI_test
            
        return self
        
    def fit(self, XI_train, y_train, XF_train=None):
        """ Fits the deep network.
        
        Parameters
        ----------
        XI_train : array-like
            Array of training images
        y_train : array-like
            Training labels
        XF_train : array-like or None, default None
            Optional: Array of training patterns (features)            
        """
        
        X = copy.deepcopy(XI_train)
        y = copy.deepcopy(y_train)
        XF_train = copy.deepcopy(XF_train)
        
        if self.verbose > 0:
            print("\nNumber of samples used to train network: %i" % len(XI_train))
            print("")

        self.ABI.features = XF_train

        self.net.fit(X, y)
        
        return self

    def predict(self, X, XF=None):
        """ Computes predictions for new instances
        
        Parameters
        ----------
        X : array-like
            Array of input images
        XF : array-like or None
            Optional: Array of patterns (features)
            
        Returns
        -------
        preds : array-like
            The predictions computed by the model
        """
        
        X = copy.deepcopy(X)
        
        # compute predictions
        preds = self.net.predict(X)
          
        return preds
    
    def predict_proba(self, X, XF=None):
        """ Computes predictions (probs) for new instances
        
        Parameters
        ----------
        X : array-like
            Array of input images
        XF : array-like or None
            Optional: Array of patterns (features)
            
        Returns
        -------
        preds : array-like
            The predictions computed by the model
        """
        
        X = copy.deepcopy(X)
        
        # compute predictions
        preds_probs = self.net.predict_proba(X)
          
        return preds_probs

    def add_augment(self, method, args):
        """ Method to add augmentation steps after initialization\of the AugmentationBatchIterator. Calls the method in the AugmentationBatchIterator.
            
        Parameters
        ----------
        method : string
            Name of the augmentation that should be added.
            Should be key in self._methods
        augments : dictionary, default {}
            Parameters for augmentation that is to be added. 
            Should be given in the form {'k1':v1, 'k2':v2}.
        """
        self.ABI.add_augment(method, args)

    def save_weights(self, odir, ofname):
        """ Saves the weights of the current model to a pickle file, to be loaded at a later stage.
            
        Parameters
        ----------
        odir : string
            Relative path to the output directory.
        ofname : string
            Filename of the output pickle file.
        """
        self.net.save_params_to(os.path.join(odir,ofname))

    def load_weights(self, fname):
        """ Loads previously saved weights. Weights should be loaded from a network with the same architecture as the current network only.
            
        Parameters
        ----------
        fname : string
            Relative path to the pickle file containing the
            weights.
        """
        if not hasattr(self, 'model'):
            print "Initialization of model incomplete!"
        else:
            self.net.load_params_from(fname)
