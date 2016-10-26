'''
Created on 29.05.2016

@author: Fabian Gieseke
'''

import os
import copy
import numpy as np

import lasagne
from nolearn.lasagne.visualize import draw_to_file
from nolearn.lasagne.visualize import plot_loss
from nolearn.lasagne.visualize import plot_conv_weights
from lasagne.layers import Conv2DLayer
        
from astronet.util import ensure_dir, start_via_single_process
from .input import AugmentationBatchIterator


class AstroWrapper(object):
    """ Deep net for supernova classification.
    
    Parameters
    ----------
    net_type : str, default 'astronet1'
        The predefined network type 
    epochs : int
        The number of epochs of the network
    learning_rate : float, default 0.0002
        The learning (update) rate
    transforms : list, default []
        Transformations that shall be applied to 
        the images/patterns before training or
        applying the model.
    augments : list, default []
        Augmentation steps that shall be applied
        to the data before training the model. The
        ordering does matter. Each element is a list
        with two elements, where the first element 
        specifies the method and the second one the
        parameters for the augmentation method. Example:
        
        augments = [
                    ('rotate_positives', {'ratio': 0.3, 'rotations':[90, 180, 270]}),
                    ('rotate_all', {'factor': 0.6, 'rotations':[90, 180, 270]}),
                   ]        
    seed : int, default 0
        The seed to be used
    verbose : int, default 1
        Verbosity level                   
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

    def get_params(self, deep=True):
        
        return {"net_type": self.net_type,
                "epochs": self.epochs,
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "augments": self.augments,
                "transforms": self.transforms,
                "seed": self.seed,
                "verbose": self.verbose,
                }
    
    def set_params(self, **parameters):
        
        for parameter, value in parameters.items():
            self.setattr(parameter, value)
            
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
        self.ABI.add_augment(method, args)

    def save_weights(self, odir, ofname):
        self.net.save_params_to(os.path.join(odir,ofname))

    def load_weights(self, fname):
        if not hasattr(self, 'model'):
            print "Initialization of model incomplete!"
        else:
            self.net.load_params_from(fname)
    
    #def save_model(self, ofname):
        
        #ensure_dir(ofname)
        #np.savez(ofname, *lasagne.layers.get_all_param_values(self.net.layers_))
    
    def save_details(self, odir, figsize=(10,10)):
        #start_via_single_process(AstroNet._save_details, [odir, self.net], {'figsize':figsize})
        self._save_details(odir, self.net, figsize=figsize)

    @staticmethod
    def _save_details(odir, net, figsize=(10,10)):
        ensure_dir(os.path.join(odir, "layers.png")) 
        draw_to_file(net.layers, os.path.join(odir, "layers.png"), output_shape=False)
        
        plt = plot_loss(net)
        plt.savefig(os.path.join(odir, "loss.png"))
        plt.close()
