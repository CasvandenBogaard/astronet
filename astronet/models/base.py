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
from nolearn.lasagne import BatchIterator
from lasagne.layers import Conv2DLayer
        
from astronet.util import ensure_dir, start_via_single_process
from .input import InputHandler, AugmentationBatchIterator
from .net import NetGenerator


class AstroNet(object):
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
    
    def __init__(self,
                 net_type="astronet1",
                 epochs=500,
                 learning_rate=0.0002,
                 batch_size=128,
                 balanced=False,
                 augments=[],
                 transforms=[],
                 seed=0,
                 verbose=1,
                 ):
        
        self.net_type = net_type
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.balanced = balanced
        self.augments = augments
        self.transforms = transforms
        self.seed = seed
        self.verbose = verbose

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

        self._input_handler = InputHandler(seed=self.seed, verbose=self.verbose)
        self._network_generator = NetGenerator()
        
        XI_train = copy.deepcopy(XI_train)
        y_train = copy.deepcopy(y_train)
        XF_train = copy.deepcopy(XF_train)
        
        if self.verbose > 0:
            print("\nNumber of samples used to train network: %i" % len(XI_train))
            counts = np.bincount(y_train.astype(np.int64))
            for i in xrange(len(counts)):
                if counts[i] > 0:
                    print("-> Number of examples for class %i: %i" % (i, counts[i]))
            print("")

                    

        # transform images and patterns
        XI_train, XF_train = self._transform_patterns(XI_train, XF=XF_train, fit=True)

        self.ABI = AugmentationBatchIterator(XF_train, self.batch_size, balanced=self.balanced, augments=self.augments)
        self.ABI_test = AugmentationBatchIterator(None, self.batch_size, augments=self.transforms)
        
        X = XI_train.astype(np.float32)
        y = y_train.astype(np.int32)
                
        self.model = self._network_generator.get_instance(X, y, self.net_type, 
                                                          epochs=self.epochs, 
                                                          learning_rate=self.learning_rate, 
                                                          verbose=self.verbose,
                                                          batch_iterator_train=self.ABI,
                                                          #batch_iterator_test=self.ABI_test,
                                                          )
        

        # fit final model
        self.model.fit(X, y)
        
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
        
        X, XF = copy.deepcopy(X), copy.deepcopy(XF)
        
        # transform images/patterns
        #if self.ABI_test == None:
        X, XF = self._transform_patterns(X, XF=XF, fit=False)
        
        # compute predictions
        preds = self.model.predict(X.astype(np.float32))
          
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
        
        X, XF = copy.deepcopy(X), copy.deepcopy(XF)
        
        # transform images/patterns
        X, XF = self._transform_patterns(X, XF=XF, fit=False)
        
        # compute predictions
        preds_probs = self.model.predict_proba(X.astype(np.float32))
          
        return preds_probs

    def _augment_data(self, X, XF, y):
        
        for i in xrange(len(self.augments)):
            
            method = self.augments[i][0]
            params = self.augments[i][1]
            
            X, XF, y = self._input_handler.apply(X, XF, y, method=method, params=params)
        
        return X, XF, y
    
    def _transform_patterns(self, X, XF=None, fit=False):
        """
        """
        
        for i in xrange(len(self.transforms)):
            
            method = self.transforms[i][0]
            params = copy.deepcopy(self.transforms[i][1])
            
            assert "fit" not in params.keys()
            params["fit"] = fit
        
            X, XF, _ = self._input_handler.apply(X, XF, None, method=method, params=params)
            
        return X, XF

    def save_weights(self, odir, ofname):
        self.model.save_params_to(os.path.join(odir,ofname))

    def load_weights(self, fname):
        if not hasattr(self, 'model'):
            print "Initialization of model incomplete!"
        else:
            self.model.load_params_from(fname)
    
    #def save_model(self, ofname):
        
        #ensure_dir(ofname)
        #np.savez(ofname, *lasagne.layers.get_all_param_values(self.model.layers_))
    
    def save_details(self, odir, figsize=(10,10)):
        
        #start_via_single_process(AstroNet._save_details, [odir, self.model], {'figsize':figsize})
        self._save_details(odir, self.model, figsize=figsize)
    
    @staticmethod
    def _save_details(odir, model, figsize=(10,10)):
        ensure_dir(os.path.join(odir, "layers.png")) 
        draw_to_file(model.layers, os.path.join(odir, "layers.png"), output_shape=False)
        
        plt = plot_loss(model)
        plt.savefig(os.path.join(odir, "loss.png"))
        plt.close()
        plt.close()
                
        for i in xrange(len(model.layers_)):
            name = model.layers_[i].name
            layer = model.layers_[name]
            if type(layer) == lasagne.layers.Conv2DLayer:
                plt = plot_conv_weights(layer, figsize=figsize)
                plt.savefig(os.path.join(odir, "layer_%s.png" % str(name)))
                plt.close()
                plt.close()
