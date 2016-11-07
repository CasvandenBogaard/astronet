import sys, os
import copy
import numpy as np
from scipy.ndimage.interpolation import shift, rotate
from scipy.ndimage.filters import gaussian_filter
from skimage.transform import rescale
import skimage.io as io
from nolearn.lasagne import BatchIterator


import matplotlib.pyplot as plt

class AugmentationBatchIterator(BatchIterator):
    """Class that handles data augmentation and batch processing.
        
    Parameters
    ----------

    batch_size : integer
        Number of input patterns per batch.
    augments : list, default []
        Augmentations that will be applied to the data in each batch. Each item in this list
        should be a class that extends the 'Identity' augmentation. The 'apply' method is called
        on each of the augmentations in this list in order.
    balanced: boolean, default False
        If set to True, will force each batch to contain 
        an equal amount of samples from each class. Only
        works for binary classification problems.
    regression: boolean, default False
        Should be set to True when the labels are real
        values (regression) instead of distinct classes
        (classification).
    seed: integer, default 0
        Integer to seed the random number generator.
    verbose: integer, default 0
        Specifies verbosity of the model.
    """

    def __init__(self, batch_size, augments=[], balanced=False, regression=False, seed=0, verbose=1):
        super(AugmentationBatchIterator, self).__init__(batch_size, shuffle=True, seed=seed)

        self.features = []

        self.augments = augments
        self.balanced = balanced
        self.regression = regression

        self._methods_cache = {}        
        self.verbose = verbose
        self.seed = seed
        
        np.random.seed(self.seed)

    def add_augment(self, method):
        """Method to add augmentation steps after initialization
        of the AugmentationBatchIterator.
            
        Parameters
        ----------
        method : object extending 'Identity', or list of those.
            Augmentation(s) of the same form as those in the augments list during initiation.
        """
        self.augments.append( method )

    # TO DO:
    # Look at balancing for multiple classes
    def transform(self, Xb, yb):
        """Load the data if input of filenames, then applies
        balancing and augmentations to the data if applicable.
            
        Parameters
        ----------
        Xb : array-like
            Can be either an array of of input patterns, or
            a list of filenames for the input patterns. In 
            case this is a list of filenames, files will be
            opened before applying augmentation steps.
        yb : array-like
            Labels corresponding to input patterns. Should be
            set to None when input patterns are unknown.

        Returns
        -------
        X : array-like, same as Xb or numpy.ndarray
            Input patterns after all augmentations are applied.
        Y : array-like, same as yb
            Labels after all augmentations are applied.
        """
        X = copy.deepcopy(Xb)
        y = copy.deepcopy(yb)
        Xf = copy.deepcopy(self.features)

        if isinstance(X[0], str):
            # Input is list of filenames
            images = np.array( [io.imread(os.path.join(X[0]))] )
            for i in range(1,len(X)):
                im = np.array( [io.imread(os.path.join(X[i]))] )
                images = np.vstack((images, im))

            images = images.transpose(0,3,1,2)
            X = images


        if self.balanced:
            pos_ind = np.argwhere(y>0).ravel()
            neg_ind = np.argwhere(y==0).ravel()

            if len(pos_ind) == 0:
                pos_ind = [0]

            if len(neg_ind) > len(pos_ind):
                neg_ind = np.random.choice(neg_ind.ravel(), len(pos_ind), replace=False)
                indices = list(pos_ind) + list(neg_ind)
           
                X = np.array([X[i] for i in indices])
                y = np.array([y[i] for i in indices])
                if Xf is not None:
                    Xf = np.array([Xf[i] for i in indices])

        for aug in self.augments:
            X, y, Xf = aug.apply(X, y, Xf)
        
        X = X.astype(np.float32)
        if y is not None:
            if self.regression:
                y = y.astype(np.float32)
            else:
                y = y.astype(np.int32)


        return X, y
       
