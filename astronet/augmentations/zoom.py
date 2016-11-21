import numpy as np
from skimage.transform import rescale
import copy

class ZoomIn():
    """ Zoom in each image with a randomly selected factor.
        
    Parameters
    ----------
    selected_classes : list or None, default None
        Labels of classes to which this augmentation should 
        be applied. Should be None for regression.
    factors : int, default [1, 1.0]
        Possible factors with which zooming is done.
    verbose : int, default 0
        Verbosity level.
    """
    def __init__(self, selected_classes=None, factors=[1,1.1], verbose=0):
        self.selected_classes = selected_classes
        self.factor = factors

    def apply(self, X, Y, features):
        """Apply the augmentation corresponding to this class to the input data.
            
        Parameters
        ----------
        X : array-like
            Array of input patterns.
        Y : array-like
            Array of input labels.
        features : array-like
            Array of extra input features.

        Returns
        -------
        Xtransformed : array-like (same as X)
            Array of transformed input patterns.
        Y : array-like (same as Y)
            Array of transformed input labels.
        features : array--like (same as features)
            Array of transformed extra input features.
        """
        Xtransformed = copy.deepcopy(X)
        if self.selected_classes is not None:
            mask = np.in1d(Y, list(self.selected_classes))
        else:
            mask = np.ones(len(Y))
        zooms = np.random.randint(len(self.factor), size=len(Y))

        imshape = (Xtransformed.shape[2], Xtransformed.shape[3])

        for (i,m) in enumerate(mask):
            if m:
                if self.factor[zooms[i]] == 1:
                    pass
                else:
                    for j in range(len(X[i])):
                        Xtransformed[i,j] = self._crop_zoom(rescale(Xtransformed[i,j], self.factor[zooms[i]], preserve_range=True), imshape)

        return Xtransformed, Y, features

    @staticmethod
    def _crop_zoom(X, imshape):
        cur_x = X.shape[0]
        cur_y = X.shape[1]       
        hw = imshape[0]/2
        hh = imshape[1]/2 

        return X[cur_x/2-hw:cur_x/2+hw, cur_y/2-hh:cur_y/2+hh]
