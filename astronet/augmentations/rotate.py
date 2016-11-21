import numpy as np
from scipy.ndimage.interpolation import rotate
import copy

class Rotate():
    """ Rotate each input with a randomly selected angle.
        
    Parameters
    ----------
    selected_classes : list or None, default None
        Labels of classes to which this augmentation should 
        be applied. Should be None for regression.
    rotations : list, default [0,90,180,270]
        Values of the angles that the algorithm can choose from.
    verbose : int, default 0
        Verbosity level.
    """
    def __init__(self, selected_classes=None, rotations=[0,90,180,270], verbose=0):
        self.selected_classes = selected_classes
        self.rotations = rotations
        self.verbose = verbose

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
        if self.selected_classes is not None:
            mask = np.in1d(Y, list(self.selected_classes))
        else:
            mask = np.ones(len(Y))
        rotat = np.random.randint(len(self.rotations), size=len(Y))
        
        Xtransformed = copy.deepcopy(X)
        for (i,rot) in enumerate(mask):
            if (rot and rotat[i]):
                Xtransformed[i] = rotate(Xtransformed[i], self.rotations[rotat[i]], axes=(1, 2), reshape=False)

        return Xtransformed, Y, features
