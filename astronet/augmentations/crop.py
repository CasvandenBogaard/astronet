import numpy as np

class Crop():
    """Crop an input to the selected range of values.
        
    Parameters
    ----------
    x_range : tuple or list, default (0,-1)
        Contains beginning and endpoints of the slice in the x direction.
    y_range : tuple or list, default (0,-1)
        Contains beginning and endpoints of the slice in the y direction.
    verbose : int, default 0
        Verbosity level.
    """
    def __init__(self, x_range=(0,-1), y_range=(0,-1), verbose=0):
        self.x_range = x_range
        self.y_range = y_range
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
        X : array-like (same as X)
            Array of transformed input patterns.
        Y : array-like (same as Y)
            Array of transformed input labels.
        features : array--like (same as features)
            Array of transformed extra input features.
        """
        X = X[:, :, self.x_range[0]:self.x_range[1], self.y_range[0]:self.y_range[1]]

        return X, Y, features
