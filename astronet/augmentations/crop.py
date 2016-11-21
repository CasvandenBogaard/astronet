import numpy as np
import copy

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
    def __init__(self, x_range=(0,None), y_range=(0,None), verbose=0):
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
        Xtransformed = copy.deepcopy(X)
        Xtransformed = Xtransformed[:, :, self.x_range[0]:self.x_range[1], self.y_range[0]:self.y_range[1]]

        return Xtransformed, Y, features

class RandomCrop():
    """Crop an input to the selected range of values.
        
    Parameters
    ----------
    x_len : integer, default 0
        Width of the cropped input, no cropping when set to 0.
    y_len : integer, default 0
        Height of the cropped input, no cropping when set to 0.
    verbose : int, default 0
        Verbosity level.
    """
    def __init__(self, x_len=0, y_len=0, verbose=0):
        self.x_len = x_len
        self.y_len = y_len
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
        
        if self.x_len == 0:
            x_start = np.zeros(len(Y))
            x_end = np.ones(len(Y))*X.shape[2]
        else:
            x_start = np.random.randint(X.shape[2]-self.x_len, size=len(Y))
            x_end = x_start + self.x_len
            
        if self.y_len == 0:
            x_start = np.zeros(len(Y))
            y_end = np.ones(len(Y))*X.shape[3]
        else:
            y_start = np.random.randint(X.shape[3]-self.y_len, size=len(Y))
            y_end = y_start + self.y_len
            
        Xtransformed = np.zeros((X.shape[0], X.shape[1], self.x_len, self.y_len), dtype=X.dtype)
        for i in range(len(Y)):
            Xtransformed[i] = X[i, :, x_start[i]:x_end[i], y_start[i]:y_end[i]]
        
        return Xtransformed, Y, features
