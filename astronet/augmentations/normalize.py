import numpy as np
import copy

class Normalize():
    """ Calculates the batch mean and standard deviation. Then normalizes inputs by first 
    subtracting the mean, then dividing by the standard deviation. This prevents rounding
    errors when dealing with large inputs and allows for easier learning of the network.
        
    Parameters
    ----------
    verbose : int, default 0
        Verbosity level.
    """
    def __init__(self, verbose=0):
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
        Xtransformed = copy.deepcopy(X)
        
        mean = [np.mean(X[:,i]) for i in range(X.shape[1])]
        std = [np.std(X[:,i]) for i in range(X.shape[1])]   

        for i in range(X.shape[1]):
            Xtransformed[:,i] = X[:,i] - mean[i]
            Xtransformed[:,i] = Xtransformed[:,i]/std[i]

        return Xtransformed, Y, features
