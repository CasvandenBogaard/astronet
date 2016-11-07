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
        Xtransformed = copy.deepcopy(X)
        
        mean = [np.mean(X[:,i]) for i in range(X.shape[1])]
        std = [np.std(X[:,i]) for i in range(X.shape[1])]   

        for i in range(X.shape[1]):
            Xtransformed[:,i] = X[:,i] - mean[i]
            Xtransformed[:,i] = Xtransformed[:,i]/std[i]

        return Xtransformed, Y, features
