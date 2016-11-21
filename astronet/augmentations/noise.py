import numpy as np
import copy

class AddNoise():
    """ Add randomly generated noise from a given distribution to each input.
        
    Parameters
    ----------
    scale : list or int, default 10
        Value for the highest possible noise. If a list, this highest value will be
        randomly selected from the provided range.
    distribution : string, default 'gaussian'
        Type of distribution the noise should have. Supported values: 'gaussian', 'uniform'.
    verbose : int, default 0
        Verbosity level.
    """
    def __init__(self, scale=10, distribution='gaussian', verbose=0):
        self.values = scale
        self.dist = distribution
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
        if isinstance(self.values, int):
            if self.dist == 'gaussian':
                Xtransformed += np.random.normal(scale=self.values, size=Xtransformed.shape)
            elif self.dist == 'uniform':
                Xtransformed += np.random.uniform(low=0, high=self.values, size=Xtransformed.shape)

        else:
            scales = np.random.rand(len(Y)) * (self.values[1]-self.values[0]) + self.values[0]
            if self.dist == 'gaussian':
                for i in range(len(Xtransformed)):
                    Xtransformed[i] += np.random.normal(scale=scales[i], size=Xtransformed[i].shape)
            elif self.dist == 'uniform':
                for i in range(len(Xtransformed)):
                    Xtransformed[i] += np.random.uniform(low=0, high=scales[i], size=Xtransformed[i].shape)
                
        return Xtransformed, Y, features

class AddConstant():
    """ Either adds a constant value to every image, or a randomly selected value for each
    image. 
        
    Parameters
    ----------
    range : int or list, default 10
        Adds the same constant to all inputs if input is an integer, otherwise adds a randomly 
        selected constant from the given range.
    verbose : int, default 0
        Verbosity level.
    """
    def __init__(self, values=[-50,50], verbose=0):
        self.values = values
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
        if isinstance(self.values, int):
            Xtransformed += self.values
        else:
            Xtransformed += np.random.rand(*X.shape) * (self.values[1]-self.values[0]) + self.values[0]

        return Xtransformed, Y, features
