class SelectDimensions():
    """ Select certain dimensions from the input (e.g. only R channel from an RGB image, or
    only u and i filters from a stack of images in ugriz-bands.)
        
    Parameters
    ----------
    dimensions : list or None, default None
        Indices of the dimensions that should be kept.
    verbose : int, default 0
        Verbosity level.
    """
    def __init__(self, dimensions=None, verbose=0):
        self.dims = dimensions
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
        if self.dims is not None:
            X = X[:, self.dims]

        return X, Y, features    
