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
        if self.dims is not None:
            X = X[:, self.dims]

        return X, Y, features    
