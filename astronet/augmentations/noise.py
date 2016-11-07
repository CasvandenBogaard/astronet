import numpy as np

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
        if isinstance(self.values, int):
            if self.dist == 'gaussian':
                X += np.random.normal(scale=self.values, size=X.shape)
            elif self.dist == 'uniform':
                X += np.random.uniform(low=0, high=self.values, size=X.shape)

        else:
            scales = np.random.rand(len(Y)) * (self.values[1]-self.values[0]) + self.values[0]
            if self.dist == 'gaussian':
                for i in range(len(X)):
                    X[i] += np.random.normal(scale=self.values[i], size=X[i].shape)
            elif self.dist == 'uniform':
                for i in range(len(X)):
                    X[i] += np.random.uniform(low=0, high=self.values[i], size=X[i].shape)
                
        return X, Y, features

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
        if isinstance(self.values, int):
            X += self.values
        else:
            X += np.random.rand(*X.shape) * (self.values[1]-self.values[0]) + self.values[0]

        return X, Y, features
