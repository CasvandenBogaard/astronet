import numpy as np
import copy

class EdgeError():
    """Sets a random edge to values smaller than 1 with
    a given probability. 
        
    Parameters
    ----------
    selected_classes : list or None, default None
        Labels of classes to which this augmentation should 
        be applied. Should be None for regression.
    prob : float, default 0.5
        A value between 0 and 1, which sets the probability
        with which an image is selected to be augmented.
    verbose : int, default 0
        Verbosity level.
    """
    def __init__(self, selected_classes=None, prob=0.5, verbose=0):
        self.selected_classes = selected_classes
        self.prob = prob
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
        add = np.random.random(len(Y)) < self.prob
        
        Xtransformed = copy.deepcopy(X)
        for i in range(len(Xtransformed)):
            if add[i]:
                Xtransformed[i] = self._add_edge_error(Xtransformed[i])

        return Xtransformed, Y, features

    @staticmethod
    def _add_edge_error(im):
        side = np.random.randint(4)
        dim = np.random.randint(im.shape[0])
        width = np.random.randint(int(im.shape[1]/2))
        height = np.random.randint(int(im.shape[2]/2))
        value = np.random.random()

        if side == 0:
            im[dim, :width, :] = value
        if side == 1:
            im[dim, :, :height] = value
        if side == 2:
            im[dim, im.shape[1]-width:, :] = value
        if side == 3:
            im[dim, :, im.shape[2]-height:] = value

        return im

class DeadColumn():
    """With a given probability, set all values in a randomly selected
    column to zero.
        
    Parameters
    ----------
    selected_classes : list or None, default None
        Labels of classes to which this augmentation should 
        be applied. Should be None for regression.
    prob : float, default 0.5
        A value between 0 and 1, which sets the probability
        with which an image is selected to be augmented.
    verbose : int, default 0
        Verbosity level.
    """
    def __init__(self, selected_classes=None, prob=0.5, verbose=0):
        self.selected_classes = selected_classes
        self.prob = prob
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
        add = np.random.random((X.shape[1],len(Y))) < self.prob

        Xtransformed = copy.deepcopy(X)
        for i in range(len(Xtransformed)):
            for col in range(X.shape[1]):
                if add[col,i]:
                    col_i = np.random.randint(X.shape[2])
                    Xtransformed[i,col,:,col_i] = 0

        return Xtransformed, Y, features
