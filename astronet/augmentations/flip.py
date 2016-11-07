import numpy as np
import copy

class FlipLR():
    """ Horizontally mirrors each input pattern with a given probability.
        
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
        if self.selected_classes is not None:
            mask = np.in1d(Y, list(self.selected_classes))
        else:
            mask = np.ones(len(Y))
        flips = np.random.rand(len(mask))
        flips = flips<self.prob
        mask = np.logical_and(mask,flips)

        Xtransformed = copy.deepcopy(X)
        for (i, flip) in enumerate(mask):          
            if flip:
                Xtransformed[i] = Xtransformed[i,:,:,::-1] 

        return Xtransformed, Y, features

class FlipUD():
    """Vertically mirrors each input pattern with a given probability.
        
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
        if self.selected_classes is not None:
            mask = np.in1d(Y, list(self.selected_classes))
        else:
            mask = np.ones(len(Y))
        flips = np.random.rand(len(mask))
        flips = flips<self.prob
        mask = np.logical_and(mask,flips)

        Xtransformed = copy.deepcopy(X)
        for (i, flip) in enumerate(mask):
            if flip:
                Xtransformed[i] = Xtransformed[i,:,::-1]

        return Xtransformed, Y, features
