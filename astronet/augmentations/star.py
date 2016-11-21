import numpy as np
import copy
from scipy.ndimage.filters import gaussian_filter

class AddStar():
    """ With a given probability, add a foreground star to the input image. 
        
    Parameters
    ----------
    selected_classes : list or None, default None
        Labels of classes to which this augmentation should 
        be applied. Should be None for regression.
    prob : float, default 0.5
        Probability that a foreground star is added.
        Value between 0 and 1.
    L_range : list, default [0,100]
        Range of values from which the brightness of the 
        added star is selected.
    var_range : list, default [1,3]
        Range of values from which the width of the added
        star is selected.
    verbose : int, default 0
        Verbosity level.
    """
    def __init__(self, selected_classes=None, prob=0.5, L_range=[0,100], var_range=[1,3], verbose=0):
        self.selected_classes = selected_classes
        self.prob = prob
        self.mean_range = L_range
        self.var_range = var_range
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
                Xtransformed[i,:] += self._add_star(Xtransformed[i])

        return Xtransformed, Y, features

    def _add_star(self, im):
        shape = im.shape[1:]
        x_c = np.random.randint(shape[0])
        y_c = np.random.randint(shape[1])

        mean, var = np.random.rand(2)
        mean = self.mean_range[0] + mean*(self.mean_range[1]-self.mean_range[0])
        var = self.var_range[0] + var*(self.var_range[1]-self.var_range[0])

        delta_im = np.zeros((2*shape[0]+1, 2*shape[1]+1))
        delta_im[shape[0], shape[1]] = 1
        to_add = gaussian_filter(delta_im, var)[shape[0]-x_c:2*shape[0]-x_c, shape[1]-y_c:2*shape[1]-y_c]
        to_add = to_add*mean

        return to_add.astype(np.float32)
