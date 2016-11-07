import numpy as np
from scipy.ndimage.interpolation import shift
import copy

class Shift():
    """ Shift each image with a randomly selected number of pixels, both horizontally and
    vertically. Empty pixels can be filled with a constant value or filled in another way.
    (See 'scipy.ndimage.interpolation.shift' documentation.)
        
    Parameters
    ----------
    selected_classes : list or None, default None
        Labels of classes to which this augmentation should 
        be applied. Should be None for regression.
    x_range : list, default [-2,0,2]
        Possible number of pixels to shift the image in x direction.
    y_range : list, default [-2,0,2]
        Possible number of pixels to shift the image in y direction.
    mode : string, default 'constant'
        Mode of interpolation used for empty pixels. Possible
        values are 'constant', 'nearest', 'reflect', 'wrap'. The
        function used is scipy.ndimage.interpolate.shift().
    cval : float, default 0.0
        Constant value to use for all empty pixels. Only used when
        'mode' is set to 'constant'.
    verbose : int, default 0
        Verbosity level.
    """
    def __init__(self, selected_classes=None, x_range=[-2,0,2], y_range=[-2,0,2], mode='constant', cval=0, verbose=0):
        self.selected_classes = selected_classes
        self.x_range = x_range
        self.y_range = y_range
        self.mode = mode
        self.cval = cval
        self.verbose = verbose

    def apply(self, X, Y, features):
        if self.selected_classes is not None:
            mask = np.in1d(Y, list(self.selected_classes))
        else:
            mask = np.ones(len(Y))
        shift_x = np.random.randint(len(self.x_range), size=len(Y))
        shift_y = np.random.randint(len(self.y_range), size=len(Y))

        Xtransformed = copy.deepcopy(X)
        for (i,b) in enumerate(mask):
            if b:
                for j in xrange(X.shape[1]):
                    Xtransformed[i,j,:,:] = shift(Xtransformed[i,j,:,:], 
                                       [self.x_range[shift_x[i]], self.y_range[shift_y[i]]],
                                       output=X.dtype, mode=self.mode, cval=self.cval)

        return Xtransformed, Y, features
