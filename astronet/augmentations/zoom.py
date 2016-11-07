import numpy as np
from skimage.transform import rescale

class ZoomIn():
    """ Zoom in each image with a randomly selected factor.
        
    Parameters
    ----------
    selected_classes : list or None, default None
        Labels of classes to which this augmentation should 
        be applied. Should be None for regression.
    factors : int, default [1, 1.0]
        Possible factors with which zooming is done.
    verbose : int, default 0
        Verbosity level.
    """
    def __init__(self, selected_classes=None, factors=[1,1.0], verbose=0):
        self.selected_classes = selected_classes
        self.factor = factors

    def apply(self, X, Y, features):
        if self.selected_classes is not None:
            mask = np.in1d(Y, list(self.selected_classes))
        else:
            mask = np.ones(len(Y))
        zooms = np.random.randint(len(self.factor), size=len(Y))

        imshape = (X.shape[2], X.shape[3])

        for (i,m) in enumerate(mask):
            if m:
                if self.factor[zooms[i]] == 1:
                    pass
                else:
                    for j in range(len(X[i])):
                        X[i,j] = self._crop_zoom(rescale(X[i,j], self.factor[zooms[i]], preserve_range=True), imshape)

        return X, Y, features

    @staticmethod
    def _crop_zoom(X, imshape):
        cur_x = X.shape[0]
        cur_y = X.shape[1]       
        hw = imshape[0]/2
        hh = imshape[1]/2 

        return X[cur_x/2-hw:cur_x/2+hw, cur_y/2-hh:cur_y/2+hh]
