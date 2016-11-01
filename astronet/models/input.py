'''
Created on 01.06.2016

@authors: Jonas Kindler, Fabian Gieseke, Cas van den Bogaard
'''

import sys, os
import copy
import numpy as np
from scipy.ndimage.interpolation import shift, rotate
from scipy.ndimage.filters import gaussian_filter
from skimage.transform import rescale
import skimage.io as io
from nolearn.lasagne import BatchIterator


import matplotlib.pyplot as plt

# TO DO:
# Some more rigorous testing of augmentations
class AugmentationBatchIterator(BatchIterator):
    """Class that handles data augmentation and batch processing.
        
    Parameters
    ----------

    batch_size : integer
        Number of input patterns per batch.
    augments : list, default []
        Augmentation steps that shall be applied
        to the data in each batch. The ordering does matter. 
        Each element is a tuple with two elements, where the 
        first element is a string specifying the method and 
        the second is a dictionary with the parameters for the 
        augmentation method. 
    balanced: boolean, default False
        If set to True, will force each batch to contain 
        an equal amount of samples from each class. Only
        works for binary classification problems.
    regression: boolean, default False
        Should be set to True when the labels are real
        values (regression) instead of distinct classes
        (classification).
    seed: integer, default 0
        Integer to seed the random number generator.
    verbose: integer, default 0
        Specifies verbosity of the model.
    """

    def __init__(self, batch_size, augments=[], balanced=False, regression=False, seed=0, verbose=1):
        super(AugmentationBatchIterator, self).__init__(batch_size, shuffle=True, seed=seed)

        self._methods = {
            "crop_images":AugmentationBatchIterator.crop_images,
            "select_dimensions":AugmentationBatchIterator.select_dimensions,
            "shift_images":AugmentationBatchIterator.shift_images,
            "rotate_images":AugmentationBatchIterator.rotate_images,
            "fliplr":AugmentationBatchIterator.fliplr,
            "flipud":AugmentationBatchIterator.flipud,
            "zoom_in":AugmentationBatchIterator.zoom_in,
            "add_const":AugmentationBatchIterator.add_const,
            "add_noise":AugmentationBatchIterator.add_noise,
            "edge_error":AugmentationBatchIterator.edge_error,
            "add_star":AugmentationBatchIterator.add_star,
            "add_dead_column":AugmentationBatchIterator.add_dead_column,
            "normalize":AugmentationBatchIterator.normalize,
        }

        self.features = []

        self.augments = augments
        self.balanced = balanced
        self.regression = regression

        self._methods_cache = {}        
        self.verbose = verbose
        self.seed = seed
        
        np.random.seed(self.seed)

    def add_augment(self, method, args):
        """Method to add augmentation steps after initialization
        of the AugmentationBatchIterator.
            
        Parameters
        ----------
        method : string
            Name of the augmentation that should be added.
            Should be key in self._methods
        augments : dictionary, default {}
            Parameters for augmentation that is to be added. 
            Should be given in the form {'k1':v1, 'k2':v2}.
        """
        self.augments.append( (method, args) )

    # TO DO:
    # Look at balancing for multiple classes
    def transform(self, Xb, yb):
        """Load the data if input of filenames, then applies
        balancing and augmentations to the data if applicable.
            
        Parameters
        ----------
        Xb : array-like
            Can be either an array of of input patterns, or
            a list of filenames for the input patterns. In 
            case this is a list of filenames, files will be
            opened before applying augmentation steps.
        yb : array-like
            Labels corresponding to input patterns. Should be
            set to None when input patterns are unknown.

        Returns
        -------
        X : array-like, same as Xb or numpy.ndarray
            Input patterns after all augmentations are applied.
        Y : array-like, same as yb
            Labels after all augmentations are applied.
        """
        X = copy.deepcopy(Xb)
        y = copy.deepcopy(yb)
        Xf = copy.deepcopy(self.features)

        if isinstance(X[0], str):
            # Input is list of filenames
            images = np.array( [io.imread(os.path.join(X[0]))] )
            for i in range(1,len(X)):
                im = np.array( [io.imread(os.path.join(X[i]))] )
                images = np.vstack((images, im))

            images = images.transpose(0,3,1,2)
            X = images


        if self.balanced:
            pos_ind = np.argwhere(y>0).ravel()
            neg_ind = np.argwhere(y==0).ravel()

            if len(pos_ind) == 0:
                pos_ind = [0]

            if len(neg_ind) > len(pos_ind):
                neg_ind = np.random.choice(neg_ind.ravel(), len(pos_ind), replace=False)
                indices = list(pos_ind) + list(neg_ind)
           
                X = np.array([X[i] for i in indices])
                y = np.array([y[i] for i in indices])
                if Xf is not None:
                    Xf = np.array([Xf[i] for i in indices])

        for aug in self.augments:
            X, y, Xf = self._methods[aug[0]](self, X, y, Xf, args=aug[1], verbose=self.verbose)
        
        X = X.astype(np.float32)
        if y is not None:
            if self.regression:
                y = y.astype(np.float32)
            else:
                y = y.astype(np.int32)


        return X, y
       
    def crop_images(self, X, Y, features, args={}, verbose=0):
        """Crops an input to the selected size
            
        Parameters
        ----------
        X : array-like
            Array of input patterns
        Y : array-like
            Array of input labels
        features : array-like
            Extra input features
        args : dictionary, default {}
            Contains the key:value pairs for the
            parameters of this augmentation:
                'x_range': list or tuple, default (0, X.shape[2])
                    Contains beginning and endpoints of the slice in
                    the x direction.
                'y_range': list or tuple, default (0, X.shape[3])
                    Contains beginning and endpoints of the slice in
                    the y direction.
        verbose : integer, default 0
            Controls verbosity of this method

        Returns
        -------
        X : array-like, same as X
            Input patterns after application of this augmentation step.
        Y : array-like, same as Y
            Labels after application of this augmentation step.
        features : array-like, xame as features
            Result of this augmentation step applied to features.
        """
        x_range = args['x_range'] if 'x_range' in args else (0, X.shape[2])
        y_range = args['y_range'] if 'y_range' in args else (0, X.shape[3])
        
        X = X[:, :, x_range[0]:x_range[1], y_range[0]:y_range[1]]

        return X, Y, features

    def select_dimensions(self, X, Y, features, args={}, verbose=0):
        """Selects which dimensions of X should be used 
        (e.g. only R,G from RGB image).
            
        Parameters
        ----------
        X : array-like
            Array of input patterns
        Y : array-like
            Array of input labels
        features : array-like
            Extra input features
        args : dictionary, default {}
            Contains the key:value pairs for the
            parameters of this augmentation:
                'dimensions': list or None, default None
                    Indices of dimensions that should be kept.
        verbose : integer, default 0
            Controls verbosity of this method

        Returns
        -------
        X : array-like, same as X
            Input patterns after application of this augmentation step.
        Y : array-like, same as Y
            Labels after application of this augmentation step.
        features : array-like, xame as features
            Result of this augmentation step applied to features.
        """        
        dims = args['dimensions'] if 'dimensions' in args else None
        
        if dims is not None:
            X = X[:, dims, :]

        return X, Y, features    

    def shift_images(self, X, Y, features, args={}, verbose=0):
        """Shift images horizontally and vertically, randomly
        selected from a given range. Shifting is done using
        the shift function from scipy.ndimage.interpolate().
            
        Parameters
        ----------
        X : array-like
            Array of input patterns
        Y : array-like
            Array of input labels
        features : array-like
            Extra input features
        args : dictionary, default {}
            Contains the key:value pairs for the
            parameters of this augmentation:
                'selected_classes': list or None, default None
                    Labels of classes to which this augmentation should 
                    be applied. Should be None for regression.
                'x_range': list, default [-2,0,2]
                    Possible number of pixels to shift the image in x direction.
                'y_range': list, default [-2,0,2]
                    Possible number of pixels to shift the image in y direction.
                'mode': string, default 'constant'
                    Mode of interpolation used for empty pixels. Possible
                    values are 'constant', 'nearest', 'reflect', 'wrap'. The
                    function used is scipy.ndimage.interpolate.shift().
                'cval': float, default 0.0
                    Constant value to use for all empty pixels. Only used when
                    'mode' is set to 'constant'.
        verbose : integer, default 0
            Controls verbosity of this method

        Returns
        -------
        X : array-like, same as X
            Input patterns after application of this augmentation step.
        Y : array-like, same as Y
            Labels after application of this augmentation step.
        features : array-like, xame as features
            Result of this augmentation step applied to features.
        """   
        selected_classes = args['selected_classes'] if 'selected_classes' in args else None
        xrang = args['x_range'] if 'x_range' in args else [-2,0,2]
        yrang = args['y_range'] if 'y_range' in args else [-2,0,2]
        mode = args['mode'] if 'mode' in args else 'constant'
        cval = args['cval'] if 'cval' in args else 0.0

        if selected_classes is not None:
            mask = np.in1d(Y, list(selected_classes))
        else:
            mask = np.ones(len(Y))
        shift_x = np.random.randint(len(xrang), size=len(Y))
        shift_y = np.random.randint(len(yrang), size=len(Y))

        for (i,b) in enumerate(mask):
            if b:
                for j in xrange(X.shape[1]):
                    X[i,j,:,:] = shift(X[i,j,:,:], 
                                       [xrang[shift_x[i]], yrang[shift_y[i]]],
                                       output=X.dtype, mode=mode, cval=cval)

        return X, Y, features

    
    def rotate_images(self, X, Y, features, args={}, verbose=0):
        """Rotate images, rotation angle randomly selected from
        a given range. Interpolation done using 
        scipy.ndimage.interpolation.rotate()
            
        Parameters
        ----------
        X : array-like
            Array of input patterns
        Y : array-like
            Array of input labels
        features : array-like
            Extra input features
        args : dictionary, default {}
            Contains the key:value pairs for the
            parameters of this augmentation:
                'selected_classes': list or None, default None
                    Labels of classes to which this augmentation should 
                    be applied. Should be None for regression.
                'rotations': list, default [0,90,180,270]
                    Possible values in degrees to rotate the image by.
        verbose : integer, default 0
            Controls verbosity of this method

        Returns
        -------
        X : array-like, same as X
            Input patterns after application of this augmentation step.
        Y : array-like, same as Y
            Labels after application of this augmentation step.
        features : array-like, xame as features
            Result of this augmentation step applied to features.
        """   

        rotations = args['rotations'] if 'rotations' in args else [0,90,180,270]
        selected_classes = args['selected_classes'] if 'selected_classes' in args else None

        if selected_classes is not None:
            mask = np.in1d(Y, list(selected_classes))
        else:
            mask = np.ones(len(Y))
        rotat = np.random.randint(len(rotations), size=len(Y))
        
        for (i,rot) in enumerate(mask):
            if (rot and rotat[i]):
                X[i] = rotate(X[i], rotations[rotat[i]], axes=(1, 2), reshape=False)

        return X, Y, features       

    
    def fliplr(self, X, Y, features, args={}, verbose=0):
        """Horizontally mirrors each input pattern with a given probability.
            
        Parameters
        ----------
        X : array-like
            Array of input patterns
        Y : array-like
            Array of input labels
        features : array-like
            Extra input features
        args : dictionary, default {}
            Contains the key:value pairs for the
            parameters of this augmentation:
                'selected_classes': list or None, default None
                    Labels of classes to which this augmentation should 
                    be applied. Should be None for regression.
                'prob': float, default 0.5
                    Probability that pattern is mirrored, value between
                    0 and 1.
        verbose : integer, default 0
            Controls verbosity of this method

        Returns
        -------
        X : array-like, same as X
            Input patterns after application of this augmentation step.
        Y : array-like, same as Y
            Labels after application of this augmentation step.
        features : array-like, xame as features
            Result of this augmentation step applied to features.
        """               
        selected_classes = args['selected_classes'] if 'selected_classes' in args else None
        prob = args['prob'] if 'prob' in args else 0.5


        if selected_classes is not None:
            mask = np.in1d(Y, list(selected_classes))
        else:
            mask = np.ones(len(Y))
        flips = np.random.rand(len(mask))
        flips = flips<prob
        mask = np.logical_and(mask,flips)

        for (i, flip) in enumerate(mask):          
            if flip:
                X[i] = X[i,:,:,::-1] 
    
        return X, Y, features

    def flipud(self, X, Y, features, args={}, verbose=0):
        """Vertically mirrors each input pattern with a given probability.
            
        Parameters
        ----------
        X : array-like
            Array of input patterns
        Y : array-like
            Array of input labels
        features : array-like
            Extra input features
        args : dictionary, default {}
            Contains the key:value pairs for the
            parameters of this augmentation:
                'selected_classes': list or None, default None
                    Labels of classes to which this augmentation should 
                    be applied. Should be None for regression.
                'prob': float, default 0.5
                    Probability that pattern is mirrored, value between
                    0 and 1.
        verbose : integer, default 0
            Controls verbosity of this method

        Returns
        -------
        X : array-like, same as X
            Input patterns after application of this augmentation step.
        Y : array-like, same as Y
            Labels after application of this augmentation step.
        features : array-like, xame as features
            Result of this augmentation step applied to features.
        """              
        selected_classes = args['selected_classes'] if 'selected_classes' in args else None
        prob = args['prob'] if 'prob' in args else 0.5

        if selected_classes is not None:
            mask = np.in1d(Y, list(selected_classes))
        else:
            mask = np.ones(len(Y))
        flips = np.random.rand(len(mask))
        flips = flips<prob
        mask = np.logical_and(mask,flips)

        for (i, flip) in enumerate(mask):
            if flip:
                X[i] = X[i,:,::-1,:]
    
        return X, Y, features


    def zoom_in(self, X, Y, features, args={}, verbose=0):
        """Zooms in on the image, zoom factor randomly
        selected from the given range of factors.
        Image then cropped to original size.
            
        Parameters
        ----------
        X : array-like
            Array of input patterns
        Y : array-like
            Array of input labels
        features : array-like
            Extra input features
        args : dictionary, default {}
            Contains the key:value pairs for the
            parameters of this augmentation:
                'selected_classes': list or None, default None
                    Labels of classes to which this augmentation should 
                    be applied. Should be None for regression.
                'factors': list, default [1.0, 1.1]
                    Possible zooming factors, should all be >= 1.
        verbose : integer, default 0
            Controls verbosity of this method

        Returns
        -------
        X : array-like, same as X
            Input patterns after application of this augmentation step.
        Y : array-like, same as Y
            Labels after application of this augmentation step.
        features : array-like, xame as features
            Result of this augmentation step applied to features.
        """

        selected_classes = args['selected_classes'] if 'selected_classes' in args else None
        factor = args['factors'] if 'factors' in args else [1.0, 1.1]

        if selected_classes is not None:
            mask = np.in1d(Y, list(selected_classes))
        else:
            mask = np.ones(len(Y))
        zooms = np.random.randint(len(factor), size=len(Y))

        imshape = (X.shape[2], X.shape[3])

        for (i,m) in enumerate(mask):
            if m:
                if factor[zooms[i]] == 1:
                    pass
                else:
                    for j in range(len(X[i])):
                        X[i,j] = self._crop_zoom(rescale(X[i,j], factor[zooms[i]], preserve_range=True), imshape)

        return X, Y, features

    def _crop_zoom(self, X, imshape):
        cur_x = X.shape[0]
        cur_y = X.shape[1]       
        hw = imshape[0]/2
        hh = imshape[1]/2 

        return X[cur_x/2-hw:cur_x/2+hw, cur_y/2-hh:cur_y/2+hh]


    def add_const(self, X, Y, features, args={}, verbose=0):
        """Adds a constant value to every input image. This
        can be either the same value for every image, or 
        a randomly selected value from a given range.
            
        Parameters
        ----------
        X : array-like
            Array of input patterns
        Y : array-like
            Array of input labels
        features : array-like
            Extra input features
        args : dictionary, default {}
            Contains the key:value pairs for the
            parameters of this augmentation:
                'range': int or list, default 10
                    Adds the same constant to all inputs if input
                    is an integer, otherwise adds a randomly 
                    selected constant from the given range.
        verbose : integer, default 0
            Controls verbosity of this method

        Returns
        -------
        X : array-like, same as X
            Input patterns after application of this augmentation step.
        Y : array-like, same as Y
            Labels after application of this augmentation step.
        features : array-like, xame as features
            Result of this augmentation step applied to features.
        """
        values = args['range'] if 'range' in args else [-50, 50]
        if isinstance(values, int):
            X += values
        else:
            X += np.random.rand(*X.shape) * (values[1]-values[0]) + values[0]

        return X, Y, features

    def add_noise(self, X, Y, features, args={}, verbose=0):
        """Adds random noise to each input pattern.
            
        Parameters
        ----------
        X : array-like
            Array of input patterns
        Y : array-like
            Array of input labels
        features : array-like
            Extra input features
        args : dictionary, default {}
            Contains the key:value pairs for the
            parameters of this augmentation:
                'scale': list or int, default 10
                    Value for the highest possible noise.
                    If a list, this highest value will be
                    randomly selected from the provided range.
                'distribution': string, default 'gaussian'
                    Type of distribution the noise should have.
                    Supported values: 'gaussian', 'uniform'
        verbose : integer, default 0
            Controls verbosity of this method

        Returns
        -------
        X : array-like, same as X
            Input patterns after application of this augmentation step.
        Y : array-like, same as Y
            Labels after application of this augmentation step.
        features : array-like, xame as features
            Result of this augmentation step applied to features.
        """
        values = args['scale'] if 'scale' in args else 10
        dist = args['distribution'] if 'distribution' in args else 'gaussian'

        if isinstance(values, int):
            if dist == 'gaussian':
                X += np.random.normal(scale=values, size=X.shape)
            elif dist == 'uniform':
                X += np.random.uniform(low=0, high=values, size=X.shape)

        else:
            scales = np.random.rand(len(Y)) * (values[1]-values[0]) + values[0]
            if dist == 'gaussian':
                for i in range(len(X)):
                    X[i] += np.random.normal(scale=values[i], size=X[i].shape)
            elif dist == 'uniform':
                for i in range(len(X)):
                    X[i] += np.random.uniform(low=0, high=values[i], size=X[i].shape)
                
        return X, Y, features

    def edge_error(self, X, Y, features, args={}, verbose=0):
        """Sets one side of the input pattern to a small
        value, as seen in real CCD data.
            
        Parameters
        ----------
        X : array-like
            Array of input patterns
        Y : array-like
            Array of input labels
        features : array-like
            Extra input features
        args : dictionary, default {}
            Contains the key:value pairs for the
            parameters of this augmentation:
                'selected_classes': list or None, default None
                    Labels of classes to which this augmentation should 
                    be applied. Should be None for regression.
                'prob': float, default 0.5
                    Probability that an edge is set to zero, value 
                    between 0 and 1.
        verbose : integer, default 0
            Controls verbosity of this method

        Returns
        -------
        Xtransformed : array-like, same as X
            Input patterns after application of this augmentation step.
        Y : array-like, same as Y
            Labels after application of this augmentation step.
        features : array-like, xame as features
            Result of this augmentation step applied to features.
        """
        selected_classes = args['selected_classes'] if 'selected_classes' in args else None
        prob = args['prob'] if 'prob' in args else 0.1

        if selected_classes is not None:
            mask = np.in1d(Y, list(selected_classes))
        else:
            mask = np.ones(len(Y))
        add = np.random.random(len(Y)) < prob
        
        Xtransformed = copy.deepcopy(X)
        for i in range(len(Xtransformed)):
            if add[i]:
                Xtransformed[i] = self._add_edge_error(Xtransformed[i])

        return Xtransformed, Y, features

    def _add_edge_error(self, im):
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

    def add_dead_column(self, X, Y, features, args={}, verbose=0):
        """Sets a random column for an input pattern to zero with
        a given probability, as is common for faulty CCDs.
            
        Parameters
        ----------
        X : array-like
            Array of input patterns
        Y : array-like
            Array of input labels
        features : array-like
            Extra input features
        args : dictionary, default {}
            Contains the key:value pairs for the
            parameters of this augmentation:
                'selected_classes': list or None, default None
                    Labels of classes to which this augmentation should 
                    be applied. Should be None for regression.
                'prob': float, default 0.5
                    Probability that an input patterns will have
                    a dead column. Value between 0 and 1.
        verbose : integer, default 0
            Controls verbosity of this method

        Returns
        -------
        Xtransformed : array-like, same as X
            Input patterns after application of this augmentation step.
        Y : array-like, same as Y
            Labels after application of this augmentation step.
        features : array-like, xame as features
            Result of this augmentation step applied to features.
        """
        selected_classes = args['selected_classes'] if 'selected_classes' in args else None
        prob = args['prob'] if 'prob' in args else 0.1

        if selected_classes is not None:
            mask = np.in1d(Y, list(selected_classes))
        else:
            mask = np.ones(len(Y))
        add = np.random.random((X.shape[1],len(Y))) < prob

        Xtransformed = copy.deepcopy(X)
        for i in range(len(Xtransformed)):
            for col in range(X.shape[1]):
                if add[col,i]:
                    col_i = np.random.randint(X.shape[2])
                    Xtransformed[i,col,:,col_i] = 0

        return Xtransformed, Y, features

    # TO DO:
    # Currently only creates stars with perfect Gaussian PSF
    def add_star(self, X, Y, features, args={}, verbose=0):
        """Adds a foreground star to an image with a given
        probability. Foreground star has a Gaussian PSF.
            
        Parameters
        ----------
        X : array-like
            Array of input patterns
        Y : array-like
            Array of input labels
        features : array-like
            Extra input features
        args : dictionary, default {}
            Contains the key:value pairs for the
            parameters of this augmentation:
                'selected_classes': list or None, default None
                    Labels of classes to which this augmentation should 
                    be applied. Should be None for regression.
                'prob': float, default 0.5
                    Probability that a foreground star is added.
                    Value between 0 and 1.
                'L_range': list, default [0,100]
                    Range of values from which the brightness of the 
                    added star is selected.
                'var_range': list, default [1,3]
                    Range of values from which the width of the added
                    star is selected.
        verbose : integer, default 0
            Controls verbosity of this method

        Returns
        -------
        Xtransformed : array-like, same as X
            Input patterns after application of this augmentation step.
        Y : array-like, same as Y
            Labels after application of this augmentation step.
        features : array-like, xame as features
            Result of this augmentation step applied to features.
        """
        selected_classes = args['selected_classes'] if 'selected_classes' in args else None
        prob = args['prob'] if 'prob' in args else 0.5
        mean_range = args['L_range'] if 'L_range' in args else [0,100]
        var_range = args['var_range'] if 'var_range' in args else [1,3]

        if selected_classes is not None:
            mask = np.in1d(Y, list(selected_classes))
        else:
            mask = np.ones(len(Y))
        add = np.random.random(len(Y)) < prob
        
        Xtransformed = copy.deepcopy(X)
        for i in range(len(Xtransformed)):
            if add[i]:
                Xtransformed[i,:] += self._add_star(Xtransformed[i], mean_range, var_range)

        return Xtransformed, Y, features

    def _add_star(self, im, means_range, var_range):
        shape = im.shape[1:]
        x_c = np.random.randint(shape[0])
        y_c = np.random.randint(shape[1])

        mean, var = np.random.rand(2)
        mean = means_range[0] + mean*(means_range[1]-means_range[0])
        var = var_range[0] + var*(var_range[1]-var_range[0])

        delta_im = np.zeros((2*shape[0]+1, 2*shape[1]+1))
        delta_im[shape[0], shape[1]] = 1
        to_add = gaussian_filter(delta_im, var)[shape[0]-x_c:2*shape[0]-x_c, shape[1]-y_c:2*shape[1]-y_c]
        to_add = to_add*mean

        return to_add.astype(np.float32)


    def normalize(self, X, Y, features, args={}, verbose=0):
        """Normalizes the input by computing the mean and
        standard deviation of the current batch. The
        mean is subtracted from the input and the input
        is then divided by the standard deviation.
            
        Parameters
        ----------
        X : array-like
            Array of input patterns
        Y : array-like
            Array of input labels
        features : array-like
            Extra input features
        args : dictionary, default {}
            This method does not have any parameters.
        verbose : integer, default 0
            Controls verbosity of this method

        Returns
        -------
        Xtransformed : array-like, same as X
            Input patterns after application of this augmentation step.
        Y : array-like, same as Y
            Labels after application of this augmentation step.
        features : array-like, xame as features
            Result of this augmentation step applied to features.
        """
        Xtransformed = copy.deepcopy(X)
        
        mean = [np.mean(X[:,i]) for i in range(X.shape[1])]
        std = [np.std(X[:,i]) for i in range(X.shape[1])]   

        for i in range(X.shape[1]):
            Xtransformed[:,i] = X[:,i] - mean[i]
            Xtransformed[:,i] = Xtransformed[:,i]/std[i]

        return Xtransformed, Y, features
