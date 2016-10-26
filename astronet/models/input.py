'''
Created on 01.06.2016

@authors: jonaskindler and fgieseke
'''

import sys, os
import copy
import numpy as np
import scipy.ndimage.interpolation
from scipy.ndimage.interpolation import shift
from scipy.ndimage.filters import gaussian_filter
from skimage.transform import rescale
import skimage.io as io
from nolearn.lasagne import BatchIterator


import matplotlib.pyplot as plt

class AugmentationBatchIterator(BatchIterator):
    def __init__(self, batch_size, augments=[], balanced=False, regression=False, seed=0, verbose=1):
        super(AugmentationBatchIterator, self).__init__(batch_size, shuffle=True, seed=seed)

        self._methods = {
            "extend":AugmentationBatchIterator.extend_data,
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
            "submean":AugmentationBatchIterator.submean,
            "submean_per_image":AugmentationBatchIterator.submean_per_image,
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
        self.augments.append( (method, args) )

    def transform(self, Xb, yb):
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
        
        crop_ranges = args['ranges'] if 'ranges' in args else None
        
        if crop_ranges is not None:
            X = X[:, :, crop_ranges[0][0]:crop_ranges[0][1], crop_ranges[1][0]:crop_ranges[1][1]]

        return X, Y, features
    
    def select_dimensions(self, X, Y, features, args={}, verbose=0):
        
        dims = args['dimensions'] if 'dimensions' in args else None
        
        if dims is not None:
            X = X[:, dims, :]

        return X, Y, features    
    
    def shift_images(self, X, Y, features, args={}, verbose=0):
        
        selected_classes = args['selected_classes'] if 'selected_classes' in args else None

        xrang = args['xrange'] if 'xrange' in args else [2,4]
        yrang = args['yrange'] if 'yrange' in args else [2,4]

        if selected_classes is not None:
            mask = np.in1d(Y, list(selected_classes))
        else:
            mask = np.ones(len(Y))
        shift_x = np.random.randint(len(xrang), size=len(Y))
        shift_y = np.random.randint(len(yrang), size=len(Y))

        for (i,b) in enumerate(mask):
            if b:
                for j in xrange(X.shape[1]):
                    X[i,j,:,:] = shift(X[i,j,:,:], [xrang[shift_x[i]], yrang[shift_y[i]]], output=X.dtype)

        return X, Y, features

    
    def rotate_images(self, X, Y, features, args={}, verbose=0):
        rotations = args['rotations'] if 'rotations' in args else [0,90,180,270]
        selected_classes = args['selected_classes'] if 'selected_classes' in args else None


        if selected_classes is not None:
            mask = np.in1d(Y, list(selected_classes))
        else:
            mask = np.ones(len(Y))
        rotate = np.random.randint(len(rotations), size=len(Y))
        
        for (i,rot) in enumerate(mask):
            if (rot and rotate[i]):
                X[i] = scipy.ndimage.interpolation.rotate(X[i], rotations[rotate[i]], axes=(1, 2), reshape=False)

        return X,Y,features       

    
    def extend_data(self, X, Y, features, args={}, verbose=0):
    
        cl = args['class']
        data = args['data']        

        if verbose > 0:
            print("Extending class %s with %i elements ..." % (str(cl), len(data)))
            
        assert features is None
        
        X_extended = [X, data]
        Y_extended = [Y, cl * np.ones(len(data))]
        
        newX = np.concatenate(X_extended, axis=0)
        newY = np.concatenate(Y_extended, axis=0)
        
        return newX, newY, None            
    
    def fliplr(self, X, Y, features, args={}, verbose=0):
        
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
        selected_classes = args['selected_classes'] if 'selected_classes' in args else None
        factor = args['factor'] if 'factor' in args else [1.0, 1.1]

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
        values = args['range'] if 'range' in args else [-50, 50]

        const = np.random.rand() * (values[1]-values[0]) + values[0]
        X += const

        return X, Y, features

    def add_noise(self, X, Y, features, args={}, verbose=0):
        value = args['range'] if 'range' in args else 10

        shape = X.shape
        X += np.random.normal(scale=value, size=shape)

        return X, Y, features

    def edge_error(self, X, Y, features, args={}, verbose=0):
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

    #NEEDS TO BE LOOKED AT
    def add_star(self, X, Y, features, args={}, verbose=0):
        selected_classes = args['selected_classes'] if 'selected_classes' in args else None
        prob = args['prob'] if 'prob' in args else 0.999
        mean_range = args['L_range'] if 'L_range' in args else [90000, 100000]
        var_range = args['var_range'] if 'var_range' in args else [2,6]

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

    def submean(self, X, Y, features, args={}, verbose=0):
        
        fit = args['fit'] if 'fit' in args else True
        
        if fit == True:
            self._methods_cache["submean"] = {}
            for i in xrange(X.shape[1]):
                self._methods_cache["submean"][i] = np.mean(X[:,i,:,:], axis=0)
        
        Xtransformed = copy.deepcopy(X)
        for i in xrange(X.shape[1]):
            try:
                Xtransformed[:,i,:,:] = X[:,i,:,:] - self._methods_cache["submean"][i]
            except Exception as e:
                print("Could not applying 'submean': %s" % str(e))    
        
        return Xtransformed, Y, features
    
    def submean_per_image(self, X, Y, features, args={}, verbose=0):
        
        dims = args['dimensions'] if 'dimensions' in args else None
        subtype = args['subtype'] if 'subtype' in args else "mean_all"
        
        Xtransformed = copy.deepcopy(X)
        if dims is None:
            dims = range(Xtransformed.shape[1])
        
        if subtype == "mean_all":    
            for i in xrange(Xtransformed.shape[0]):
                for j in dims:
                    Xtransformed[i,j,:,:] = X[i,j,:,:] - np.mean(X[i,j,:,:])

        elif subtype == "mean_edge":
            for i in xrange(Xtransformed.shape[0]):
                for j in dims:
                    mean = (np.sum(X[i,j,:,:]) - np.sum(X[i,j,20:30, 20:30]))/2400
                    Xtransformed[i,j,:,:] = X[i,j,:,:] - mean

        elif subtype == "median_all":
            for i in xrange(Xtransformed.shape[0]):
                for j in dims:
                    Xtransformed[i,j,:,:] = X[i,j,:,:] - np.median(X[i,j,:,:])

        elif subtype == "median_edge":
            for i in xrange(Xtransformed.shape[0]):
                for j in dims:
                    Xtransformed[i,j,20:30, 20:30] = np.amin(X[i,j,:,:])
                    median = np.median(np.sort(Xtransformed[i,j,:,:], axis=None)[100:])
                    Xtransformed[i,j,:,:] = X[i,j,:,:] - median
                    Xtransformed[i,j,20:30,20:30] = X[i,j,20:30,20:30]
        
        return Xtransformed, Y, features    
