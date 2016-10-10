'''
Created on 01.06.2016

@authors: jonaskindler and fgieseke
'''

import copy
import numpy as np
import scipy.ndimage.interpolation
from scipy.ndimage.interpolation import shift
from skimage.transform import rescale
from nolearn.lasagne import BatchIterator

from .zca import ZCA

import matplotlib.pyplot as plt

class AugmentationBatchIterator(BatchIterator):
    def __init__(self, features, batch_size, augments=[], balanced=False, seed=0, verbose=1):
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
            "submean":AugmentationBatchIterator.submean,
            "submean_per_image":AugmentationBatchIterator.submean_per_image,
            "whitening":AugmentationBatchIterator.whitening,
        }

        self.features = features
        self.augments = augments

        self.balanced = balanced

        self._methods_cache = {}        
        self.verbose = verbose
        self.seed = seed
        
        np.random.seed(self.seed)
        if self.verbose > 0:
            print("\n---------------------------------------")
            print("---------- DATA AUGMENTATION ----------")
            print("---------------------------------------")
            for aug in augments:
                print("-> Applying method '%s': " % (str(aug)) )

    def transform(self, Xb, yb):
        X = copy.deepcopy(Xb)
        y = copy.deepcopy(yb)
        Xf = copy.deepcopy(self.features)

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
        
        selected_classes = args['selected_classes'] if 'selected_classes' in args else set(Y)

        xrang = args['xrange'] if 'xrange' in args else [2,4]
        yrang = args['yrange'] if 'yrange' in args else [2,4]

        mask = np.in1d(Y, list(selected_classes))
        shift_x = np.random.randint(len(xrang), size=len(Y))
        shift_y = np.random.randint(len(yrang), size=len(Y))

        for (i,b) in enumerate(mask):
            if b:
                for j in xrange(X.shape[1]):
                    X[i,j,:,:] = shift(X[i,j,:,:], [xrang[shift_x[i]], yrang[shift_y[i]]], output=X.dtype)

        return X, Y, features

    
    def rotate_images(self, X, Y, features, args={}, verbose=0):
        rotations = args['rotations'] if 'rotations' in args else [0,90,180,270]
        selected_classes = args['selected_classes'] if 'selected_classes' in args else set(Y)

        mask = np.in1d(Y, list(selected_classes))
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
        
        selected_classes = args['selected_classes'] if 'selected_classes' in args else set(Y)
        prob = args['prob'] if 'prob' in args else 0.5

        mask = np.in1d(Y, list(selected_classes))
        flips = np.random.rand(len(mask))
        flips = flips<prob
        mask = np.logical_and(mask,flips)

        for (i, flip) in enumerate(mask):          
            if flip:
                X[i] = X[i,:,:,::-1] 
    
        return X, Y, features

    def flipud(self, X, Y, features, args={}, verbose=0):

        selected_classes = args['selected_classes'] if 'selected_classes' in args else set(Y)
        prob = args['prob'] if 'prob' in args else 0.5

        mask = np.in1d(Y, list(selected_classes))
        flips = np.random.rand(len(mask))
        flips = flips<prob
        mask = np.logical_and(mask,flips)

        for (i, flip) in enumerate(mask):
            if flip:
                X[i] = X[i,:,::-1,:]
    
        return X, Y, features

    def zoom_in(self, X, Y, features, args={}, verbose=0):
        selected_classes = args['selected_classes'] if 'selected_classes' in args else set(Y) 
        factor = args['factor'] if 'factor' in args else [1.0, 1.1]

        mask = np.in1d(Y, list(selected_classes))
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
    
    def whitening(self, X, Y, features, args={}, verbose=0):
        """ Assuming X of shape [N, F, n, m], where N is
        the number of instances, F the number of images per
        instance, and nxm the size of each image.
        """
        
        fit = args['fit'] if 'fit' in args else True
        
        if fit == True:
            self._methods_cache["whitening"] = {}
            for i in xrange(X.shape[1]):
                whitening = ZCA()
                whitening.fit(X[:,i,:,:].reshape((X[:,i,:,:].shape[0], -1)))
                self._methods_cache["whitening"][i] = whitening
        
        Xtransformed = copy.deepcopy(X)
        for i in xrange(X.shape[1]):
            try:
                whitening = self._methods_cache["whitening"][i]
                tmp = whitening.transform(X[:,i,:,:].reshape((X[:,i,:,:].shape[0], -1)))
                Xtransformed[:,i,:,:] = tmp.reshape(X[:,i,:,:].shape)
            except Exception as e:
                print("Could not applying 'whitening': %s" % str(e))

#             Xsub = X[:,i,:,:]
#             # generate N x (nm) array and center it
#             Xnew = Xsub.reshape((Xsub.shape[0], -1))
#             Xcentered = Xnew - np.mean(Xnew, axis=0)
#             
#             # whitening
#             cov = np.dot(Xcentered.T, Xcentered)
#             U, S, V = np.linalg.svd(cov)
#             Xrot = np.dot(Xcentered, U)
#             Xwhite = Xrot / np.sqrt(S + 1e-5)

#             Xtransformed[:,i,:,:] = Xwhite.reshape(Xsub.shape)
            
        return Xtransformed, Y, features

class InputHandler(object):
        
    def __init__(self, seed=0, verbose=0):
        
        self._methods = {
            "select_dimensions":InputHandler.select_dimensions,
            "crop_images":InputHandler.crop_images,
            "shift_images":InputHandler.shift_images,
            "rotate_images":InputHandler.rotate_images,
            "fliplr":InputHandler.fliplr,
            "flipud":InputHandler.flipud,
        }

        self._methods_cache = {}        
        self.verbose = verbose
        self.seed = seed
        
        np.random.seed(self.seed)
                                    
    def apply(self, X, XF, y, method=None, params={}):
        
        if not method in self._methods:
                raise RuntimeError(str(method) + " is not a valid input_action!")
        
        if self.verbose > 0:
            print("-> Applying method '%s' with parameters: %s" % (str(method), str(params)[:50] + "..."))
            
        X, y, XF = self._methods[method](self, X, y, XF, args=params, verbose=self.verbose)
            
        return X, XF, y

    def select_dimensions(self, X, Y, features, args={}, verbose=0):
        
        dims = args['dimensions'] if 'dimensions' in args else None
        
        if dims is not None:
            X = X[:, dims, :]

        return X, Y, features    
        
    def crop_images(self, X, Y, features, args={}, verbose=0):
        
        crop_ranges = args['ranges'] if 'ranges' in args else None
        
        if crop_ranges is not None:
            X = X[:, :, crop_ranges[0][0]:crop_ranges[0][1], crop_ranges[1][0]:crop_ranges[1][1]]

        return X, Y, features
    
    def shift_images(self, X, Y, features, args={}, verbose=0):
        
        selected_classes = args['selected_classes'] if 'selected_classes' in args else set(Y)

        mask = np.in1d(Y, list(selected_classes))
        if features is not None:
            X_mask, Y_mask, features_mask = X[mask], Y[mask], features[mask]
        else:
            X_mask, Y_mask, features_mask = X[mask], Y[mask], None
                    
        xrang = args['xrange'] if 'xrange' in args else [2,4]
        yrang = args['yrange'] if 'yrange' in args else [2,4]
        
        X_shifted = []
        Y_shifted = []
        features_shifted = []
        
        for x in xrang:
            for y in yrang: 
                images_shifted = self._shift_batch_images(X_mask, x, y)
                X_shifted.append(images_shifted)
                Y_shifted.append(Y_mask)
                if features is not None:
                    features_shifted.append(features_mask)
                    
        X_shifted.append(X)
        Y_shifted.append(Y)
        if features is not None:  
            features_shifted.append(features)
            
        Xnew = np.concatenate(X_shifted, axis=0)
        Ynew = np.concatenate(Y_shifted, axis=0)
        
        if features is not None:
            featuresnew = np.concatenate(features_shifted, axis=0)
        else:
            featuresnew = None                
        
        return Xnew, Ynew, featuresnew
                                    
    def _shift_batch_images(self, X, x, y):
        
        Xnew = copy.deepcopy(X)
        for i in xrange(X.shape[0]):
            for j in xrange(X.shape[1]):
                Xnew[i,j,:,:] = shift(X[i,j,:,:], [x,y], output=X.dtype)
                
        return Xnew
    
    def rotate_images(self, X, Y, features, args={}, verbose=0):
        
        rotations = args['rotations'] if 'rotations' in args else range(5, 361, 60)
        selected_classes = args['selected_classes'] if 'selected_classes' in args else set(Y)
        
        mask = np.in1d(Y, list(selected_classes))
        if features is not None:
            X_mask, Y_mask, features_mask = X[mask], Y[mask], features[mask]
        else:
            X_mask, Y_mask, features_mask = X[mask], Y[mask], None
         
        X_rotated = []
        Y_rotated = []
        features_rotated = []
        
        for i in range(len(rotations)):
            rot_imgs = scipy.ndimage.interpolation.rotate(X_mask, rotations[i], axes=(2, 3), reshape=False)
            X_rotated.append(rot_imgs)
            Y_rotated.append(Y_mask)
            if features is not None:
                features_rotated.append(features_mask)

        X_rotated.append(X)
        Y_rotated.append(Y)
        if features is not None:  
            features_rotated.append(features)
                  
        # combine all rotated images to single array and
        # generate both extended labels and features
        newX = np.concatenate(X_rotated, axis=0)
        newY = np.concatenate(Y_rotated, axis=0)
        if features is not None:
            newFeatures = np.concatenate(features_rotated, axis=0)
        else:
            newFeatures = None

        return newX, newY, newFeatures                 
    
    def fliplr(self, X, Y, features, args={}, verbose=0):
        
        selected_classes = args['selected_classes'] if 'selected_classes' in args else set(Y)

        mask = np.in1d(Y, list(selected_classes))
        if features is not None:
            X_mask, Y_mask, features_mask = X[mask], Y[mask], features[mask]
        else:
            X_mask, Y_mask, features_mask = X[mask], Y[mask], None
                    
        retX = np.concatenate([X_mask[:, :, :, ::-1], X], axis=0)
        retY = np.concatenate([Y_mask, Y], axis=0)
        
        if features is not None:
            retFeatures = np.concatenate([features_mask, features], axis=0)
        else:
            retFeatures = None
    
        return retX, retY, retFeatures

    def flipud(self, X, Y, features, args={}, verbose=0):

        selected_classes = args['selected_classes'] if 'selected_classes' in args else set(Y)

        mask = np.in1d(Y, list(selected_classes))
        if features is not None:
            X_mask, Y_mask, features_mask = X[mask], Y[mask], features[mask]
        else:
            X_mask, Y_mask, features_mask = X[mask], Y[mask], None
                    
        retX = np.concatenate([X_mask[:, :, ::-1, :], X], axis=0)
        retY = np.concatenate([Y_mask, Y], axis=0)
        
        if features is not None:
            retFeatures = np.concatenate([features_mask, features], axis=0)
        else:
            retFeatures = None
    
        return retX, retY, retFeatures
