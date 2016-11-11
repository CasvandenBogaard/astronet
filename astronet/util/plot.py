'''
Created on 26.07.2016

@author: Fabian Gieseke, Cas van den Bogaard
'''

#import matplotlib
#matplotlib.use('Agg')

import os
import numpy as np
import theano
import theano.tensor as T
from lasagne.layers import get_output
from itertools import product

from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn_evaluation import plot as sklearnplot
from sklearn.metrics import matthews_corrcoef, accuracy_score, precision_score, recall_score, roc_curve
import matplotlib.ticker as ticker
from nolearn.lasagne.visualize import occlusion_heatmap, plot_conv_activity
from sklearn import metrics
import seaborn
import pandas

from .io import store_results, ensure_dir

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


def _colorbar_fmt(x, pos):
    fm = '% *d' % (5, x)
    return fm
        
    # TO DO:
    # Allow both RGB images and a subplot for each layer.
def plot_image(img, ofname, titles=None, figsize=(10,5), mode="2d"):
    """ Plots and saves a given image, one subplot for each dimension.
        
    Parameters
    ----------
    img : array-like
        The image that is to be plotted.
    ofname : string
        Filename for the output image.
    titles : list, default None
        A list of strings that are used as title for each
        of the plotted dimensions.
    figsize: tuple, default (10,5)
        Size of the output figure.
    mode : string
        Mode of plotting. Either '2d' or '3d'.
    """
    
    assert mode in ["2d", "3d"]
    
    fig = plt.figure(tight_layout=True, figsize=figsize)
    
    for i in range(img.shape[0]):
        
        if mode == "3d":
            ax = fig.add_subplot(1, img.shape[0], i + 1, projection='3d')
            X, Y = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[2]))        
            surf = ax.plot_surface(X, Y, img[i, :, :], rstride=1, cstride=1,
                                   cmap=cm.coolwarm, linewidth=0, antialiased=False)
            ax.view_init(15, 15)
            if titles is not None:
                ax.set_title(titles[i])            
            
        elif mode == "2d":
            ax = fig.add_subplot(1, img.shape[0], i + 1)
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            if titles is not None:
                ax.set_title(titles[i])
            im = ax.imshow(img[i,:,:])
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax,  format=ticker.FuncFormatter(_colorbar_fmt))
        else:    
            raise Exception("Unknown plotting mode: %s" % str(mode))
    
    
    plt.savefig(ofname, bbox_inches='tight')
    plt.close()

# TO DO:
# Add plot_loss and draw_to_file
class PlotHandler():
    """ Class that helps with the plotting of graphs
        after testing of the performance.
        
    Parameters
    ----------
    plots : list
        List of tuples, the first value being the
        name of the plotting method, the second a
        dictionary of parameters for that method.
        Example:
        plots = [
                    ('confusion', {}),
                    ('roc', {'title': "ROC-curve"}),
                ]   
    preds : array-like
        Class predictions that the model has made.
    preds_proba : array-like
        Prediction probabilities that the model
        has made.
    y_test : array-like
        True labels for the test patterns.
    X_test : array-like
        The test patterns.
    indices_test : array-like
        
    model : AstroWrapper
        The AstroWrapper model used to make these predictions.
    odir : string
        Relative path to the output directory.
    verbose: integer, default 0
        Verbosity level.
    """
    def __init__(self, plots, preds, preds_proba, y_test, X_test, indices_test, model, odir, verbose=0):
        self._methods = {
            "confusion": self.confusion,
            "roc": self.roc,
            "occlusion": self.occlusion,
            "conv weights": self.conv_weights,
            "conv activation": self.conv_activation,
            "misses": self.misses,
            }

        self.preds = preds
        self.preds_proba = preds_proba
        self.y_test = y_test
        self.X_test = X_test
        self.indices_test = indices_test
        self.model = model
        self.network = model.model
        self.odir = odir

        self.plots = plots

        self.verbose = verbose

        self.X_test_trans, self.y_test_trans = self.model.ABI_test.transform(self.X_test, self.y_test)

    def plot(self):
        """ Method to calls all selected plot methods.
        """
        for pl in self.plots:
            self._methods[pl[0]](pl[1], self.odir, self.verbose)

    def confusion(self,args,odir,verbose):
        """ Plots and saves the confusion matrix.
            Only works for classification problems.
            
        Parameters
        ----------
        args : dictionary, default {}
            Contains the key:value pairs for the
            parameters of this plot:
                'clabels': list, default ["Negative", "Positive"]
                    Labels for the negative and positive classes.
                'ofname': string, default "confusion.png"
                    Name for the output file.
                'figsize': tuple, default (10,10)
                    Size of the output figure.
                'cmap': string, default "YlGnBu"
                    Name of the Seaborn color map that should be used.
        odir : string
            Relative path to the output directory.
        verbose : integer, default 0
            Verbosity level.
        """
        clabels =   args['clabels']  if 'clabels'   in args else ["Negative", "Positive"]
        ofname =    args['ofname']   if 'ofname'    in args else "confusion.png"
        figsize =   args['figsize']  if 'figsize'   in args else (10,10)
        cmap =      args['cmap']     if 'cmap'      in args else "YlGnBu"

        cm = metrics.confusion_matrix(self.y_test, self.preds)
        df_cm = pandas.DataFrame(cm, index=clabels, columns = clabels)
        plt.figure(figsize=figsize)
        seaborn.set(font_scale=6.5)
        seaborn.heatmap(df_cm, annot=True, fmt="d",  linewidths=.5,  cmap=cmap, square=True, cbar=False)
        plt.xlabel("Prediction")
        plt.ylabel("True Class")
        plt.savefig(os.path.join(odir,ofname), bbox_inches='tight')
        plt.close()

    def roc(self,args,odir,verbose):
        """ Plots and saves the ROC curve.
            Only works for classification problems.
            
        Parameters
        ----------
        args : dictionary, default {}
            Contains the key:value pairs for the
            parameters of this plot:
                'pos_label': string or integer, default 1
                    Value of the label of the positive class
                'ofname': string, default "confusion.png"
                    Name for the output file.
                'title': string or None, default None
                    Title to be used in the plot. Set to
                    None for no title.
                'figsize': tuple, default (10,10)
                    Size of the output figure.
        odir : string
            Relative path to the output directory.
        verbose : integer, default 0
            Verbosity level.
        """
        pos_label = args['pos_label']   if 'pos_label'  in args else 1
        title =     args['title']       if 'title'      in args else None
        ofname =    args['ofname']      if 'ofname'     in args else "roc.png"
        figsize =   args['figsize']     if 'figsize'    in args else (10,10)

        fpr, tpr, thres = roc_curve(self.y_test, self.preds_proba[:,pos_label], pos_label=pos_label)
        plt.plot(fpr, tpr)
        if title:
            plt.title(title)
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.savefig(os.path.join(odir, ofname))
        plt.close()

    def occlusion(self,args,odir,verbose):
        """ Computes and plots the occlusion heatmap.
            This requires testing the image once for
            every pixel and thus may be slow.
            
        Parameters
        ----------
        args : dictionary, default {}
            Contains the key:value pairs for the
            parameters of this plot:
                'image_range': slice, default slice(0,1)
                    The indices of the images for which to
                    compute the occlusion heatmap.
                'ofname': string, default "occlusion"
                    Name for the output files, without extension.
                'title': string or None, default None
                    Title to be used in the plot. Set to
                    None for no title.
                'figsize': tuple, default (10,10)
                    Size of the output figure.
                'square_length': integer, default 5
                    Length of the sides of the square used
                    in the occlusion.
        odir : string
            Relative path to the output directory.
        verbose : integer, default 0
            Verbosity level.
        """
        im_range =  args['image_range']     if 'image_range'    in args else slice(0,1)
        title =     args['title']           if 'title'          in args else None
        ofname =    args['ofname']          if 'ofname'         in args else "occlusion"
        figsize =   args['figsize']         if 'figsize'        in args else (10,10)
        length =    args['square_length']   if 'square_length'  in args else 5

        imgs = self.X_test[im_range]
        labels = self.y_test[im_range]

        for i in range(len(labels)):
            hm = occlusion_heatmap(self.network, imgs[i:i+1], labels[i], square_length=length)
            
            fig = plt.figure(tight_layout=True, figsize=figsize)
            for j in range(len(imgs[i])):
                ax = fig.add_subplot(1, len(imgs[i])+1, j + 1)
                ax.xaxis.set_visible(False)
                ax.yaxis.set_visible(False)

                im = ax.imshow(imgs[i,j,:,:])
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax, format=ticker.FuncFormatter(_colorbar_fmt))    

            ax = fig.add_subplot(1, len(imgs[i])+1, len(imgs[i])+1)
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)

            im = ax.imshow(hm, vmin=0, vmax=1)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax)    
            cbar.set_ticks([0,0.2,0.4,0.6,0.8,1])
            cbar.set_ticklabels([0,0.2,0.4,0.6,0.8,1])

            plt.savefig(os.path.join(odir, ofname+str(i)+".png"), bbox_inches='tight')
            plt.close()          

    def conv_weights(self,args,odir,verbose):
        """ Plots the weights for a single 
            convolutional layer.
            
        Parameters
        ----------
        args : dictionary, default {}
            Contains the key:value pairs for the
            parameters of this plot:
                'layer': integer, default 1
                    Index of the layer for which the weights
                    are plotted.
                'ofname': string, default "occlusion"
                    Name for the output files, without extension.
                'figsize': tuple, default (10,10)
                    Size of the output figure.
        odir : string
            Relative path to the output directory.
        verbose : integer, default 0
            Verbosity level.
        """
        layer_i =  args['layer']    if 'layer'      in args else 1
        ofname =   args['ofname']   if 'ofname'     in args else "weights"
        figsize =  args['figsize']  if 'figsize'    in args else (10,10)

        layer = self.network.layers_[layer_i]
        W = layer.W.get_value()
        shape = W.shape
        nrows = np.ceil(np.sqrt(shape[0])).astype(int)
        ncols = nrows

        vmin = np.min(W)
        vmax = np.max(W)

        for feature_map in range(shape[1]):
            fig, axes = plt.subplots(int(nrows), int(ncols), figsize=figsize, squeeze=False)

            for i, ax in enumerate(axes.flatten()):
                ax.set_xticks([])
                ax.set_yticks([])
                ax.axis('off')

                if i < shape[0]:
                    ax.set_title( str(round(W[i, feature_map].sum(), 3)), fontsize=10)
                    im = ax.imshow(W[i, feature_map], cmap='gray', interpolation='none', vmin=vmin, vmax=vmax)

            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            fig.colorbar(im, cax=cbar_ax)

            fig.savefig( os.path.join(odir, ofname+str(feature_map)+".png") )

    def conv_activation(self,args,odir,verbose):
        """ Plots the activations of a single  
            convolutional layer for a given image.
            
        Parameters
        ----------
        args : dictionary, default {}
            Contains the key:value pairs for the
            parameters of this plot:
                'layer': integer, default 1
                    Index of the convolutional layer
                    for which the activations are plotted.
                'image_index': integer, default 0
                    Index of the image for which the 
                    activations are plotted.
                'ofname': string, default "occlusion"
                    Name for the output file.
                'figsize': tuple, default (10,10)
                    Size of the output figure.
        odir : string
            Relative path to the output directory.
        verbose : integer, default 0
            Verbosity level.
        """
        layer_i =  args['layer']        if 'layer'          in args else 1
        image_i =  args['image index']  if 'image_index'    in args else 0
        ofname =   args['ofname']       if 'ofname'         in args else "activity.png"
        figsize =  args['figsize']      if 'figsize'        in args else (10,10)      

        x = self.X_test_trans[image_i:image_i+1]
        layer = self.network.layers_[layer_i]

        if x.shape[0] != 1:
            raise ValueError("Only one sample can be plotted at a time.")

        # compile theano function
        xs = T.tensor4('xs').astype(theano.config.floatX)
        get_activity = theano.function([xs], get_output(layer, xs))

        activity = get_activity(x)
        shape = activity.shape
        nrows = np.ceil(np.sqrt(shape[1])).astype(int)
        ncols = nrows

        fig, axes = plt.subplots(nrows + 1, ncols, figsize=figsize, squeeze=False)
        for i in range(x.shape[1]):
            axes[0, 1 + i].imshow(1 - x[0][i], cmap='Greys', interpolation='none')
        axes[0, 1].set_title('before', fontsize=30)
        axes[0, 2].set_title('after', fontsize=30)
        axes[0, 3].set_title('diff', fontsize=30)

        for ax in axes.flatten():
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('off')

        for i, (r, c) in enumerate(product(range(nrows), range(ncols))):
            if i >= shape[1]:
                break
            ndim = activity[0][i].ndim
            if ndim != 2:
                raise ValueError("Wrong number of dimensions, image data should "
                                 "have 2, instead got {}".format(ndim))
            axes[r + 1, c].imshow(-activity[0][i], cmap='Greys', interpolation='none')
        
        fig.savefig( os.path.join(odir, ofname) )

    # TO DO:
    # Make it so that RGB can be plotted in one image, instead of three
    def misses(self,args,odir,verbose):
        """ Plots all images for which the 
            predicted class is wrong.
            
        Parameters
        ----------
        args : dictionary, default {}
            Contains the key:value pairs for the
            parameters of this plot:

        #fix

        odir : string
            Relative path to the output directory.
        verbose : integer, default 0
            Verbosity level.
        """
        titles =    args['image labels']    if 'image labels'   in args else ["before", "after", "diff"]
        mode =      args['mode']            if 'mode'           in args else '2d'

        orig_indices = np.array(self.indices_test)
        
        misclassifications = np.array(range(len(self.y_test)))
        misclassifications = misclassifications[self.y_test != self.preds]
        misclassifications_indices = orig_indices[self.y_test != self.preds]
        
        if verbose > 0:
            print("Number of test elements: %i" % len(self.y_test))
            print("Misclassifications: %s" % str(misclassifications_indices))
            print("Plotting misclassifications ...")
            
        for i in xrange(len(misclassifications)):
            index = misclassifications[i]
            orig_index = misclassifications_indices[i]
            
            ofname = os.path.join(odir, str(self.y_test[index]), str(orig_index) + ".png")
            
            ensure_dir(ofname)
            plot_image(self.X_test[index], ofname, titles=titles, mode=mode)         

# TO DO:
# Test this function, add more metrics or ways to add metrics
def assess_performance(preds, y_test, odir, metrics, ofname='results.txt', verbose=False):
    results = {}
    if 'MCC' in metrics:
        results['MCC'] = matthews_corrcoef(y_test, preds)
    if 'accuracy' in metrics:
        results['accuracy'] = accuracy_score(y_test, preds)
    if 'precision' in metrics:
        results['precision'] = precision_score(y_test, preds)
    if 'recall' in metrics:
        results['recall'] = recall_score(y_test, preds)

    with open(os.path.join(odir, ofname), 'a') as result_file:
        for m in metrics:
            result_file.write(m + ": " + str(results[m]))
            if verbose>0:
                print m, ": ", results[m]

# TO DO:
# Replace this with a plotting function
def assess_classification_performance(preds, y_test, odir, plots=None, X_test=None, indices_test=None, preds_proba=None, model=None, verbose=0):
    """ Method for the calculation of performance metrics and
        creating of selected plots.
        
    Parameters
    ----------
    preds : array-like
        Class predictions that the model has made.
    y_test : array-like
        True labels for the test patterns.
    odir : string
        Relative path to the output directory.
    plots : list or None, default None
        List of tuples, the first value being the
        name of the plotting method, the second a
        dictionary of parameters for that method.
        Example:
        plots = [
                    ('confusion', {}),
                    ('roc', {'title': "ROC-curve"}),
                ]   
        Nothing is plotted when set to None.
    X_test : array-like or None, default None
        The test patterns, if necessary for plotting.
    indices_test : array-like or None, default None
        
    preds_proba : array-like
        Prediction probabilities that the model
        has made.
    model : AstroWrapper
        The AstroWrapper model used to make these predictions.

    verbose: integer, default 0
        Verbosity level.
    """

    if plots:
        plot_helper = PlotHelper(plots, preds, preds_proba, y_test, X_test, indices_test, model, odir)
        plot_helper.plot(odir)
