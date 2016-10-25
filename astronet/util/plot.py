'''
Created on 26.07.2016

@author: fgieseke
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
        
def plot_image(img, ofname, titles=None, figsize=(10,5), mode="2d"):
    """ Three sub-images given ...
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

def plot_images(imgs, ofnames, titles=None, figsize=(5,5), mode="2d"):
    
    for i in xrange(len(imgs)):
        img = imgs[i]
        plot_image(img, ofnames[i], titles=titles, figsize=figsize, mode=mode)        


class PlotHelper():
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

    def plot(self, odir):
        for pl in self.plots:
            self._methods[pl[0]](pl[1], odir, self.verbose)

    def confusion(self,args,odir,verbose):
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
        pos_label = args['pos label']   if 'pos label'  in args else 1
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
        im_range =  args['image range']     if 'image range'    in args else slice(0,1)
        title =     args['title']           if 'title'          in args else None
        ofname =    args['ofname']          if 'ofname'         in args else "occlusion"
        figsize =   args['figsize']         if 'figsize'        in args else (10,10)
        length =    args['square length']   if 'square length'  in args else 5

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

    def misses(self,args,odir,verbose):
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

def assess_classification_performance(preds, y_test, odir, plots=None, X_test=None, indices_test=None, preds_proba=None, model=None, verbose=0):
    mcc = matthews_corrcoef(y_test, preds)
    acc = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)

    if verbose > 0:
        print("----------------------------------------------------------------")
        print("Matthews Correlation Coefficient: \t" + str(mcc))
        print("Accuracy: \t\t\t\t" + str(acc))
        print("Precision: \t\t\t\t" + str(precision))
        print("Recall: \t\t\t\t" + str(recall))    

    store_results({'Matthews Correlation Coefficient':mcc, 'Accuracy':acc, 'Precision':precision, 'Recall':recall}, os.path.join(odir, "results.txt"))    

    if plots:
        plot_helper = PlotHelper(plots, preds, preds_proba, y_test, X_test, indices_test, model, odir)
        plot_helper.plot(odir)
