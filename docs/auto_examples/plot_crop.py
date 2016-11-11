""" 
Cropping the input
==================

Example usage of the Crop augmentation. Please note that the image seems to be zoomed in, but this is because matplotlib scales the images in a subplot to be of equal size.
 
"""
import sys
import numpy as np
from scipy.ndimage import imread
import matplotlib.pyplot as plt

from astronet.augmentations import Crop

X = np.array([imread('moon.jpg').transpose(2,0,1)])
y = np.ones(1)

augment = Crop(x_range=(10,40), y_range=(10,40))
Xtransformed, _, _ = augment.apply(X, y, None)

fig, ax = plt.subplots(len(X),2, figsize=(4,2), squeeze=False, 
                       subplot_kw={'xticks': [], 'yticks': []})
ax[0][0].set_title("Before")
ax[0][1].set_title("After")
for i in range(len(X)):
    ax[i][0].imshow(X[i].transpose(1,2,0), cmap=plt.get_cmap("gray"))
    ax[i][1].imshow(Xtransformed[i].transpose(1,2,0), cmap=plt.get_cmap("gray"))
