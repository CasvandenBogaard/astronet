""" 
Rotate
==========

Example usage of the Rotate augmentation.
 
"""
import sys
import numpy as np
from scipy.ndimage import imread
import matplotlib.pyplot as plt

from astronet.augmentations import Rotate

im = imread('moon.jpg').transpose(2,0,1)
X = np.array([im,im,im,im])
y = np.ones(4)

augment = Rotate()
Xtransformed, _, _ = augment.apply(X, y, None)

fig, ax = plt.subplots(len(X),2, figsize=(3,6), squeeze=False, 
                       subplot_kw={'xticks': [], 'yticks': []})
ax[0][0].set_title("Before")
ax[0][1].set_title("After")
for i in range(len(X)):
    ax[i][0].imshow(X[i].transpose(1,2,0), cmap=plt.get_cmap("gray"))
    ax[i][1].imshow(Xtransformed[i].transpose(1,2,0), cmap=plt.get_cmap("gray"))
