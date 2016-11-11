""" 
Adding noise to the input
=========================

Example usage of the AddNoise augmentation.
 
"""
import sys
import numpy as np
from scipy.ndimage import imread
import matplotlib.pyplot as plt

from astronet.augmentations import AddNoise

X = np.ones((4,1,50,50))*10
y = np.ones(4)

augment = AddNoise(scale=[0,100])
Xtransformed, _, _ = augment.apply(X, y, None)

fig, ax = plt.subplots(len(X),2, figsize=(4,8), squeeze=False, 
                       subplot_kw={'xticks': [], 'yticks': []})
ax[0][0].set_title("Before")
ax[0][1].set_title("After")
for i in range(len(X)):
    ax[i][0].imshow(X[i][0])
    ax[i][1].imshow(Xtransformed[i][0])
