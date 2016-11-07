""" 
Adding a foreground star
==========

Example usage of the AddStar augmentation.
 
"""
import numpy as np
import matplotlib.pyplot as plt

from astronet.augmentations import AddStar

X = np.random.rand(4,1,50,50)
y = np.ones(4)

augment = AddStar(prob=1, L_range=[100,500], var_range=[1,8])
Xtransformed, _, _ = augment.apply(X, y, None)

fig, ax = plt.subplots(len(X),2, figsize=(3,6), squeeze=False, 
                       subplot_kw={'xticks': [], 'yticks': []})
ax[0][0].set_title("Before")
ax[0][1].set_title("After")
for i in range(len(X)):
    ax[i][0].imshow(X[i][0])
    ax[i][1].imshow(Xtransformed[i][0])
