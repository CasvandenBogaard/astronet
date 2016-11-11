""" 
Input normalization
===================

Example usage of the Normalize augmentation. 
 
"""

import numpy as np
import matplotlib.pyplot as plt

from astronet.augmentations import Normalize

x1 = np.random.rand(1,50,50)*9
x2 = np.random.rand(1,50,50)*2
x3 = np.random.rand(1,50,50)*5
x4 = np.random.rand(1,50,50)*1
X = np.array([x1,x2,x3,x4])
y = np.ones(4)

augment = Normalize()
Xtransformed, _, _ = augment.apply(X, y, None)

fig, ax = plt.subplots(len(X),2, figsize=(3,6), subplot_kw={'xticks': [], 'yticks': []})
ax[0][0].set_title("Before")
ax[0][1].set_title("After")
for i in range(len(X)):
    ax[i][0].imshow(X[i][0], vmin=0, vmax=10)
    ax[i][1].imshow(Xtransformed[i][0], vmin=0, vmax=10)
