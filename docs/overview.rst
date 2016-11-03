Overview
========

The aim of the astronet package is to provide the astrophysics community with an easy way of implementing neural networks. The large volumes of telescope data allow neural networks to reach a high level of performance on tasks like transient detection and galaxy classification. Even though the potential of neural networks is large, their use is not yet widespread in astronomy, possibly due to their difficulty in implementation and optimization. We aim to provide astronomers with an easy-to-use package to implement neural networks, designed to ease the handling of large data sets. Users can also make use of the on-the-fly data augmentations, which are designed to be particularly useful when used on astrophysical data sets. 


Data augmentations
------------------

To improve the generalization performance of neural networks, one can resort to augmentation of the data. This helps the network learn a better representation of the real world, by slightly modifying input patterns to mimic net input patterns that could also be real. Simple and often effective augmentations are flipping and rotating the inputs. On top of these often-used augmentations, astronet offers a set of augmentations that are specifically created for use with astrophysical data, like adding CCD artifacts to images or adding foreground stars. By simply selecting a pre-defined network and choosing the relevant augmentations, one already has the basis for an effective classifier. 


Batch processing
----------------

Astronomical datasets can be too large to load in memory, even for computing nodes with large amounts of RAM. Where simple implementations of neural networks require the user to load all data into memory at the same time, astronet provides a way of training the network in batches of a more manageable size. The data will be loaded into memory once it is needed, making space for the next batch once it has been fed to the network.
