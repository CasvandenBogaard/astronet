.. -*- rst -*-


Main Classes
============

AstroWrapper
------------

'AstroWrapper' handles things like training, testing, saving/loading of the weights after supplying it with a network structure and some parameters.

.. autoclass:: astronet.wrappers.AstroWrapper
   :members:

AstroNet
--------

'AstroNet' is a simple extension of the 'nolearn.lasagne.NeuralNetwork' class, with some built-in networks available. One of the pre-made network is returned by calling AstroNet with a network name, input size and number of outputs. 

.. autoclass:: astronet.models.AstroNet
   :members:

AugmentationBatchIterator
-------------------------

The 'AugmentationBatchIterator' (ABI) splits the input up in batches of a predefined size. Every time a batch is used during training, it will first be passed through the 'transform' method in the ABI, which applies the chosen augmentations to the batch, returning the transformed input.

.. autoclass:: astronet.augmentations.AugmentationBatchIterator
   :members:
