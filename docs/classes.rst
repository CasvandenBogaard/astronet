.. -*- rst -*-


Main Classes
============

AstroWrapper
------------

.. autoclass:: astronet.models.AstroWrapper
   :members:

AstroNet
--------

'AstroNet' is a simple extension of the 'nolearn.lasagne.NeuralNetwork' class, with some built-in networks available. One of the pre-made network is returned by calling AstroNet with a network name, input size and number of outputs. 

.. autoclass:: astronet.models.AstroNet
   :members:

AugmentationBatchIterator
-------------------------

.. autoclass:: astronet.augmentations.AugmentationBatchIterator
   :members:
