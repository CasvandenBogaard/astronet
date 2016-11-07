Augmentations
=============

'astronet' contains many built-in data augmentation methods. Each of these methods
are contained in their own class, only containing their own parameters and an 'apply' method.
The augmentations are shown below. To implement your own method, simply extend the Identity
method and override the 'apply' method to apply the augmentation that you want.

Standard augmentations
----------------------

Cropping
~~~~~~~~
.. autoclass:: astronet.augmentations.Crop
   :members:

Flipping
~~~~~~~~
.. autoclass:: astronet.augmentations.FlipUD
   :members:


.. autoclass:: astronet.augmentations.FlipLR
   :members:


Noise
~~~~~
.. autoclass:: astronet.augmentations.AddNoise
   :members:

.. autoclass:: astronet.augmentations.AddConstant
   :members:

Normalization
~~~~~~~~~~~~~
.. autoclass:: astronet.augmentations.Normalize
   :members:

Rotation
~~~~~~~~
.. autoclass:: astronet.augmentations.Rotate
   :members:

Selection
~~~~~~~~~
.. autoclass:: astronet.augmentations.SelectDimensions
   :members:

Shifting
~~~~~~~~
.. autoclass:: astronet.augmentations.Shift
   :members:

Zooming
~~~~~~~
.. autoclass:: astronet.augmentations.ZoomIn
   :members:

Astronomical augmentations
--------------------------

CCD Errors
~~~~~~~~~~
.. autoclass:: astronet.augmentations.EdgeError
   :members:

.. autoclass:: astronet.augmentations.DeadColumn
   :members:

Adding foreground stars
~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: astronet.augmentations.AddStar
   :members:
