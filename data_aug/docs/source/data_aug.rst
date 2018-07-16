data_aug.data_aug
=================

.. currentmodule:: data_aug.data_aug

data_aug contains common image and bounding boxes transforms. They can be chained together using :class:`Sequence`

.. autoclass:: Sequence

Randomised Transforms
---------------------


.. autoclass:: RandomHorizontalFlip

.. autoclass:: RandomRotate

.. autoclass:: RandomScale

.. autoclass:: RandomScaleTranslate

.. autoclass:: RandomShear

.. autoclass:: RandomTranslate

Non-Randomised Transforms
-------------------------

.. autoclass:: HorizontalFlip

.. autoclass:: Rotate

.. autoclass:: Scale

.. autoclass:: Shear

.. autoclass:: Translate

Input Preprocessing based Transforms
------------------------------------

.. autoclass:: YoloResize


