data_aug.data_aug
=================

.. currentmodule:: data_aug.data_aug

data_aug contains common image and bounding boxes transforms. They can be chained together using :class:`Sequence`

.. autoclass:: Sequence

Transforms Not Involving change in Bounding Boxes
-------------------------------------------------
These transforms should be applied before the ones involving change in bounding boxes

.. autoclass:: RandomHSV


Transforms Involving Change in Bounding Boxes
---------------------------------------------


.. autoclass:: RandomHorizontalFlip

.. autoclass:: RandomRotate

.. autoclass:: RandomScale

.. autoclass:: RandomScaleTranslate

.. autoclass:: RandomShear

.. autoclass:: RandomTranslate

.. autoclass:: HorizontalFlip

.. autoclass:: Rotate

.. autoclass:: Scale

.. autoclass:: Shear

.. autoclass:: Translate

Input Preprocessing based Transforms
------------------------------------

.. autoclass:: YoloResize


