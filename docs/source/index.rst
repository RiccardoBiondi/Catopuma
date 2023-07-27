.. Catopuma documentation master file, created by
   sphinx-quickstart on Thu Mar 17 00:12:16 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Catopuma's documentation!
================================================

**Customizable Advanced Tensorflow Objects for Preoproces, Upload, Model and Augment**

CATOPUMA is a Python package that offers customizable advanced TensorFlow objects for tasks such as preprocessing, uploading, modeling, and augmenting. It includes several classes that facilitate the loading, augmentation, and preprocessing of images for deep learning models.

The main functionalities of CATOPUMA are as follows:

  - Feeder: This module contains classes to helps load images batch-wise from local directories and eventuallly perform data augmentation and image preprocessing on the fly.

  - Losses: The losses module provide a series of custom losses. The overload if the arimetic operators allows to easy combine different losses.

  - Preprocessing: CATOPUMA provides the implementation of some classes to easily perform the basic preprocessing steps on images and labels

  - Augmentation: CATOPUMA provides some classes to easily perform data augmentation.
  These classes are mainly based on albumentations.

  - Core: the core functionality of catopèuma are the bastract classess providing a base for each of the basic functionality (like preprocessin, loss, data augmentation, etc.), allowing an easy customization of the various objects.

Overall, CATOPUMA simplifies the process of preparing image datasets for deep learning models by providing convenient classes and functions for loading, augmenting, and preprocessing images


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   modules
   ./examples/examples


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`