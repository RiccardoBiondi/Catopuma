![](logo_scritta.svg)



| **Authors**  | **Project** |  **Tensorflow** | **PyTorch** |**License** | **Docs** |
|:------------:|:-----------:|:---------------:|:-----------:|:----------:|:--------:|
| [**R. Biondi**](https://github.com/RiccardoBiondi) | **CATOPUMA** | **Windows** : [![Windows CI](https://github.com/RiccardoBiondi/Catopuma/workflows/Windows%20CI%20TF/badge.svg)](https://github.com/RiccardoBiondi/Catopuma/actions/workflows/windows_ci_tf.yml)    <br/> **Ubuntu** : [![Ubuntu CI](https://github.com/RiccardoBiondi/Catopuma/workflows/Ubuntu%20CI%20TF/badge.svg)](https://github.com/RiccardoBiondi/Catopuma/actions/workflows/ubuntu_ci_tf.yml)  <br/>   | **Windows** : [![Windows CI](https://github.com/RiccardoBiondi/Catopuma/workflows/Windows%20CI%20TC/badge.svg)](https://github.com/RiccardoBiondi/Catopuma/actions/workflows/windows_ci_tc.yml)    <br/> **Ubuntu** : [![Ubuntu CI](https://github.com/RiccardoBiondi/Catopuma/workflows/Ubuntu%20CI%20TC/badge.svg)](https://github.com/RiccardoBiondi/Catopuma/actions/workflows/ubuntu_ci_tc.yml)  <br/> | [![license](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/RiccardoBiondi/Catopuma/blob/master/LICENSE.md) | [![Docs CI](https://github.com/RiccardoBiondi/Catopuma/workflows/Docs%20CI/badge.svg)](https://github.com/RiccardoBiondi/Catopuma/actions/workflows/docs_ci.yaml) <br/> [![Documentation Status](https://readthedocs.org/projects/catopuma/badge/?version=latest)](https://catopuma.readthedocs.io/en/latest/?badge=latest)|

[![GitHub pull-requests](https://img.shields.io/github/issues-pr/RiccardoBiondi/Catopuma.svg?style=plastic)](https://github.com/RiccardoBiondi/Catopuma/pulls)
[![GitHub issues](https://img.shields.io/github/issues/RiccardoBiondi/Catopuma.svg?style=plastic)](https://github.com/RiccardoBiondi/Catopuma/issues)

[![GitHub stars](https://img.shields.io/github/stars/RiccardoBiondi/Catopuma.svg?label=Stars&style=social)](https://github.com/RiccardoBiondi/Catopuma/stargazers)
[![GitHub watchers](https://img.shields.io/github/watchers/RiccardoBiondi/Catopuma.svg?label=Watch&style=social)](https://github.com/RiccardoBiondi/Catopuma/watchers)
[![GitHub forks](https://img.shields.io/github/watchers/RiccardoBiondi/Catopuma.svg?label=Forks&style=social)](https://github.com/RiccardoBiondi/Catopuma/forks)

# Customizable Advanced Tensorflow(Torch) Objects for Preoproces, Upload, Model and Augment
 

- [Customizable Advanced Tensorflow(Torch) Objects for Preoproces, Upload, Model and Augment](#customizable-advanced-tensorflowtorch-objects-for-preoproces-upload-model-and-augment)
  - [Overview](#overview)
  - [Installation](#installation)
    - [Install with pip](#install-with-pip)
    - [Install with conda](#install-with-conda)
    - [Install from source](#install-from-source)
    - [Select Default Framework](#select-default-framework)
    - [Testing](#testing)
  - [Getting Started](#getting-started)
    - [Basic Usage](#basic-usage)
    - [Add Preprocessing and Augmentation](#add-preprocessing-and-augmentation)
    - [Patch Prediction](#patch-prediction)
    - [Patch Prediction](#patch-prediction-1)
    - [More Examples](#more-examples)
  - [License](#license)
  - [Contribute](#contribute)
    - [How to Commit](#how-to-commit)
  - [Authors](#authors)
  - [References](#references)
  - [Citation](#citation)

## Overview

CATOPUMA is a Python package that offers customizable advanced TensorFlow and Torch objects for tasks such as preprocessing, uploading, modeling, and augmenting. It includes several classes that facilitate the loading, augmentation, and preprocessing of images for deep learning models. The vast majprity of these objects are agnostic and can worok both for tensorflow, keras and pytorch.

The main functionalities of CATOPUMA are as follows:

  - **Feeder**: This module contains classes to helps load images batch-wise from local directories and eventuallly perform data augmentation and image preprocessing on the fly.

  - **Losses**: The losses module provide a series of custom losses. The overload if the arimetic operators allows to easy combine different losses toghether. 

  - **Preprocessing**: CATOPUMA provides the implementation of some classes to easily perform the basic preprocessing steps on images and labels.
  Up to now the implemented preprocessing allows voxel normalization and label selection

  - **Augmentation**: CATOPUMA provides some classes to easily perform data augmentation.
 Up to now Data augmentation is supported only for 2D images and its based on Albumentation.

  - **Loaders**: CATOPUMA provides loaders to read images in medical image format (nifti, nrrd, etc.). Moreover allows also a patch based lazy loading.

  - **Prediction**: CATOPUMA provides context manager to easily conduct a sliding patch based prediction.

  - **Core**: the core functionality of catopuma are the bastract classess providing a base for each of the basic functionality (like preprocessin, loss, data augmentation, etc.), allowing an easy customization of the various objects.

Overall, CATOPUMA simplifies the process of preparing image datasets for deep learning models by providing convenient classes and functions for loading, augmenting, and preprocessing images. It is compatible both with tensorflow and pytorch.

## Installation

CATOPUMA offers different installation ways, suitable for each needs.
The installation process is the same for each Operative System and it is checked at each commit using github actions.

Supported python version: ![Python version](https://img.shields.io/badge/python-3.8|3.9|3.10|3.11-blue.svg).

**Note**: CATOPUMA work for both tensorflow.keras and pytorch but does not autoamatically install them.
Before install this project plaese make sure that at least one of them is installed.

All the other dependences are automatically checked and (eventualy) installed; see requirements.txt for the complete list of dependences.


### Install with pip

pip installer is not yet available

### Install with conda

conda installer is no yet available.

### Install from source

  Download the project or the latest release:

  ```console
  git clone https://github.com/RiccardoBiondi/Catopuma
  ```

  Now  install the required packages:

  ```console
  python -m pip install -r requirements.txt
  ```

  And you are ready to build the package:

  ```console
  python setup.py develop --user
  ```


### Select Default Framework

Once you have installed CATOPUMA, you can specify which framework it should be use.
If only one of tensorflow.keras and pytorch is installed, then it is automatically stted as default.
If both are installed, you can specify which one use as deafult by setting the environment variable `CATOPUMA_FRAMEWORK` to `torch` or `tf.keras`
  
- on **Ubuntu**:
```console 
export CATOPUMA_FRAMEWORK=your_framework
```

- on **Windows**:

```console
$env:CATOPUMA_FRAMEWORK = 'your_framwork'
```

### Testing

We have provide a test routine in [test](./test) directory. This routine use:

- pytest >= 3.0.7

- hypothesis >= 4.13.0

Please install these packages to perform the test.
You can run the full set of test with:

```console
  python -m pytest
```

## Getting Started

Once you have installed the 

### Basic Usage


### Add Preprocessing and Augmentation

### Patch Prediction

### Patch Prediction

The default framework is keras (if installed on your system or environment), however it is possible to work also with pytorch or tensorflow.keras.
Moreover, it is possible


### More Examples

More examples are provided in the [documentation](https://catopuma.readthedocs.io/en/latest/?badge=latest).

## License

The `CATOPUMA` package is licensed under the MIT "Expat" License.
[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](<https://github.com/RiccardoBiondi/Catopuma/blob/master/LICENSE.md>)


## Contribute

Any contribution is more than welcome. Just fill an [issue]() or a [pull request]() and we will check ASAP!

### How to Commit

## Authors

- <img src="https://avatars3.githubusercontent.com/u/48323959?s=400&v=4" width="25px"> **Riccardo Biondi** [github](https://github.com/RiccardoBiondi),  [unibo](https://www.unibo.it/sitoweb/riccardo.biondi7)

## References

<blockquote> Iakubovskii, P. Segmentation Models. GitHub repository (2019). https://github.com/qubvel/segmentation_models </blockquote>


<blockquote> https://github.com/GianlucaCarlini/Segmentation3D</blockquote>

## Citation

If you have found `CATOPUMA` helpful in your project

```BibTeX
@misc{catopuma,
  author = {Biondi, Riccardo},
  title = {CATOPUMA - Customizable Advanced Tensorflow Objects to Preprocess, Upload, Model and Augment},
  year = {2023},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/RiccardoBiondi/Catopuma}},
}

```
