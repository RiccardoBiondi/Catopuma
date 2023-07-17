| **Authors**  | **Project** |  **Build Status** | **License** | **Docs** |
|:------------:|:-----------:|:-----------------:|:-----------:|:--------:|
| [**R. Biondi**](https://github.com/RiccardoBiondi) | **CATOPUMA** | **Windows** : [![Windows CI](https://github.com/RiccardoBiondi/Catopuma/workflows/Windows%20CI/badge.svg)](https://github.com/RiccardoBiondi/Catopuma/actions/workflows/windows_ci.yaml)    <br/> **Ubuntu** : [![Ubuntu CI](https://github.com/RiccardoBiondi/Catopuma/workflows/Ubuntu%20CI/badge.svg)](https://github.com/RiccardoBiondi/Catopuma/actions/workflows/ubuntu_ci.yml)  <br/>   | [![license](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/RiccardoBiondi/Catopuma/blob/master/LICENSE.md) | [![Docs CI](https://github.com/RiccardoBiondi/Catopuma/workflows/Docs%20CI/badge.svg)](https://github.com/RiccardoBiondi/Catopuma/actions/workflows/docs_ci.yaml) |

[![GitHub pull-requests](https://img.shields.io/github/issues-pr/RiccardoBiondi/Catopuma.svg?style=plastic)](https://github.com/RiccardoBiondi/Catopuma/pulls)
[![GitHub issues](https://img.shields.io/github/issues/RiccardoBiondi/Catopuma.svg?style=plastic)](https://github.com/RiccardoBiondi/Catopuma/issues)

[![GitHub stars](https://img.shields.io/github/stars/RiccardoBiondi/Catopuma.svg?label=Stars&style=social)](https://github.com/RiccardoBiondi/Catopuma/stargazers)
[![GitHub watchers](https://img.shields.io/github/watchers/RiccardoBiondi/Catopuma.svg?label=Watch&style=social)](https://github.com/RiccardoBiondi/Catopuma/watchers)
[![GitHub forks](https://img.shields.io/github/watchers/RiccardoBiondi/Catopuma.svg?label=Forks&style=social)](https://github.com/RiccardoBiondi/Catopuma/forks)

# Customizable Advanced Tensorflow Objects for Preoproces, Upload, Model and Augment
 

- [Customizable Advanced Tensorflow Objects for Preoproces, Upload, Model and Augment](#customizable-advanced-tensorflow-objects-for-preoproces-upload-model-and-augment)
  - [Overview](#overview)
  - [Installation](#installation)
    - [Install with pip](#install-with-pip)
    - [Install with conda](#install-with-conda)
    - [Install from source](#install-from-source)
    - [Testing](#testing)
  - [Getting Started](#getting-started)
  - [License](#license)
  - [Contribute](#contribute)
    - [How to Commit](#how-to-commit)
  - [Authors](#authors)
  - [References](#references)
  - [Citation](#citation)


## Overview

CATOPUMA is a Python package that offers customizable advanced TensorFlow objects for tasks such as preprocessing, uploading, modeling, and augmenting. It includes several classes that facilitate the loading, augmentation, and preprocessing of images for deep learning models.

The main functionalities of CATOPUMA are as follows:

  - Feeder: This module contains claasses to helps load images batch-wise from local directories and eventuallly perform data augmentation and image preprocessing on the fly.

  - Losses: The losses module provide a series of custom losses. The overload if the arimetic operators allows to easy combine different losses.

  - Preprocessing: CATOPUMA provides the implementation of some classes to easily perform the basic preprocessing steps on images and labels

  - Core: the core functionality of catopèuma are the bastract classess providing a base for each of the basic functionality (like preprocessin, loss, data augmentation, etc.), allowing an easy customization of the various objects,,

Overall, CATOPUMA simplifies the process of preparing image datasets for deep learning models by providing convenient classes and functions for loading, augmenting, and preprocessing images

## Installation

CATOPUMA offers different installation way.
The installation process is the same for each Operative System

### Install with pip

pip installer is not yet available

### Install with conda

conda installer is no yet availabel

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
