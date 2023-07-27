Installation
============

CATOPUMA offers different installation way.
The installation process is the same for each Operative System.
The tested os are: Windows 10 and Ubuntu.

Supported python version: ![Python version](https://img.shields.io/badge/python-3.8|3.9|3.10|3.11-blue.svg).

Install with pip
----------------

pip installer is not yet available

Install with conda
------------------

conda installer is no yet availabel

Install from source
-------------------

Download the project or the latest release:

.. code-block:: bash

    git clone https://github.com/RiccardoBiondi/Catopuma


Now  install the required packages:

.. code-block:: bash

    python -m pip install -r requirements.txt


And you are ready to build the package:

.. code-block:: bash

      python setup.py develop --user


Testing
-------

We have provide a test routine in test directory. This routine use:

- pytest >= 3.0.7

- hypothesis >= 4.13.0

Please install these packages to perform the test.
You can run the full set of test with:

  .. code-block:: bash

    python -m pytest
