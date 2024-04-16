Usage
=====

Installation
------------

To install the package, you can use the following command:

.. code-block:: console

    (venv) $ pip install https://github.com/michaelbennett99/gendata_package/archive/master.zip

Usage
-----

The package provides functions for loading and merging data, and classes for storing and acting on genetic data.

Functions
~~~~~~~~~

The package provides two functions, one for loading data and one for merging data.

.. autofunction:: gendata.read_bed
    :no-index:

.. autofunction:: gendata.merge
    :no-index:

Classes
~~~~~~~

The package provides two main classes, one for integer 0/1/2 genotypes and one for standardised genotypes.

.. autoclass:: gendata.IntGenoData
    :members:
    :inherited-members:
    :no-index:

.. autoclass:: gendata.StdGenoData
    :members:
    :inherited-members:
    :no-index:
