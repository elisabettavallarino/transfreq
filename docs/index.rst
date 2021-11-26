.. transfreq documentation master file, created by
   sphinx-quickstart on Thu Sep  2 10:45:04 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

transfreq
=========

Welcome to the documentation of transfreq, a Python3 library for automatically 
computing the individual transition frequency from theta to alpha band. 
transreq only requires in input the power density of EEG data recorded during a 
resting state.  

A detailed description of the algorithm implemented in transfreq can be found
in the origial paper [1]_.

Installation
============

To install the latest stable version of this package use ``pip``:

.. code::

    pip install transfreq

If you do not have admin privileges on the computer, use the ``--user`` flag
with ``pip``. 

To check if everything worked fine, you can run:

.. code::

    python -c 'import sesameeg'

and it should not give any error messages.

Bug reports
===========

To report bugs, please use the `github issue tracker <https://github.com/elisabettavallarino/transfreq/issues>`_ .

Authors of the code
===================
| Sara Sommariva <sommariva@dima.unige.it>,
| Elisabetta Vallarino <vallarino@dima.unige.it>.

Cite our work
=============

If you use this code in your project, please consider citing our work:

.. [1] E. Vallarino, S. Sommariva, D. Arnaldi, F. Famà, M. Piana, F. Nobili. Transfreq: a Python package for computing the theta-to-alpha transition frequency from resting state EEG data. Submitted

.. toctree::
    :hidden:

    api
    auto_examples/index
    auto_paper/index




