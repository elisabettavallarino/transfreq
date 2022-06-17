.. _paper:

Reproducing paper results
=========================

The following scripts allow to preprocess the open-source data set used 
in our paper [1]_, and to reproduce the corresponding results.

The data set is downloaded from the OpenNeuro repository, accession number: ds003490 [2]_.

To run the analysis, download all codes in this page  in the same folder. Then add to the 
folder  the requirements.txt file at this 
`link <https://github.com/elisabettavallarino/transfreq/blob/master/examples_paper/requirements.txt>`_.

The results of the paper can be obtained by running the following commands in a terminal.

.. code::

	python3 -m venv transfreq_env
	source transfreq_env/bin/activate
	# In Windows replace the previous command with
	# transfreq_env\Scripts\activate 
	pip install --upgrade pip
	python3 -m pip install -r requirements.txt
	pip install transfreq
	python3 00_fetch_data.py
	python3 01_preprocessing.py
	python3 02_psd.py
	python3 03_compute_transfreq.py
	python3 04_plots.py
 
.. [1] E. Vallarino, S. Sommariva, D. Arnaldi, F. Famà, M. Piana, F. Nobili. Transfreq: a Python package for computing the theta-to-alpha transition frequency from resting state EEG data. Submitted.
.. [2] J.F. Cavanagh, P. Kumar, A.A. Mueller, S.P. Richardson, A. Mueen. Diminished  EEG habituation  to novel  events  effectively  classifies  Parkinson’s patients. Clinical Neurophysiology 129, 409–418 (2018).
