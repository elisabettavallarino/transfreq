import pkg_resources
import os


def read_sample_datapath(fname='data/transfreq_sample_evoked.fif'):

    _temp = pkg_resources.resource_stream(__name__, fname)
    datapath = os.path.dirname(_temp.name)

    return datapath
