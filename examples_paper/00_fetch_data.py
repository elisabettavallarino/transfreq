# -*- coding: utf-8 -*-
"""
Part 1: Fetch data
==================
Download to OpenNeuro dataset ds003490.
"""

import openneuro as on
import os

cwd = os.getcwd()
data_root = os.path.join(cwd, 'transfreq_data')

if not os.path.exists(data_root):
    os.mkdir(data_root)
else:
    raise Exception('{} already exists. Please check')

on.download(dataset='ds003490', target_dir=data_root)
