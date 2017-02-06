from __future__ import division, print_function
import numpy as np
import healpy as hp
from numpy.linalg import lapack_lite
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.io import fits
import copy
import sys
sys.path.insert(0, '../../PolarizationTools')
import basic_functions as polarization_tools


offs_fn = '../data/plot6plus.txt'
with open(offs_fn) as f:
    content = f.readlines()
content = [x.strip() for x in content]

alldata = [x.split() for x in content]

# All but first element of columns
allras = [line[1] for line in alldata]
alldecs = [line[2] for line in alldata]
allras = allras[1:]
alldecs = alldecs[1:]

# index of angle
indx_pang = alldata[0].index('pang')
allpangs = [line[indx_pang] for line in alldata]
allpangs = allpangs[1:]