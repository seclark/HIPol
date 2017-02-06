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

def get_alldata():
    """
    Read text file into list of lists
    """

    offs_fn = '../data/plot6plus.txt'
    with open(offs_fn) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    alldata = [x.split() for x in content]
    
    return alldata
    
def get_data_from_name(alldata, name):
    """
    slice out data by column name
    """
    
    indx = alldata[0].index(name)
    slicedata = [line[indx] for line in alldata]
    slicedata = slicedata[1:]
    
    return slicedata


def make_wcs(wcs_fn):
    #Set wcs transformation
    w = wcs.WCS(wcs_fn, naxis=2)
    
    return w

def xy_to_radec(x, y, w):
    
    #Transformation
    xy = [[x, y]]
    radec = w.wcs_pix2world(xy, 1)
    
    ra = radec[0,0]
    dec = radec[0,1]
    
    return ra, dec

def radec_to_xy(ra, dec, w):
    
    #Transformation
    radec = [[ra, dec]]
    xy = w.wcs_world2pix(radec, 1)
    
    x = xy[0,0]
    y = xy[0,1]
    
    return x, y
    
def make_circular_mask(hdr, center_x, center_y, radius):
    """
    inputs: hdr :: header to grab data size from
            center_x :: x pixel position of center annulus
            center_y :: y pixel " " "
            radius   :: radius of desired annulus
    """
    yindx, xindx = np.ogrid[-center_y:hdr['NAXIS2'] - center_y, -center_x:hdr['NAXIS1'] - center_x]
    mask = xindx**2 + yindx**2 <= radius**2
    
    return mask
    
def plot_thumbnails():

    nhidata_fn = "/Volumes/DataDavy/GALFA/DR2/NHIMaps/GALFA-HI_VLSR-036+0037kms_NHImap_noTcut.fits"
    galfa_hdr = fits.getheader(nhidata_fn)

    alldata = get_alldata()
    allras = get_data_from_name(alldata, 'ra')
    alldecs = get_data_from_name(alldata, 'dec')
    allpangs = get_data_from_name(alldata, 'pang')
    allnames = get_data_from_name(alldata, 'name')
    print(allnames)

    allintrht_fn = "/Volumes/DataDavy/GALFA/DR2/FullSkyRHT/new_thetarht_maps/intrht_coadd_974_1069.fits"
    intrht = fits.getdata(allintrht_fn)
    
    nrows = 5
    ncols = 4
    fig = plt.figure(facecolor="white")
    datar = 100
    
    for i, (ra, dec) in enumerate(zip(allras, alldecs)):
        # get x, y points from ra dec
        w = make_wcs(nhidata_fn)
        x_center, y_center = radec_to_xy(ra, dec, w)
        
        ax = fig.add_subplot(nrows, ncols, i+1)
        ax.imshow(intrht[y_center-datar:y_center+datar, x_center-datar:x_center+datar])


