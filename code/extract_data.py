from __future__ import division, print_function
import numpy as np
import healpy as hp
from numpy.linalg import lapack_lite
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import FK5
import copy
import sys
from astropy import wcs
import matplotlib
matplotlib.rcParams.update({'figure.autolayout': True})
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
    nhidata = fits.getdata(nhidata_fn)

    alldata = get_alldata()
    allras = get_data_from_name(alldata, 'ra')
    allras = [np.float(ra) for ra in allras]
    alldecs = get_data_from_name(alldata, 'dec')
    alldecs = [np.float(dec) for dec in alldecs]
    allpangs = get_data_from_name(alldata, 'pang')
    allpangs = [np.float(pang) for pang in allpangs]
    allnames = get_data_from_name(alldata, 'name')
    allppwr = get_data_from_name(alldata, 'ppwr')
    allppwr = [np.float(ppwr) for ppwr in allppwr]
    
    allintrht_fn = "/Volumes/DataDavy/GALFA/DR2/FullSkyRHT/new_thetarht_maps/intrht_coadd_974_1069.fits"
    intrht = fits.getdata(allintrht_fn)
    
    nrows = 5
    ncols = 4
    fig = plt.figure(figsize=(12, 10), facecolor="white")
    datar = 200
    
    for i, (ra, dec) in enumerate(zip(allras, alldecs)):
        # get x, y points from ra dec
        w = make_wcs(nhidata_fn)
        
        # convert ra, dec to J2000
        c = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, equinox='J1950.0')
        c1 = c.transform_to(FK5(equinox='J2000.0'))
        ra = c1.ra.value
        dec = c1.dec.value
        
        x_center, y_center = radec_to_xy(ra, dec, w)
        
        x0 = x_center - datar
        y0 = y_center - datar
        x1 = x_center + datar
        y1 = y_center + datar
        
        rastart, decstart = xy_to_radec(x0, y0, w)
        raend, decend = xy_to_radec(x1, y1, w)
        
        ax = fig.add_subplot(nrows, ncols, i+1)
        #ax.imshow(intrht[y0:y1, x0:x1], cmap='Greys')#, extent=[rastart, raend, decstart, decend])
        ax.imshow(nhidata[y0:y1, x0:x1], cmap='Greys')
        ax.set_title(allnames[i])
        ax.set_ylim(0, y1-y0)
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('ra '+str(ra))
        ax.set_ylabel('dec '+str(dec))
        #labels=ax.get_xticks().tolist()
        #print(labels)
        #ax.set_xticklabels(labels)
        
        ax.quiver(datar, datar, np.cos(np.radians(allpangs[i])), np.sin(np.radians(allpangs[i])), headaxislength=0, headlength=0, pivot='mid', color="red")#, scale=(np.max(allppwr)-allppwr[i]))

