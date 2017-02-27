from __future__ import division, print_function
import numpy as np
import healpy as hp
from numpy.linalg import lapack_lite
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.io import fits, ascii
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

def get_Planck_data(fn = None, Nside = 2048):
    """
    Get TQU and covariance matrix data.
    Currently in Galactic coordinates.
    """
    if fn is None:
        full_planck_fn = "/Volumes/DataDavy/Planck/HFI_SkyMap_353_"+str(Nside)+"_R2.02_full.fits"
    else:
        full_planck_fn = fn
        
    # resolution
    Npix = 12*Nside**2

    # input Planck 353 GHz maps (Galactic)
    # full-mission -- N.B. these maps are already in RING ordering, despite what the header says
    map353Gal = np.zeros((3,Npix)) #T,Q,U
    cov353Gal = np.zeros((3,3,Npix)) #TT,TQ,TU,QQ,QU,UU
    map353Gal[0], map353Gal[1], map353Gal[2], cov353Gal[0,0], cov353Gal[0,1], cov353Gal[0,2], cov353Gal[1,1], cov353Gal[1,2], cov353Gal[2,2], header353Gal = hp.fitsfunc.read_map(full_planck_fn, field=(0,1,2,4,5,6,7,8,9), h=True)

    return map353Gal, cov353Gal

def extract_planck_data():
    alldata = get_alldata()
    allells = get_data_from_name(alldata, 'ell')
    allells = [np.float(ell) for ell in allells]
    allbees = get_data_from_name(alldata, 'bee')
    allbees = [np.float(bee) for bee in allbees]
    allpangs = get_data_from_name(alldata, 'pang')
    allpangs = [np.float(pang) for pang in allpangs]
    allnames = get_data_from_name(alldata, 'name')
    allppwr = get_data_from_name(alldata, 'ppwr')
    allppwr = [np.float(ppwr) for ppwr in allppwr]
    
    nside = 2048
    hpindxs = np.zeros(len(allells))
    allpQs = np.zeros(len(allells))
    allpUs = np.zeros(len(allells))
    allpQeqs = np.zeros(len(allells))
    allpUeqs = np.zeros(len(allells))
    allpangs = np.zeros(len(allells))
    allpQQs = np.zeros(len(allells))
    allpUUs = np.zeros(len(allells))
     
    map353Gal, cov353Gal = get_Planck_data()
        
    planckQ = hp.fitsfunc.read_map("/Volumes/DataDavy/Planck/HFI_SkyMap_353_2048_R2.00_full.fits", field=1)
    planckU = hp.fitsfunc.read_map("/Volumes/DataDavy/Planck/HFI_SkyMap_353_2048_R2.00_full.fits", field=2)
    planckQQ = hp.fitsfunc.read_map("/Volumes/DataDavy/Planck/HFI_SkyMap_353_2048_R2.00_full.fits", field=7)
    planckUU = hp.fitsfunc.read_map("/Volumes/DataDavy/Planck/HFI_SkyMap_353_2048_R2.00_full.fits", field=9)
    
    planckQeq = hp.fitsfunc.read_map("/Volumes/DataDavy/Planck/HFI_SkyMap_353_2048_R2.00_full_Equ.fits", field=1)
    planckUeq = hp.fitsfunc.read_map("/Volumes/DataDavy/Planck/HFI_SkyMap_353_2048_R2.00_full_Equ.fits", field=2)
    planckpang = np.mod(0.5*np.arctan2(planckU, planckQ), np.pi)
    
    for i, (l_, b_) in enumerate(zip(allells, allbees)):
        indx = hp.pixelfunc.ang2pix(nside, l_, b_, lonlat = True)
        hpindxs[i] = np.int(indx)
        allpQs[i] = planckQ[indx]
        allpUs[i] = planckU[indx]
        allpQeqs[i] = planckQeq[indx]
        allpUeqs[i] = planckUeq[indx]
        allpQQs[i] = planckQQ[indx]
        allpUUs[i] = planckUU[indx]
        allpangs[i] = planckpang[indx]

    return hpindxs, allpQs, allpUs, allpQeqs, allpUeqs, allpQQs, allpUUs, allpangs
    
def write_to_txt(hpindxs, allpQs, allpUs, allpQQs, allpUUs, allpangs):
    
    #hpindxs, allpQs, allpUs, allpQeqs, allpUeqs, allpQQs, allpUUs, allpangs = extract_planck_data()
    
    psi_IAU = np.mod(0.5*np.arctan2(-allpUs, allpQs ), np.pi)
    
    table = {'hpindx': hpindxs, 'psi_IAU': psi_IAU, 'galQ_IAU': allpQs, 'galU_IAU': -allpUs, 'galQQ': allpQQs, 'galUU': allpUUs}

    ascii.write(table, '../data/planck353_IAU.dat', formats={'hpindx': '%d'})

def get_halpha_data():
    halpha = hp.fitsfunc.read_map("../data/lambda_halpha_fwhm06_0512.fits")
    
    return halpha
    
def get_galfa_interpolation_values():
    Gfile = '/Volumes/DataDavy/GALFA/DR2/FullSkyWide/GALFA_HI_W_S0900_V-090.9kms.fits'
    ghdu = fits.open(Gfile)
    gwcs = wcs.WCS(Gfile)
    xax = np.linspace(1,ghdu[0].header['NAXIS1'], ghdu[0].header['NAXIS1'] ).reshape(ghdu[0].header['NAXIS1'], 1)
    yax = np.linspace(1,ghdu[0].header['NAXIS2'], ghdu[0].header['NAXIS2'] ).reshape(1,ghdu[0].header['NAXIS2'])
    test = gwcs.all_pix2world(xax, yax, 1)
    RA = test[0]
    Dec = test[1]
    c = SkyCoord(ra=RA*u.degree, dec=Dec*u.degree, frame='icrs')
    cg = c.galactic
    tt = np.asarray(cg.l.rad)
    pp = np.pi/2-np.asarray(cg.b.rad)

    return tt, pp

def project_halpha_data():
    
    halpha = get_halpha_data()
    
    tt, pp = get_galfa_interpolation_values()
    
    halpha_galfa = hp.pixelfunc.get_interp_val(halpha ,pp, tt, nest=True)
    
    return halpha_galfa


