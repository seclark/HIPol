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
from mpl_toolkits.axes_grid1 import make_axes_locatable
sys.path.insert(0, '../../PolarizationTools')
import basic_functions as polarization_tools
#import seaborn as sns

import sys 
sys.path.insert(0, '../../FITSHandling/code')
import cutouts


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
    fig = plt.figure(figsize=(10, 8), facecolor="white")
    datar = 200
    
    for i, (ra, dec) in enumerate(zip(allras, alldecs)):
        # get x, y points from ra dec
        w = make_wcs(nhidata_fn)
        
        # convert ra, dec to J2000
        #c = SkyCoord(ra=ra*u.hourangle, dec=dec*u.degree, equinox='J1950.0')
        #c1 = c.transform_to(FK5(equinox='J2000.0'))
        #ra = c1.ra.value
        #dec = c1.dec.value
        
        ra, dec = get_source_coordinates(allnames[i])
        
        x_center, y_center = radec_to_xy(ra, dec, w)
        
        x0 = np.int(np.round(x_center - datar))
        y0 = np.int(np.round(y_center - datar))
        x1 = np.int(np.round(x_center + datar))
        y1 = np.int(np.round(y_center + datar))
        
        rastart, decstart = xy_to_radec(x0, y0, w)
        raend, decend = xy_to_radec(x1, y1, w)
        
        ax = fig.add_subplot(nrows, ncols, i+1)
        ax.imshow(intrht[y0:y1, x0:x1], cmap='Greys')#, extent=[rastart, raend, decstart, decend])
        #ax.imshow(nhidata[y0:y1, x0:x1], cmap='Greys')
        ax.set_title(allnames[i])
        ax.set_ylim(0, y1-y0)
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('ra '+str(ra))
        ax.set_ylabel('dec '+str(dec))
        #labels=ax.get_xticks().tolist()
        #print(labels)
        #ax.set_xticklabels(labels)
        
        pang = allpangs[i] + 90 # adjust for quiver coordinate system
        ax.quiver(datar, datar, np.cos(np.radians(pang)), np.sin(np.radians(pang)), headaxislength=0, headlength=0, pivot='mid', color="red")#, scale=(np.max(allppwr)-allppwr[i]))




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
    halpha_fn = "../data/lambda_halpha_fwhm06_0512.fits"
    halpha = hp.fitsfunc.read_map(halpha_fn)
    halpha_hdr = fits.getheader(halpha_fn)
    
    return halpha, halpha_hdr
    
def get_galfa_interpolation_values():
    Gfile = '/Volumes/DataDavy/GALFA/DR2/NHIMaps/GALFA-HI_NHImap_SRcorr_VLSR-090+0090kms.fits'
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
    
    halpha, halpha_hdr = get_halpha_data()
    
    tt, pp = get_galfa_interpolation_values()
    
    halpha_galfa = hp.pixelfunc.get_interp_val(halpha ,pp, tt, nest=False)
    
    return halpha_galfa
    
def plot_alltargets_halpha():
    
    #halpha_galfa = project_halpha_data()
    #halpha_galfa = halpha_galfa.T
    
    halpha_galfa = fits.getdata('/Volumes/DataDavy/Halpha/Halpha_finkbeiner03_proj_on_DR2.fits')
    
    narrownhi_fn = '/Volumes/DataDavy/GALFA/DR2/NHIMaps/GALFA-HI_VLSR-036+0037kms_NHImap_noTcut.fits'
    narrownhi_hdr = fits.getheader(narrownhi_fn)
    
    xax, ra_label = cutouts.get_xlabels_ra(narrownhi_hdr, skip = 1000.0)
    yax, dec_label = cutouts.get_ylabels_dec(narrownhi_hdr, skip = 500.0)
    
    fig = plt.figure(facecolor="white")
    ax = fig.add_subplot(111)
    im = ax.imshow(np.clip(halpha_galfa, 1, 8), cmap='Greys')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    cax.set_ylabel(r'H-$\alpha$ (R)', rotation=360)
    
    alldata = get_alldata()
    allnames = get_data_from_name(alldata, 'name')
    #allcolors = sns.color_palette("spectral", len(alldata))
    symbols = ['+']*10 + ['.']*10
    for i, name in enumerate(allnames):
        sourcex, sourcey = get_source_xy(name, narrownhi_fn)
        ax.plot(sourcex, sourcey, symbols[i], ms=5, label=name)
    
    ax.legend(loc=3, bbox_to_anchor=(0., 1.02, 1., .102), 
       ncol=2, mode="expand", borderaxespad=0.)
    
    ax.set_xticks(xax)
    ax.set_xticklabels(np.round(ra_label))
    ax.set_yticks(yax)
    ax.set_yticklabels(np.round(dec_label))
    ax.set_ylim(yax[0], yax[-1])

    #ax.set_xlim(19000, 21600)
    
    ax.set_xlabel('RA (J2000)')
    ax.set_ylabel('DEC (J2000)')
    
def get_taurus_perseus():
    # Perseus: ra, dec = 3.5486311    31.3358069
    # Taurus: ra, dec = 4.5242677    26.7657750
    
    halpha_galfa = project_halpha_data()
    halpha_galfa = halpha_galfa.T
    
    narrownhi_fn = '/Volumes/DataDavy/GALFA/DR2/NHIMaps/GALFA-HI_VLSR-036+0037kms_NHImap_noTcut.fits'
    narrownhi = fits.getdata(narrownhi_fn)
    nhi_hdr = fits.getheader(narrownhi_fn)
    
    ystart = 1200
    ystop = 2400
    xstart = 20400
    xstop = 21600
    
    # perseus x, y
    #plt.plot(21387.560959878079, 1995.6449357101287, '+', color='pink')

    # taurus x, y
    #plt.plot(21329.022880954239, 1721.4435701128598, '+', color='red')
    
    halpha_cutout_hdr, halpha_cutout = cutouts.xycutout_data(halpha_galfa, nhi_hdr, xstart = xstart, xstop = xstop, ystart = ystart, ystop = ystop)
    gnhi_cutout_hdr, gnhi_cutout = cutouts.xycutout_data(narrownhi, nhi_hdr, xstart = xstart, xstop = xstop, ystart = ystart, ystop = ystop)
    
    fits.writeto('../data/Halpha_cutout_taurus_perseus.fits', halpha_cutout, halpha_cutout_hdr)
    fits.writeto('../data/GALFA-HI_VLSR-036+0037kms_NHImap_noTcut_cutout_taurus_perseus.fits', gnhi_cutout, gnhi_cutout_hdr)
    
    return halpha_cutout_hdr, halpha_cutout, gnhi_cutout_hdr, gnhi_cutout

def get_select_data():
    # An intriguing area exists at [200:1600, 13000:15000] that's mapped in both GALFA and VTSS
    
    halpha_galfa = project_halpha_data()
    halpha_galfa = halpha_galfa.T
    
    narrownhi_fn = '/Volumes/DataDavy/GALFA/DR2/NHIMaps/GALFA-HI_VLSR-036+0037kms_NHImap_noTcut.fits'
    narrownhi = fits.getdata(narrownhi_fn)
    nhi_hdr = fits.getheader(narrownhi_fn)
    
    ystart = 200
    ystop = 1600
    xstart = 13000
    xstop = 15000
    
    halpha_cutout_hdr, halpha_cutout = cutouts.xycutout_data(halpha_galfa, nhi_hdr, xstart = xstart, xstop = xstop, ystart = ystart, ystop = ystop)
    gnhi_cutout_hdr, gnhi_cutout = cutouts.xycutout_data(narrownhi, nhi_hdr, xstart = xstart, xstop = xstop, ystart = ystart, ystop = ystop)
    
    fits.writeto('../data/Halpha_cutout_1.fits', halpha_cutout, halpha_cutout_hdr)
    fits.writeto('../data/GALFA-HI_VLSR-036+0037kms_NHImap_noTcut_cutout_1.fits', gnhi_cutout, gnhi_cutout_hdr)
    
    return halpha_cutout_hdr, halpha_cutout, gnhi_cutout_hdr, gnhi_cutout
    
def get_source_coordinates_old(sourcename, J2000=True):
    alldata = get_alldata()
    allnames = get_data_from_name(alldata, 'name')
    allras = get_data_from_name(alldata, 'ra')
    allras = [np.float(ra) for ra in allras]
    alldecs = get_data_from_name(alldata, 'dec')
    alldecs = [np.float(dec) for dec in alldecs]
    
    indx = allnames.index(sourcename)
    
    ra = allras[indx]
    dec = alldecs[indx]
    
    if J2000 is True:
        # convert ra, dec to J2000
        c = SkyCoord(ra=ra*u.hourangle, dec=dec*u.degree, equinox='J1950.0')
        c1 = c.transform_to(FK5(equinox='J2000.0'))
        ra = c1.ra.degree
        dec = c1.dec.degree 
    
    return ra, dec

def get_source_coordinates(sourcename):
    alldata = get_alldata()
    allnames = get_data_from_name(alldata, 'name')
    allells = get_data_from_name(alldata, 'ell')
    allells = [np.float(ell) for ell in allells]
    allbees = get_data_from_name(alldata, 'bee')
    allbees = [np.float(bee) for bee in allbees]
    
    indx = allnames.index(sourcename)
    
    ell = allells[indx]
    bee = allbees[indx]
    
    c = SkyCoord("galactic", l=ell*u.degree, b=bee*u.degree)
    c_icrs = c.icrs
    
    return c_icrs.ra.degree, c_icrs.dec.degree

def get_source_xy(sourcename, areafn):
    sourcera, sourcedec = get_source_coordinates(sourcename)
    w = make_wcs(areafn)
    sourcex, sourcey = radec_to_xy(sourcera, sourcedec, w)
    
    return sourcex, sourcey
    
def get_source_pang(sourcename):

    alldata = get_alldata()
    allnames = get_data_from_name(alldata, 'name')
    allpangs = get_data_from_name(alldata, 'pang')
    allpangs = [np.float(pang) for pang in allpangs]
    
    indx = allnames.index(sourcename)
    pang = allpangs[indx]
    
    return pang
    
def plot_select_data():
    
    #halpha_cutout_hdr, halpha_cutout, gnhi_cutout_hdr, gnhi_cutout = get_select_data()
    
    #halpha_cutout_fn = '../data/Halpha_cutout_1.fits'
    halpha_cutout_fn = '../data/Halpha_cutout_taurus_perseus.fits'
    halpha_cutout = fits.getdata(halpha_cutout_fn)
    halpha_cutout_hdr = fits.getheader(halpha_cutout_fn)
    
    #gnhi_cutout_fn = '../data/GALFA-HI_VLSR-036+0037kms_NHImap_noTcut_cutout_1.fits'
    gnhi_cutout_fn = '../data/GALFA-HI_VLSR-036+0037kms_NHImap_noTcut_cutout_taurus_perseus.fits'
    gnhi_cutout = fits.getdata(gnhi_cutout_fn)
    
    xax, ra_label = cutouts.get_xlabels_ra(halpha_cutout_hdr, skip = 200.0)
    yax, dec_label = cutouts.get_ylabels_dec(halpha_cutout_hdr, skip = 100.0)
    print(len(dec_label))

    fig = plt.figure(figsize=(6, 4), facecolor = "white")
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    im1 = ax1.imshow(np.clip(halpha_cutout, 0, 10))
    im2 = ax2.imshow(np.clip(gnhi_cutout, 0, 0.8E21))
    
    overplotsource = "3C207"
    sourcera, sourcedec = get_source_coordinates(overplotsource)
    smallregw = make_wcs(halpha_cutout_fn)
    sourcex, sourcey = radec_to_xy(sourcera, sourcedec, smallregw)
    
    allax = [ax1, ax2]
    allim = [im1, im2]
    allunits = ["R", r"cm$^{-2}$"]
    for ax, im, unit in zip(allax, allim, allunits):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        cax.set_ylabel(unit, rotation=360)
    
        ax.set_xticks(xax)
        ax.set_xticklabels(np.round(ra_label))
        ax.set_yticks(yax)
        ax.set_yticklabels(np.round(dec_label))
        ax.set_ylim(yax[0], yax[-1])
        
        ax.set_xlabel('RA')
        ax.set_ylabel('DEC')
        
        ax.plot(sourcex, sourcey, '+', color='white', ms=20)

    ax1.set_title(r'H-$\alpha$')
    ax2.set_title('NHI vlsr -/+36 kms')
    
def get_single_source_cutout(name = "3C207", save=False, cutoutr = 200, returnhdrs=False):

    halpha_galfa = fits.getdata('/Volumes/DataDavy/Halpha/Halpha_finkbeiner03_proj_on_DR2.fits')
    
    #narrownhi_fn = '/Volumes/DataDavy/GALFA/DR2/NHIMaps/GALFA-HI_VLSR-036+0037kms_NHImap_noTcut.fits'
    #narrownhi_fn = '/Volumes/DataDavy/GALFA/DR2/NHIMaps/GALFA-HI_NHImap_SRcorr_VDEV-060+0060kms.fits'
    narrownhi_fn = "/Volumes/DataDavy/GALFA/DR2/FullSkyRHT/new_thetarht_maps/intrht_coadd_974_1069.fits"
    
    narrownhi = fits.getdata(narrownhi_fn)
    nhi_hdr = fits.getheader(narrownhi_fn)
    
    sourcex, sourcey = get_source_xy(name, narrownhi_fn)
    
    # cutout dims
    xstart = np.int(np.round(sourcex - cutoutr))
    ystart = np.int(np.round(sourcey - cutoutr))
    xstop = np.int(np.round(sourcex + cutoutr))
    ystop = np.int(np.round(sourcey + cutoutr))
    
    print(xstart, ystart, xstop, ystop)
    
    halpha_cutout_hdr, halpha_cutout = cutouts.xycutout_data(halpha_galfa, nhi_hdr, xstart = xstart, xstop = xstop, ystart = ystart, ystop = ystop)
    gnhi_cutout_hdr, gnhi_cutout = cutouts.xycutout_data(narrownhi, nhi_hdr, xstart = xstart, xstop = xstop, ystart = ystart, ystop = ystop)
    
    if save:
        fits.writeto('../data/cutouts/Halpha_cutout_'+name+'.fits', halpha_cutout, halpha_cutout_hdr)
        fits.writeto('../data/cutouts/GALFA_HI_narrow_cutout_'+name+'.fits', gnhi_cutout, gnhi_cutout_hdr)
        #fits.writeto('../data/cutouts/RHT_cutout_'+name+'.fits', gnhi_cutout, gnhi_cutout_hdr)
    
    if returnhdrs:
        return halpha_cutout, gnhi_cutout, halpha_cutout_hdr, gnhi_cutout_hdr
     
    else:
        return halpha_cutout, gnhi_cutout
        
def make_cutout_by_radec(ra=27.5, dec=10.5, cutoutr=200, save=True):

    halpha_galfa = fits.getdata('/Volumes/DataDavy/Halpha/Halpha_finkbeiner03_proj_on_DR2.fits')
    #narrownhi_fn = '/Volumes/DataDavy/GALFA/DR2/NHIMaps/GALFA-HI_VLSR-036+0037kms_NHImap_noTcut.fits'
    narrownhi_fn = '/Volumes/DataDavy/GALFA/DR2/NHIMaps/GALFA-HI_NHImap_SRcorr_VDEV-060+0060kms.fits'
    narrownhi = fits.getdata(narrownhi_fn)
    nhi_hdr = fits.getheader(narrownhi_fn)
    
    w = make_wcs(narrownhi_fn)
    sourcex, sourcey = radec_to_xy(27.5, 10.5, w)
    
    # cutout dims
    xstart = np.int(np.round(sourcex - cutoutr))
    ystart = np.int(np.round(sourcey - cutoutr))
    xstop = np.int(np.round(sourcex + cutoutr))
    ystop = np.int(np.round(sourcey + cutoutr))
    
    halpha_cutout_hdr, halpha_cutout = cutouts.xycutout_data(halpha_galfa, nhi_hdr, xstart = xstart, xstop = xstop, ystart = ystart, ystop = ystop)
    gnhi_cutout_hdr, gnhi_cutout = cutouts.xycutout_data(narrownhi, nhi_hdr, xstart = xstart, xstop = xstop, ystart = ystart, ystop = ystop)
    
    if save:
        #fits.writeto('../data/cutouts/Halpha_cutout_VLAFiber.fits', halpha_cutout, halpha_cutout_hdr)
        #fits.writeto('../data/cutouts/GALFA_HI_narrow_cutout_VLAFiber.fits', gnhi_cutout, gnhi_cutout_hdr)
        fits.writeto('../data/cutouts/GALFA_HI_cutout_VLAFiber.fits', gnhi_cutout, gnhi_cutout_hdr)
    
    
def make_all_single_source_cutouts():
    alldata = get_alldata()
    allnames = get_data_from_name(alldata, 'name')
    
    # pop out the ones i've already made
    #allnames.pop(allnames.index('3C409'))
    #allnames.pop(allnames.index('3C207'))
    print(allnames)
    
    for name in allnames:
        halpha_cutout, gnhi_cutout = get_single_source_cutout(name=name, save=True, cutoutr=200)
        
def plot_HI_vs_Halpha_thumbnails():
    
    alldata = get_alldata()
    allnames = get_data_from_name(alldata, 'name')
    
    nrows = 5
    ncols = 4
    fig = plt.figure(figsize=(10, 8), facecolor="white")
    
    for i, name in enumerate(allnames):
        #halpha_cutout, gnhi_cutout = get_single_source_cutout(name=name, save=False, cutoutr=200)
        #chunkfn = '../data/GALFA_HI_cutout_'+name+'.fits'
        #chunkhdr = fits.getheader(chunkfn)
        #sourcex, sourcey = get_source_xy(name, chunkfn)
        halpha_cutout, gnhi_cutout, halpha_cutout_hdr, gnhi_cutout_hdr = get_single_source_cutout(name=name, save=False, cutoutr=200, returnhdrs=True)
        
        sourcex, sourcey = get_source_xy(name, halpha_cutout_hdr)
        
        gnhi_data = np.array((gnhi_cutout/1.0E20).flatten())
        halpha_data = np.array(halpha_cutout.flatten())
        
        ax = fig.add_subplot(nrows, ncols, i+1)
        ax.scatter(gnhi_data, halpha_data, color='black', alpha=0.1, s=1)
        
        ax.set_title(name)
    
def plot_single_source_cutout(name = "3C409", contour='nhi', nonegnhi=True, norm=False):
    
    #halpha_cutout, gnhi_cutout = get_single_source_cutout(name=name, cutoutr = 200, save=True)
    #chunkfn = '../data/cutouts/GALFA_HI_narrow_cutout_'+name+'.fits'
    chunkfn = '../data/cutouts/RHT_cutout_'+name+'.fits'
    gnhi_cutout = fits.getdata(chunkfn)
    
    if norm:
        gnhi_cutout = gnhi_cutout/np.nanmax(gnhi_cutout)
    
    HAchunkfn = '../data/cutouts/Halpha_cutout_'+name+'.fits'
    halpha_cutout = fits.getdata(HAchunkfn)
    
    if nonegnhi:
        gnhi_cutout[np.where(gnhi_cutout <= 0)] = None
    
    ny, nx = gnhi_cutout.shape
    
    chunkhdr = fits.getheader(chunkfn)
    sourcex, sourcey = get_source_xy(name, chunkfn)
    
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.set_ylim(0, ny)
    
    if contour is 'nhi':
        im = ax.imshow(halpha_cutout, cmap='Greys')
    else:
        im = ax.imshow(gnhi_cutout, cmap='Greys')
        
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
    if contour is 'nhi':
        cax.set_ylabel(r'H-$\alpha$ (R)', rotation=360, labelpad=20)
    else:
        #cax.set_ylabel(r'NHI', rotation=360, labelpad=20)
        cax.set_ylabel(r'Relative HI Linear Intensity', rotation=270, labelpad=20)
    
    pang = get_source_pang(name)
    #ax.plot(sourcex, sourcey, '+', color='red', zorder=10)
    pang += 90 # adjust for quiver coordinate system
    ax.quiver(sourcex, sourcey, np.cos(np.radians(pang)), np.sin(np.radians(pang)), headaxislength=0, headlength=0, pivot='mid', color="red", zorder=10)
    
    if contour is 'nhi':
        cs = ax.contour(gnhi_cutout) 
    elif contour is 'halpha':
        cs = ax.contour(halpha_cutout)#, levels=[0.2, 0.4, 0.6, 0.8, 1.0, 1.2])

    #print(cs.levels)
    
    ax.set_title(name)
    
    xax, ra_label = cutouts.get_xlabels_ra(chunkhdr, skip = 100.0)
    yax, dec_label = cutouts.get_ylabels_dec(chunkhdr, skip = 50.0)
    
    ax.set_xticks(xax)
    ax.set_xticklabels(np.round(ra_label))
    ax.set_yticks(yax)
    ax.set_yticklabels(np.round(dec_label))
    ax.set_xlabel('RA (J2000)')
    ax.set_ylabel('DEC (J2000)')
    
def plot_VLA_fiber():
    chunkfn = '../data/cutouts/GALFA_HI_cutout_VLAFiber.fits'
    halphafn = '../data/cutouts/Halpha_cutout_VLAFiber.fits'
    gnhi_cutout = fits.getdata(chunkfn)
    chunkhdr = fits.getheader(chunkfn)
    
    ny, nx = gnhi_cutout.shape

    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.set_ylim(0, ny)
    
    im = ax.imshow(gnhi_cutout, cmap='Greys')
    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    
    ax.set_title("VLA Fiber")
    
    xax, ra_label = cutouts.get_xlabels_ra(chunkhdr, skip = 100.0)
    yax, dec_label = cutouts.get_ylabels_dec(chunkhdr, skip = 50.0)
    
    ax.set_xticks(xax)
    ax.set_xticklabels(np.round(ra_label))
    ax.set_yticks(yax)
    ax.set_yticklabels(np.round(dec_label))
    ax.set_xlabel('RA (J2000)')
    ax.set_ylabel('DEC (J2000)')
    