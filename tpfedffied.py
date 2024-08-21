# coding: utf-8

import astropy.constants as astrocon
from astropy.coordinates import SkyCoord
import astropy.io.fits as fits
from astroquery.mast import Tesscut
from bs4 import BeautifulSoup
import juliet
import lightkurve as lk
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import os
from pycheops.utils import lcbin
import requests
from scipy.interpolate import interp1d
from tqdm import tqdm
import urllib.request
import warnings

################################################################################################################

def _make_interp(t,x,scale=None):
    '''
    A function to interpolate data over time via differing scaling methods.
    
    t = time as a 1D array
    x = data as a 1D array
    scale = scaling method as None or string

    returns the interpolated data
    '''
    if scale is None:
        z = x
    elif scale == 'max':
        z = (x-min(x))/np.ptp(x) 
    elif scale == 'range':
        z = (x-np.median(x))/np.ptp(x)
    else:
        raise ValueError('scale must be None, max or range')
        
    return interp1d(t,z,bounds_error=False, fill_value=(z[0],z[-1]))

################################################################################################################

def findFFIdata(name):
    '''
    A function to find TESS Full Frame Image (FFI) data.
    
    name = target name as a string
    '''
    warnings.simplefilter('ignore', category=AstropyWarning)
    try:
        cutout_coord = SkyCoord.from_name(name)
        hdulist = Tesscut.get_cutouts(cutout_coord, 5)
        print(hdulist[0].info())
    except:
        print("No TESS data present for target ",name)

################################################################################################################

def getFFImanifest(name,dxarcmin=7,dyarcmin=7):
    '''
    A function to retrieve TESS Full Frame Image (FFI) data cutouts.
    
    name = target name as a string
    dxarcmin = x-axis width of image cutout in arcminutes as an integer or float
    dyarcmin = y-axis width of image cutout in arcminutes as an integer or float

    returns the TESS Full Frame Image (FFI) data cutout
    '''
    warnings.simplefilter('ignore', category=AstropyWarning)
    try:
        cutout_coord = SkyCoord.from_name(name)
        print('download_cutout information:')
        manifest = Tesscut.download_cutouts(cutout_coord, [dxarcmin,dyarcmin]*u.arcmin)
        print(manifest)
        
        return(manifest)
    except:
        print("No TESS data present for target ",name)

################################################################################################################

def getFFItpf(manifest,slot=0):
    '''
    A function to produce lightkurve Target Pixel Files from TESS Full Frame Image (FFI) data cutouts.
    
    manifest = TESS Full Frame Image (FFI) data cutout as a 2D array cube
    slot = extension of data in manifest as an integer

    returns the lightkurve Target Pixel File
    '''
    max=np.size(manifest['Local Path'])
    try:
        path=manifest['Local Path'][slot]
        tpf=lk.TessTargetPixelFile(path)
        print('Target Pixel File for ',tpf.mission,' sector = ',tpf.sector)
        print('Target pixel file has ',tpf.time.shape[0], ' time samples.')
        print('Target pixel file has',tpf.flux.shape[1],' rows and ',tpf.flux.shape[2],' columns.')
        
        return(tpf)
    except:
        print('Bad slot number. Maximum is ',max-1)

################################################################################################################

def circaper(tpf,radius=3.2,centre=None):
    '''
    A function to produce a circular aperture array for use on lightkurve Target Pixel Files with centering options.
    
    tpf = TESS Target Pixel File as a lightkurve tpf object
    radius = radius of the circular aperture in pixels as an integer or float
    centre = centering method (geometric, photocentre, or background relative) as None or a string

    returns the cicular aperture array
    '''
    aper = np.zeros(tpf.shape[1:], dtype=np.int64)
    nx=tpf.flux.shape[2]
    ny=tpf.flux.shape[1]
    if centre == None:
        xoff = 0
        yoff = 0
        xc = nx/2+xoff-.5
        yc = ny/2+yoff-.5
    elif centre == 'brightest':
        fitsfile = tpf._hdu._file.name
        hdu = fits.open(fitsfile)
        data = hdu[1].data
        flux = data['FLUX']
        meanim = np.mean(flux,axis=0)
        yc,xc = np.where(meanim == np.max(meanim))
    elif centre == 'thresh':
        
        return(tpf.create_threshold_mask(threshold=5, reference_pixel=(ny/2,nx/2)))
    else:
        yoff,xoff = centre
        xc = nx/2+xoff-.5
        yc = ny/2+yoff-.5
        
    y,x = np.ogrid[:ny,:nx]
    r = np.sqrt((x-xc)**2+(y-yc)**2)
    aper = r < radius
    
    return(aper)

################################################################################################################

def get_tpf(method,name,sector,quality_bitmask=7407,diam=10):
    '''
    A function to produce lightkurve Target Pixel Files from long cadence TESS Full Frame Image (FFI) data or short cadence Target Pixel Files (TPF).
    
    method = selection of FFI or TPF data as a string
    name = target name as a string
    sector = TESS sector for which data is to be retreived as an integer
    quality_bitmask = quality bitmask value to control the quality of retrieved TPFs as an integer
    diam = width of image cutout in arcminutes as an integer or float

    returns the lightkurve Target Pixel File
    '''
    if method == 'FFI':
        file_name = 'tess_sector'+str(sector)+'_diam'+str(diam)+'_'+name+'_FFI.fits'
        if os.path.isfile(file_name):
            tpf = lk.TessTargetPixelFile(file_name)
            
            return tpf
        else:
            findFFIdata(name)
            manifest = getFFImanifest(name,dxarcmin=diam,dyarcmin=diam)
            tpf=getFFItpf(manifest,slot=np.argmin([abs(sector-int(manifest[i]['Local Path'][10:12])) for i in np.arange(0,len(manifest))]))
            tpf.to_fits(file_name)
            
            return tpf
    elif method == 'TPF':
        tpf = lk.search_targetpixelfile(name, mission='TESS', sector=sector, exptime='fast').download(quality_bitmask=quality_bitmask)
        if tpf == None:
            tpf = lk.search_targetpixelfile(name, mission='TESS', sector=sector, exptime='short').download(quality_bitmask=quality_bitmask)
            
        return tpf 

################################################################################################################

def get_tpf_lc(tpf,method,T0,P,W,ap_rad=-999.,ap_off=(-999.,-999.),clipsig=3,correction=True):
    '''
    A function to produce lightkurve lightcurve objects from lightkurve Target Pixel Files of long cadence TESS Full Frame Image (FFI) data or short cadence Target Pixel Files (TPF). This uses either a defined aperture radius and offsets or a computed cirular aperture. Detrending corrections can also be applied.
    
    tpf = TESS Target Pixel File as a lightkurve tpf object
    method = selection of FFI or TPF data as a string
    T0 = transit centre time of an transiting planet for use in masking out transits during lightcurve correction as a 1D array of ufloat objects
    P = orbital period of an transiting planet for use in masking out transits during lightcurve correction as a 1D array of ufloat objects
    W = transit width of an transiting planet for use in masking out transits during lightcurve correction as a 1D array of ufloat objects
    ap_rad = radius of the photometric aperture as an integer or float
    ap_off = x- and y-offset of the photometric aperture as a 1D array of integers or floats
    clipsig = sigma value used in outlier rejection of the data as an integer or float
    correction = option to detrend or not the lightcurve as a Boolean

    returns the lightkurve corrected lightcurve object, and corrected time, flux, and flux errors 1D arrays
    '''         
    aper_cen = circaper(tpf,3,(0,0))
    y_cen = np.nanmean(tpf.estimate_centroids(aper_cen)[1]).value-(tpf.row+int(tpf.shape[1]/2))
    x_cen = np.nanmean(tpf.estimate_centroids(aper_cen)[0]).value-(tpf.column+int(tpf.shape[1]/2))
    
    CDPP_best = 1.e6
    for i in np.arange(2,4,0.1):
        aper_t = circaper(tpf,i,(y_cen,x_cen))
        lc_t = tpf.to_lightcurve(aperture_mask=aper_t.astype(bool)).remove_nans().remove_outliers(clipsig)
        if lc_t.estimate_cdpp(transit_duration=30) < CDPP_best:
            CDPP_best = lc_t.estimate_cdpp(transit_duration=30)
            aprad_best = i

    if ap_rad == -999.:
        aprad_best = aprad_best
    else:
        aprad_best = ap_rad
    if ap_off[0] == -999.:
        y_cen = y_cen
    else:
        y_cen = ap_off[0]
    if ap_off[1] == -999.:
        x_cen = x_cen
    else:
        x_cen = ap_off[1]
    
    aper_t = circaper(tpf,aprad_best,(y_cen,x_cen))
    if method == "TPF":
        threshold = 1
    else:
        threshold = 5
    aper_b = ~tpf.create_threshold_mask(threshold=1, reference_pixel=None)

    lc_t,sigmask = tpf.to_lightcurve(aperture_mask=aper_t.astype(bool)).remove_outliers(clipsig,return_mask=True)
    lc_b = tpf.to_lightcurve(aperture_mask=aper_b.astype(bool))[~sigmask]
    
    lc_t.time = lc_t.time + 2457000
    lc_t.flux = lc_t.flux-(np.sum(aper_t)*lc_b.flux/np.sum(aper_b))
    image_frame_no = 500
    aperture_radius = aprad_best             #Adjust based on star size
    aperture_offset = (y_cen,x_cen)          #Adjust based on star location and surroundings

    f, ax1 = plt.subplots(1,1, figsize=(15,10))
    plt.subplots_adjust(wspace=0., hspace=0)

    tpf.plot(ax=ax1,frame=image_frame_no,aperture_mask=aper_t, mask_color='red',show_colorbar=False,title="")
    tpf.plot(ax=ax1,frame=image_frame_no,aperture_mask=(aper_b), mask_color='green',show_colorbar=False,title="")

    ax1.set_xlabel('Pixel Column Number', fontsize=24)
    ax1.set_ylabel('Pixel Row Number', fontsize=24)
    ax1.tick_params(axis='both', labelsize=24)
    ax1.tick_params(axis="x", direction="inout", length=16, width=2, which='major', bottom=True, top=True)
    ax1.tick_params(axis="y", direction="inout", length=10, width=2, which='major', left=True, right=True)
    ax1.tick_params(axis="x", direction="inout", length=8, width=1, which='minor', bottom=True, top=True)
    ax1.tick_params(axis="y", direction="inout", length=5, width=1, which='minor', left=True, right=True)
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())

    lc_t.scatter()

    if np.sum(np.isnan(lc_t.flux_err.value)) == len(lc_t.flux_err.value):
        g = 5.3 ## From TICA headers
        N_a = np.sum(aper_t)
        N_b = np.sum(aper_b)
        N_i = 1
        k = 1

        noise = []
        for index, i in enumerate(tqdm(tpf.hdu[1].data["FLUX"][:,aper_t])):
            S = tpf.hdu[1].data["FLUX"][:,aper_t][index]
            B = np.nanmean(tpf.hdu[1].data["FLUX"][:,aper_b][index])
            sig_B = (np.quantile(tpf.hdu[1].data["FLUX"][:,aper_b][index],[0.50])[0]-np.quantile(tpf.hdu[1].data["FLUX"][:,aper_b][index],[0.16])[0])**2
            noise = np.append(noise, np.sqrt((1/g)*np.sum((S-B)/N_i) + (N_a + k*(N_a**2/N_b))*sig_B))

        lc_t.flux_err = noise[~sigmask]    
    
    if correction:
        
        transitmask = [True]*len(lc_t.time.value)
        if len(T0) > 0:
            for index, i in enumerate(T0):
                phi = (lc_t.time.value-T0[index].n)%P[index].n
                tmask1 = phi < P[index].n-W[index].n/2
                tmask2 = phi > W[index].n/2
                transitmask = transitmask*tmask1*tmask2

        transitmask = transitmask[np.isnan(lc_t.flux_err)==False]
        
        # Make a design matrix
        dm = lk.DesignMatrix(tpf.flux[:, ~aper_t][~sigmask][np.isnan(lc_t.flux_err)==False], name='pixels').pca(5).append_constant()
        lc_t = lc_t[np.isnan(lc_t.flux_err)==False]

        # Regression Corrector Object
        reg = lk.RegressionCorrector(lc_t)
        corrected_lc = reg.correct(dm,cadence_mask=transitmask)

        ax = lc_t.errorbar(label='Raw light curve')
        corrected_lc.errorbar(ax=ax, label='Corrected light curve');

        better = np.where((corrected_lc.flux>(np.nanmean(corrected_lc.flux)-clipsig*np.nanstd(corrected_lc.flux))) & (corrected_lc.flux<(np.nanmean(corrected_lc.flux)+clipsig*np.nanstd(corrected_lc.flux))))[0]
        corrected_lc = corrected_lc[better]
        corrected_lc.scatter()

        cbvCorrector = lk.correctors.CBVCorrector(corrected_lc, interpolate_cbvs=True, extrapolate_cbvs=True)

        # Select which CBVs to use in the correction
        cbv_type = ['MultiScale.1', 'MultiScale.2', 'MultiScale.3', 'Spike', 'SingleScale']
        # Select which CBV indices to use
        cbv_indices = ['ALL','ALL','ALL', 'ALL', 'ALL']
        # Perform the correction
        cbvCorrector.correct(cbv_type=cbv_type, cbv_indices=cbv_indices, target_over_score=0.95,
                             target_under_score=0.94, alpha_bounds=[1.0,10]);

        lc = corrected_lc

    else:
        lc = lc_t

    t = lc.time.value
    f = lc.flux.value/np.nanmedian(lc.flux.value)
    e = lc.flux_err.value/np.nanmedian(lc.flux.value)     

    f = np.array([x for _,x in sorted(zip(t,f))])
    e = np.array([x for _,x in sorted(zip(t,e))])
    t = np.array(sorted(t))

    return lc, t, f, e

################################################################################################################

def get_CBVs_quats_SCALPS(lc,method,t,f,e,name,sector,T0,P,p,b,ecc,omega,Mstar,Rstar,jul_CBV=False,jul_quats=False,photSCALPs=False):
    '''
    A function to produce corrected time, flux, and flux errors 1D arrays from lightkurve lightcurve objects using co-trending basis vectors, quaternions, and/or photometric SCALPELS (Wilson et al. 2022) basis vectors.
    
    tpf = TESS lightcurve as a lightkurve corrected lightcurve object
    method = selection of FFI or TPF data as a string
    t = times as a 1D array
    f = fluxes as a 1D array 
    e = flux errors as a 1D array 
    name = target name as a string
    sector = TESS sector for which data is to be retreived as an integer
    T0 = transit centre time of an transiting planet for use in masking out transits during lightcurve correction as a 1D array of ufloat objects
    P = orbital period of an transiting planet for use in masking out transits during lightcurve correction as a 1D array of ufloat objects
    p = planet-star radius ratio of an transiting planet for use in masking out transits during lightcurve correction as a 1D array of ufloat objects
    b = transit impact parameter of an transiting planet for use in masking out transits during lightcurve correction as a 1D array of ufloat objects
    ecc = orbital eccentricity of an transiting planet for use in masking out transits during lightcurve correction as a 1D array of ufloat objects
    omega = orbital argument of periastron of an transiting planet for use in masking out transits during lightcurve correction as a 1D array of ufloat objects
    Mstar = stellar mass of the host star for use in masking out transits during lightcurve correction as an ufloat object
    Rstar = stellar radius of the host star for use in masking out transits during lightcurve correction as an ufloat object
    jul_CBV = option to detrend or not the lightcurve with co-trending basis vectors as a Boolean
    jul_quats = option to detrend or not the lightcurve with quaternions as a Boolean
    photSCALPs = option to detrend or not the lightcurve with photometric SCALPELS (Wilson et al. 2022) basis vectors as a Boolean

    returns the corrected time, flux, and flux errors 1D arrays
    '''
    instr_BVs = np.array([])
    
    if jul_CBV:        
        for ii in [1,2,3]:
            cbvs = lk.correctors.download_tess_cbvs(sector=lc.sector, camera=lc.camera, ccd=lc.ccd, cbv_type='MultiScale', band=ii)
            cbvs_aligned = cbvs.interpolate(lc, extrapolate=True)
            for jj in np.arange(len(cbvs_aligned.keys())):
                try:
                    dcbv = _make_interp(t, cbvs_aligned['VECTOR_'+str(jj+1)], scale='range')(t)
                    if instr_BVs.size == 0:
                        instr_BVs = np.array(dcbv)
                    else:
                        instr_BVs = np.vstack((instr_BVs,dcbv))
                except:
                    continue

    if jul_quats:

        quats_file = 'tess_sector'+str(sector)+'-quats.fits'
        if os.path.isfile(quats_file):
            print("Quats fits on file:",quats_file)
        else:
            url = 'https://archive.stsci.edu/missions/tess/engineering'
            ext = 'fits'
            page = requests.get(url).text
            soup = BeautifulSoup(page, 'html.parser')
            eng_fits = [url + '/' + node.get('href') for node in soup.find_all('a') if node.get('href').endswith(ext)]
            quats_fits = [i for i in eng_fits if i[-10:] == '-quat.fits']

            if sector < 10:
                quats_url = [i for i in quats_fits if i[-12:-10] == '0'+str(sector)][0]
            else:
                quats_url = [i for i in quats_fits if i[-12:-10] == str(sector)][0]
            urllib.request.urlretrieve(quats_url,quats_file);
            print("Quats fits downloaded:",quats_file)

        quats_data = fits.open(quats_file)[lc.camera].data
        quats_time, quats_q1, quats_q2, quats_q3 = quats_data['TIME']+2457000, quats_data['C'+str(lc.camera)+'_Q1'], quats_data['C'+str(lc.camera)+'_Q2'], quats_data['C'+str(lc.camera)+'_Q3']
        if method == "TPF":
            n_sec = 60
        else:  
            n_sec = 60*15
            
        time_split = np.array([quats_time[i*n_sec:(i+1)*n_sec] for i in range((len(quats_time)+n_sec-1)//n_sec)],dtype=object)
        
        time_avg, q1_avg_temp, q1_std_temp, q2_avg_temp, q2_std_temp, q3_avg_temp, q3_std_temp = [],[],[],[],[],[],[]
        for iindex, ii in enumerate(time_split):
            time_avg = np.append(time_avg,np.nanmean(time_split[iindex]))
            q1_avg_temp = np.append(q1_avg_temp,np.nanmean(quats_q1[iindex*n_sec:(iindex+1)*n_sec]))
            q1_std_temp = np.append(q1_std_temp,np.nanstd(quats_q1[iindex*n_sec:(iindex+1)*n_sec]))
            q2_avg_temp = np.append(q2_avg_temp,np.nanmean(quats_q2[iindex*n_sec:(iindex+1)*n_sec]))
            q2_std_temp = np.append(q2_std_temp,np.nanstd(quats_q2[iindex*n_sec:(iindex+1)*n_sec]))
            q3_avg_temp = np.append(q3_avg_temp,np.nanmean(quats_q3[iindex*n_sec:(iindex+1)*n_sec]))
            q3_std_temp = np.append(q3_std_temp,np.nanstd(quats_q3[iindex*n_sec:(iindex+1)*n_sec]))

        q1_avg, q1_std, q2_avg, q2_std, q3_avg, q3_std  = [],[],[],[],[],[]
        for times in range(len(lc.time.value)):
            tdx = np.argmin(np.abs(time_avg-(lc.time.value[times])))
            q1_avg = np.append(q1_avg,q1_avg_temp[tdx])
            q1_std = np.append(q1_std,q1_std_temp[tdx])
            q2_avg = np.append(q2_avg,q2_avg_temp[tdx])
            q2_std = np.append(q2_std,q2_std_temp[tdx])
            q3_avg = np.append(q3_avg,q3_avg_temp[tdx])
            q3_std = np.append(q3_std,q3_std_temp[tdx])

        if jul_CBV:
            instr_BVs = np.append(instr_BVs,[q1_avg,q1_std,q2_avg,q2_std,q3_avg,q3_std],axis=0)
        else:
            instr_BVs = np.array([q1_avg,q1_std,q2_avg,q2_std,q3_avg,q3_std])
            
    if photSCALPs:
    
        data = np.loadtxt(name+'_'+method+'_S'+str(sector)+'_photSCALPELS_lc_lm.dat',delimiter=",",skiprows=1,dtype=str)

        time = np.array([float(i) for i in np.array(data.T[0])])
        flux = np.array([float(i) for i in np.array(data.T[1])])
        flux_err = np.array([float(i) for i in np.array(data.T[2])])
        photSCALPELS_BVs = data.T[5:-1].astype(float)

        t_temp,f_temp,e_temp = [],[],[]
        if jul_CBV or jul_quats:
            instr_BVs_temp = [[]]*len(instr_BVs)
            
        for times in range(len(time)):
            tdx = np.where(np.abs(t-time[times]) < 1/86400)
            if len(tdx[0]) == 1:
                t_temp = np.append(t_temp,t[tdx[0]][0])
                f_temp = np.append(f_temp,f[tdx[0]][0])
                e_temp = np.append(e_temp,e[tdx[0]][0])
                if jul_CBV or jul_quats:
                    for iindex, ii in enumerate(instr_BVs):
                        instr_BVs_temp[iindex] = np.append(instr_BVs_temp[iindex], instr_BVs[iindex][tdx[0][0]])
            elif len(tdx[0]) > 1:
                t_temp = np.append(t_temp,t[np.argmin(np.abs(t-time[times]))])
                f_temp = np.append(f_temp,f[np.argmin(np.abs(t-time[times]))])
                e_temp = np.append(e_temp,e[np.argmin(np.abs(t-time[times]))])
                if jul_CBV or jul_quats:
                    for iindex, ii in enumerate(instr_BVs):
                        instr_BVs_temp[iindex] = np.append(instr_BVs_temp[iindex], instr_BVs[iindex][np.argmin(np.abs(t-time[times]))])
        t = t_temp
        f = f_temp
        e = e_temp
        if jul_CBV or jul_quats:
            instr_BVs = np.array([instr_BVs_temp[0]])

    lrs = [[]]*len(t)
    for index, i in enumerate(t):
        if jul_CBV or jul_quats:
            if photSCALPs:
                lrs[index] = np.append(instr_BVs.T[index],photSCALPELS_BVs.T[index])
            else:
                lrs[index] = instr_BVs.T[index]
        else:
            lrs[index] = photSCALPELS_BVs.T[index]
    lrs = np.array(lrs)  

    instrument = 'TPFEDFFIED'
    times, fluxes, fluxes_error = {},{},{}
    times[instrument], fluxes[instrument], fluxes_error[instrument] = t, f, e
    linear_regressors = {}
    linear_regressors[instrument] = lrs            
            
    priors = {}
    params = ['q1_'+instrument,'q2_'+instrument,'mdilution_'+instrument,'mflux_'+instrument,'sigma_w_'+instrument]
    dists = ['uniform','uniform','fixed','normal','loguniform']

    hyperps = [[0.,1],[0.,1],1.0,[0,0.1],[0.1,10000]]
    sigma_mult = 3

    if len(P) > 0:
        for h in np.arange(len(P)):
            params.append('P_p'+str(h+1))
            dists.append('normal') 
            hyperps.append([P[h].n,sigma_mult*P[h].s])
            params.append('t0_p'+str(h+1))
            dists.append('normal') 
            hyperps.append([T0[h].n,sigma_mult*T0[h].s])
            params.append('p_p'+str(h+1))
            dists.append('normal') 
            hyperps.append([p[h].n,sigma_mult*p[h].s])
            params.append('b_p'+str(h+1))
            dists.append('normal') 
            hyperps.append([b[h].n,sigma_mult*b[h].s])
            params.append('ecc_p'+str(h+1))
            dists.append('fixed') 
            hyperps.append(ecc[h].n)
            params.append('omega_p'+str(h+1))
            dists.append('fixed') 
            hyperps.append(omega[h].n)

        rhostar = (Mstar*astrocon.M_sun.value)/((4*np.pi/3)*(Rstar*astrocon.R_sun.value)**3)

        params.append('rho')
        dists.append('normal') 
        hyperps.append([rhostar.n,rhostar.s])
    
    for i in np.arange(0,np.shape(lrs)[1]):
        params.append('theta'+str(i)+'_'+instrument)
        dists.append('uniform') 
        hyperps.append([-1.,1.])

    for param, dist, hyperp in zip(params, dists, hyperps):
        priors[param] = {}
        priors[param]['distribution'], priors[param]['hyperparameters'] = dist, hyperp

    outdir_temp = 'TEMP'
    dataset = juliet.load(priors = priors, t_lc = times, y_lc = fluxes, yerr_lc = fluxes_error,
                          linear_regressors_lc = linear_regressors,
                          out_folder = outdir_temp, verbose=False)

    results = dataset.fit(n_live_points=1000, sampler='dynesty')
    
    transit_model, components  = results.lc.evaluate(instrument, return_components = True)
    oot_flux = np.median(1./(1. + results.posteriors['posterior_samples']['mflux_'+instrument]))

    fig, ax1 = plt.subplots(nrows=3,sharex=True,figsize=(15,13))
    plt.subplots_adjust(wspace=0, hspace=0)
    
    oot_flux1 = np.median(1./(1. + (components['lm'])))
    oot_flux2 = oot_flux1-(oot_flux1-np.median(dataset.data_lc[instrument])+np.median(components['lm']))

    ax1[0].errorbar(dataset.times_lc[instrument], dataset.data_lc[instrument], 
                    yerr = dataset.errors_lc[instrument], fmt = '.', ms=15, label="Raw Data")
    ax1[0].plot(dataset.times_lc[instrument], oot_flux2 + components['lm'], 
            color='k', lw = 3, zorder=5, label = "Linear Model")

    ax1[1].errorbar(dataset.times_lc[instrument], dataset.data_lc[instrument]/(components['lm'] + oot_flux2), \
               yerr = dataset.errors_lc[instrument], fmt = '.', ms=15)        
    ax1[1].plot(dataset.times_lc[instrument], transit_model/(components['lm'] + oot_flux2), color='k', lw = 3, zorder=5, label = "Transit Model")

    ax1[2].errorbar(dataset.times_lc[instrument], 
                (dataset.data_lc[instrument]/ (components['lm']+oot_flux2)) - (transit_model/(components['lm'] + oot_flux2)), \
                yerr = dataset.errors_lc[instrument], 
                fmt = '.',  ms=15)
    ax1[2].plot([dataset.times_lc[instrument][0],dataset.times_lc[instrument][-1]], [0,0], color='k', ls="--", lw = 3, zorder=5)
    
    ax1[0].set_ylabel('Normalised Flux', fontsize=20)
    ax1[1].set_ylabel('Detrended Flux', fontsize=20)
    ax1[2].set_ylabel('Residuals', fontsize=20)   
    ax1[2].set_xlabel('Time (BJD - 2457000)', fontsize=20)
    
    
    
    t = dataset.times_lc[instrument]
    f = dataset.data_lc[instrument]/(oot_flux + components['lm'])
    e = dataset.errors_lc[instrument]/(oot_flux)

    f = np.array([x for _,x in sorted(zip(t,f))])
    e = np.array([x for _,x in sorted(zip(t,e))])
    t = np.array(sorted(t))

    os.remove('TEMP/_dynesty_NS_posteriors.pkl')
    os.remove('TEMP/lc.dat')
    os.remove('TEMP/posteriors.dat')
    os.remove('TEMP/priors.dat')
    os.rmdir('TEMP')

    return t,f,e

################################################################################################################
