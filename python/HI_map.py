####################################################
#
# SCRIPT CURRENTLY SET UP FOR GUGLIELMO
#
####################################################
#
#
# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# ---------------------------------
# 
# # Try 2 #
#
# ## Plan: ##
# 
# - slice out the HI region for each spectrum
# - perform a 3D Gaussian kernel convolution, with the X,Y coordinates
#   set to have the same angular size on the sky, and with the Z size
#   set to be some reasonable width in frequency
#   - do this by calculating the true angular seperation between points
#     and running a 1D Gaussian weighting filter through it.
#   - the end result of the above will be a regular grid in RA, Dec, with spectra
#     smoothed onto that grid.
#   - then see what reasonable smoothing parameter should be used in the spectral dimension
# - see if I can find any HI in there!

# <codecell>

from subprocess import Popen, PIPE
from threading import Thread
from datetime import datetime
import pyfits as pf
import pymongo as pm
from numbapro import autojit
from glob import glob
import re
from astro.coord import ang_sep
import gc
import os
import numpy as np

def remove_fits( location='/fast_scratch/ishivvers/tmp/'):
    '''Deletes fits files from location'''
    o,e = Popen( 'rm {}*.fits'.format(location), shell=True, stdout=PIPE, stderr=PIPE ).communicate()

def run_raw(infile, output='/fast_scratch/ishivvers/tmp/'):
    '''
    Start a subprocess to run and save a whole row worth of .raw files, using 
     the Siemion code.
    Produces a file Row_??\.\d+\.\d+\.fits where ?? is the row number
    '''
    s = 'gpuspec -i {} -b /home/siemion/sw/gbt_seti/lib/bandpass_models/fifteen.bin -c 1 -v -V -s {} -f 1032192 > {}gpuspec.log'.format(infile, output, output)
    # have it print all the output to the terminal, like it should
    Popen( s, shell=True ).communicate()
    return

def get_freqs( fitsfile ):
    '''
    Takes in a Siemion-format fits file location, returns
     an array of frequencies for that file and the table length.
    '''
    hdu = pf.open( fitsfile )
    l = hdu[0].data.shape[-1]
    fs = hdu[0].header['fcntr'] + hdu[0].header['deltaf']*np.linspace(-l/2,l/2-1,l)
    hdu.close()
    return fs, len(hdu)
    
def get_table( fitsfile, itab ):
    '''
    Takes in a Siemion-format fits file location and a table number;
     gives back the data from that row with center-of-band spikes
     removed and a tuple of the RA and Dec.
    
    Note that it seems like we need to open and close the fits file and
     cannot simply pull all data from it at once; it is just too big.
     NOTE: hopefully the memmap=True argument fixes this problem!
    '''
    hdu = pf.open( fitsfile, memmap=True )
    d = hdu[itab].data[0,:]
    coords = (hdu[itab].header['ra'], hdu[itab].header['dec'])
    hdu.close()
    # fix up the middle spikes for the relevant table
    c = len(d)/32/2
    s = 2*c
    d[c::s] = (d[c-1::s] + d[c+1::s])/2
    return d, coords

def table_iter( fitsfile ):
    """
    Takes in a Siemion-format fits file location and opens it,
     providing an iterator for the data tables within it.
    Each call to *.next() yields a tuple of (ra, dec)
     and a numpy array of spectra (with central spikes cleaned).
    Example:
     ti = table_iter( 'myfile.fits' )
     for coords, spec in ti:
         DO SOMETHING
    """
    hdu = pf.open( fitsfile, memmap=True )
    for itab in range(len(hdu)):
        if not itab%500:
            # need to close the file and run the garbage collector occasionally,
            #  else we run out of file handlers! (internals of numpy)
            hdu.close()
            gc.collect()
            hdu = pf.open( fitsfile, memmap=True )
        coords = (hdu[itab].header['ra'], hdu[itab].header['dec'])
        d = hdu[itab].data[0,:]
        # clean the center spikes
        c = len(d)/32/2
        s = 2*c
        d[c::s] = (d[c-1::s] + d[c+1::s])/2
        yield coords, d
    hdu.close()

def find_peaks( d ):
    std = np.std( d )
    mean = np.mean( d )
    peaks = np.arange(len(d))[ np.abs(d-mean)>(10*std) ]
    return peaks        

def get_Gaussian( x, y ):
    '''
    Auto-fits a Gaussian and returns the central wavelength and integrated flux.
    '''
    G = fit_gaussian( x, y, plot=False )
    totflux = G['A'] * np.abs(G['sigma']) * np.pi**0.5
    return G['mu'], totflux
    

# <codecell>

# CALCULATING AND SAVING THE SPECTRA #


client = pm.MongoClient( )
DB = client.himap
## Make sure to run "ensureIndex({coords:"2dsphere"})" in mongo shell on the db.himap collection!

# set the local storage for each machine, and then move the results
#  together when ready to build the final map
# local_storage = '/fast_scratch/ishivvers/storage/'
local_storage = '/mydisks/a/users/ishivvers/storage/'
# tempdir holds the .fits files
# tempdir = '/fast_scratch/ishivvers/tmp/'
tempdir = '/mydisks/a/users/ishivvers/tmp/'
# have a logfile for the gpu and the cpu
# gpulog = '/export/home/spec-hpc-01/ishivvers/working/gpulog.txt'
gpulog = '/mydisks/a/users/ishivvers/gpulog.txt'
# cpulog = '/export/home/spec-hpc-01/ishivvers/working/cpulog.txt'
cpulog = '/mydisks/a/users/ishivvers/cpulog.txt'

def run_guppi( rawfile ):
    """
    a wrapper around run_raw, for use with get_spectra
    """
    with open(gpulog,'a') as logfile:
        logfile.write( '\n\n'+'*'*40+'\n'+str(datetime.now())+'\n\n')
        logfile.write( 'starting %s\n' %rawfile )
    
    if test_row_complete( rawfile ):
        with open(gpulog,'a') as logfile:
            logfile.write( '%s has already been run!\n' %rawfile )
        return
    
    rowstr = re.search('Row_\d\d', rawfile).group()
    donefiles = glob( tempdir + '*.fits')
    if any( [rowstr in f for f in donefiles] ):
        with open(gpulog,'a') as logfile:
            logfile.write( 'using already-created fits file\n' )
    else:
        with open(gpulog,'a') as logfile:
            logfile.write( 'running raw file\n' )
        run_raw( rawfile, output=tempdir )
    # record in database
    entry = { "rawfile" : rawfile }
    DB.log.update( entry, {"$set": entry}, upsert=True )

def deal_with_spectra( rawfile, freqw, freqc ):
    """
    for use with get_spectra
    """
    rowstr = re.search('Row_\d\d', rawfile).group()
    fitsfile = glob( tempdir + '{}*.fits'.format(rowstr) )[0]
    with open(cpulog,'a') as logfile:
        logfile.write( '\n\n'+'*'*40+'\n'+str(datetime.now())+'\n\n')
        logfile.write( 'opening and processing output file: %s\n' %fitsfile )
    
    fs, nhdus = get_freqs( fitsfile )
    
    mask = (fs>(freqc-freqw))&(fs<(freqc+freqw))
    with open(cpulog,'a') as logfile:
        logfile.write( 'running all tables\n' )
    
    # save frequency array
    freqs_fn = os.path.split(rawfile)[1].strip('.raw') + '_freqs.npy'
    np.save( local_storage + freqs_fn, fs[mask] )
    
    data = table_iter( fitsfile )
    # go through, save spectra, and insert everything into the database
    for i, row in enumerate(data):
        c, d = row
        with open(cpulog,'a') as logfile:
            logfile.write( '%d -- ra: %.6f\n -- dec: %.6f\n' %(i, c[0],c[1]) )
        
        # if we've already run this one, move on
        res = DB.log.find_one( {'rawfile':rawfile} )
        if (res.get('tables') != None) and (i in res.get('tables')):
            with open(cpulog,'a') as logfile:
                logfile.write( 'already ran table %d\n' %i )
            continue
        
        # save the spectrum, and then insert a reference to it into the database
        spec_fn = os.path.split(rawfile)[1].strip('.raw') + '_spec_%d.npy' %i
        np.save( local_storage + spec_fn, d[mask] )
        
        # the mongoDB lat/long definitions are just slightly different
        #  than RA, Dec, but normal RA,Dec queries map onto the correct
        #  coordinates (so you only need to worry about it during an insert).
        if c[0] > 180:
            newra = c[0]-360.0
        else:
            newra = c[0]
        entry = { "coords":{"type":"Point",
                            "coordinates":[newra, c[1]]},
                  "ra":c[0],
                  "dec":c[1],
                  "freqsFile":freqs_fn,
                  "specFile":spec_fn }
        DB.himap.insert( entry )
        # record our success in the DB
        DB.log.update( {'rawfile':rawfile}, {'$addToSet':{'tables':i}} )
    # remove file once we're done
    o,e = Popen( 'rm %s' %fitsfile, shell=True, stdout=PIPE, stderr=PIPE ).communicate()
    
def get_spectra(srchstr='/disks/sting/kepler_disk_*/disk_*/gpu4/*_Row_*.0000.raw',
                freqw=150.0, freqc=1420.40575177, ngpus=1):
    """
    Pull out slices of frequency width [freqw, in km/s] centered around [freqc in MHz]
     from the spectra created from all of the raw files that match
     [srchstr].
    Saves result into host of *.npy savefiles, with references all managed by
     a MongoDB server running on sting
    This function uses threads to run the GPU and CPU simultaneously. 
    """
    
    # convert freqw from km/s to MHz
    freqw = (freqw/3e5)*freqc
    
    rawfiles = glob(srchstr)
    for ir,rawfile in enumerate(rawfiles):
        # start the gpu calculation!
        gpu_thread = Thread( target=run_guppi, args=(rawfile,) )
        gpu_thread.start()
        gpu_thread.join()
        
        # make sure the previous CPU thread has finished before starting another
        if ir != 0:
            cpu_thread.join()
        cpu_thread = Thread( target=deal_with_spectra, args=(rawfile, freqw, freqc) )
        cpu_thread.start()

def test_row_complete( firstRaw, ntables_needed=1500 ):
    """
    Tests whether or not at least most of a row has been run.
    """
    res = DB.log.find_one( {'rawfile':firstRaw} )
    if res == None:
        return False
    elif res.get('tables') == None:
        return False
    elif len(res['tables']) < ntables_needed:
        return False
    else:
        return True
            

# <codecell>

get_spectra()

# <codecell>

# HANDLING THE MAPS #

# @autojit(target='cpu')
# def ang_sep_gauss_weight(ra, dec, ras, decs, sigma):
#     """
#     Calculates and returns the weights applied to values at
#      each of ras, decs to interpolate/smooth with a Gaussian
#      kernel of width sigma onto the gridpoint ra, dec.
#     """
#     a = 1./(sigma * (2.*np.pi)**.5)
#     dists = ang_sep( ra, dec, ras, decs )
#     g = a * np.exp( -0.5 * (dists/sigma)**2 )
#     return g

# @autojit(target='cpu')
# def resample_data( w, rmin=None,rmax=None, dmin=None,dmax=None, rn=10, dn=10):
#     """
#     Uses true-sky angular distances to resample the spectra
#      onto a regular grid in RA and Dec, smoothing with a Gaussian kernel.
#      ALL SPECTRA MUST BE THE SAME LENGTH.
#     w: Gaussian width (in degrees)
#     rmin,rmax: RA min and max; data limits if not given
#     dmin,dmax: Dec min and max; data limits if not given
#     rn: Size of array in RA dimension
#     dn: Size of array in Dec dimension
#     """
    
#     # build the working arrays
#     if rmin==None:
#         res = DB.himap.aggregate( [ {"$group": {"_id":0,
#                                                 "rmin": {"$min$":"$ra"}
#                                                 }
#                                      } ] )
#         rmin = res["result"][0]["rmin"]
#     if rmax==None:
#         res = DB.himap.aggregate( [ {"$group": {"_id":0,
#                                                 "rmax": {"$max$":"$ra"}
#                                                 }
#                                      } ] )
#         rmax = res["result"][0]["rmax"]
#     if dmin==None:
#         res = DB.himap.aggregate( [ {"$group": {"_id":0,
#                                                 "dmin": {"$min$":"$dec"}
#                                                 }
#                                      } ] )
#         dmin = res["result"][0]["dmin"]
#     if dmax==None:
#         res = DB.himap.aggregate( [ {"$group": {"_id":0,
#                                                 "dmax": {"$max$":"$dec"}
#                                                 }
#                                      } ] )
#         dmax = res["result"][0]["dmax"]
    
#     rvec = np.linspace(rmin, rmax, rn)
#     dvec = np.linspace(dmin, dmax, dn)
#     outspecs = np.zeros( (len(rvec), len(dvec), len(specs[0])) )
    
#     for i,ra in enumerate(rvec):
#         for j,dec in enumerate(dvec):
#             print 'interpolating to %.5f, %.5f' %(ra, dec)
#             # get all spectra within 3*w of this location
#             radius = 3 * np.deg2rad( w )
#             query = { "coords" :
#                       { "$geoWithin":
#                         { "$centerSphere": [ [ra, dec], radius ] }
#                     }}
#             curs = DB.himap.find( query, {"ra":1,"dec":1,"spec_fn":1} )
#             ras, decs, specs = [], [], []
#             for obs in curs:
#                 ras.append( obs['ra'] )
#                 decs.apppend( obs['dec'] )
#                 spec_fn = obs['spec_fn']
#                 specs.append( np.load( spec_fn ) )
#             # calculate the Gaussian weights and apply them
#             weights = ang_sep_gauss_weight(ra, dec, ras, decs, w)
#             outspecs[i,j,:] = np.dot( weights, specs )
    
#     return outspecs
            
# def inspect_sampling( w, rmin=None,rmax=None, dmin=None,dmax=None, rn=10, dn=10 ):
#     """
#     Same arguments as before, but instead of calculating everything just
#      produces a plot showing the resampling scheme overlain on the data.
#     """
#     # build the working arrays
#     if rmin==None:
#         res = DB.himap.aggregate( [ {"$group": {"_id":0,
#                                                 "rmin": {"$min$":"$ra"}
#                                                 }
#                                      } ] )
#         rmin = res["result"][0]["rmin"]
#     if rmax==None:
#         res = DB.himap.aggregate( [ {"$group": {"_id":0,
#                                                 "rmax": {"$max$":"$ra"}
#                                                 }
#                                      } ] )
#         rmax = res["result"][0]["rmax"]
#     if dmin==None:
#         res = DB.himap.aggregate( [ {"$group": {"_id":0,
#                                                 "dmin": {"$min$":"$dec"}
#                                                 }
#                                      } ] )
#         dmin = res["result"][0]["dmin"]
#     if dmax==None:
#         res = DB.himap.aggregate( [ {"$group": {"_id":0,
#                                                 "dmax": {"$max$":"$dec"}
#                                                 }
#                                      } ] )
#         dmax = res["result"][0]["dmax"]
#     # get all the ras,decs observed
#     curs = DB.himap.find( {}, {"ra":1, "dec":1} )
#     ras,decs = [],[]
#     for obs in curs:
#         ras.append( obs['ra'] )
#         decs.append( obs['dec'] )
    
#     plt.scatter( ras, decs, marker='o', c='b', label='data' )
    
#     rvec = np.linspace(rmin, rmax, rn)
#     dvec = np.linspace(dmin, dmax, dn)
#     RV,DV = np.meshgrid(rvec,dvec)
#     RV = RV.flatten()
#     DV = DV.flatten()
#     plt.scatter( RV, DV, marker='+', c='k', label='grid' )
    
#     plt.vlines( np.mean(rvec), np.mean(dvec)-w/2, np.mean(dvec)+w/2, lw=2, c='k' )
#     plt.hlines( np.mean(dvec), np.mean(rvec)-w/2, np.mean(rvec)+w/2, lw=2, c='k',
#                label='kernel' )
    
#     plt.legend(loc='best')
#     plt.show()

