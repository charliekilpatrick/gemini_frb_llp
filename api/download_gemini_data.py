import requests
import sys
import os
import shutil
import tarfile
import glob
import copy
import numpy as np

from astropy.time import Time, TimeDelta
from astropy import units as u

def add_options(parser=None, usage=None):
    import argparse
    if parser == None:
        parser = argparse.ArgumentParser(usage=usage,conflict_handler='resolve')

    # Basic arguments and options
    parser.add_argument('progids', type=str,
        help='Comma-separated list of program IDs.')
    parser.add_argument('--cookie-file','-cf', type=str, default=None,
        help='Path to cookie file for downloading from Gemini archive.')
    parser.add_argument('--date', nargs='+', type=str, default=[],
        help='Either a single date on which to download science data or a '+\
        'range of dates for downloading science data.')
    parser.add_argument('--clobber', default=False, action='store_true',
        help='Clobber files that already exist in output path instead of '+\
        'downloading them again.')
    parser.add_argument('--outdir', type=str, default='.',
        help='Output data to input directory.  Default is current directory.')

    options = parser.parse_args()

    return(options)

def get_observation_data(progid, archive_url='https://archive.gemini.edu/',
    feature='jsonsummary', cookie=None):

    fullurl = os.path.join(archive_url, feature, progid)
    if cookie:
        r = requests.get(fullurl, cookies=cookie)
    else:
        r = requests.get(fullurl)

    if r.status_code==200:
        data = r.json()
        return(data)
    else:
        return([])

def load_cookie(cfile):
    if not os.path.exists(cfile):
        return(None)

    with open(cfile, 'r') as f:
        cookie = f.readline().replace('\n','')
        return(dict(gemini_archive_session=cookie))

def get_full_outname(fileobj, makedirs=True, forcedir='', outdir=''):
    t = Time(fileobj['ut_datetime'])
    basedir = outdir + '/' + t.datetime.strftime('ut%y%m%d/')
    if forcedir:
        basedir = forcedir
    if fileobj['observation_type'].lower()=='object':
        basedir = basedir + 'science/'
    elif fileobj['observation_type'].lower() in ['arc','flat','bias']:
        basedir = basedir + 'cals/'
    if not os.path.exists(basedir) and makedirs:
        print(f'Making: {basedir}')
        os.makedirs(basedir)

    fileobj_name = fileobj['name'].replace('.fits','')
    fileobj_name = fileobj_name.replace('_bias','')
    if fileobj_name.startswith('g'):
        fileobj_name = fileobj_name[1:]

    fullfilename = basedir + fileobj_name + '.fits'
    return(fullfilename)

# Mask the json filelist to only spectral/science/OBJECT observations
def mask_object_spectral_observation(data, date=[]):
    newlist = []
    for fileobj in data:
        if 'mode' not in fileobj.keys(): continue
        if 'observation_type' not in fileobj.keys(): continue
        mode = fileobj['mode'].lower()
        obstype = fileobj['observation_type'].lower()
        obsdate = Time(fileobj['ut_datetime'])

        # Check range of input dates
        if date:
            if len(date)==1:
                t0 = Time(date[0])
                t1 = t0 + TimeDelta(86400.0 * u.s)
            elif len(date)==2:
                t0 = Time(date[0])
                t1 = Time(date[1])
            if obsdate < t0 or obsdate > t1:
                continue

        if ((mode=='spectroscopy' or mode=='ls') and (obstype=='object')):
            newlist.append(fileobj)

    return(newlist)

# Query gemini archive for calibration files associated with the input fileobj
def get_associated_cals(fileobj, archive_url='https://archive.gemini.edu/',
    cookie=None, delta_days=[0.0,0.0], cal_types=['BIAS','FLAT','ARC'],
    caljson={}):
    # get date of observation
    if ('ut_datetime' not in fileobj.keys() or
        'mode' not in fileobj.keys()):
        return([])

    # Need to match detector mode, ROI, binning, slitmask, disperser, camera
    mode=fileobj['mode'].lower()
    roi=fileobj['detector_roi_setting'].lower()
    binning=fileobj['detector_binning'].lower()
    mask=fileobj['focal_plane_mask'].lower()
    disperser=fileobj['disperser'].lower()
    camera=fileobj['camera'].lower()
    cwave=float(fileobj['central_wavelength'])

    feature = 'jsonsummary/'
    cals = []

    for dd in np.arange(delta_days[0], delta_days[1]+1):
        t = Time(fileobj['ut_datetime']) + TimeDelta(dd, format='jd')
        date = t.datetime.strftime('%Y%m%d')

        data = []
        if date in caljson.keys():
            data = caljson[date]
        else:
            url = os.path.join(archive_url, feature, date)
            print(f'Checking {url}')
            r = requests.get(url, cookies=cookie)

            if r.status_code==200:
                data = r.json()
                caljson[date]=data
            else:
                raise Exception(f'ERROR: could not get cal data from Gemini archive.')

        for dat in data:
            # All calibration frames must match these conditions
            if not dat['camera']: continue
            if dat['camera'].lower()!=camera: continue
            if dat['detector_roi_setting'].lower()!=roi: continue
            if dat['detector_binning'].lower()!=binning: continue

            # Get bias frames
            if dat['observation_type']=='BIAS' and 'BIAS' in cal_types:
                cals.append(dat)
                continue

            # Spectrograph setup is important for FLAT, ARC, and standard
            if dat['mode'].lower()!=mode: continue
            if dat['focal_plane_mask'].lower()!=mask: continue
            if dat['disperser'].lower()!=disperser: continue
            if float(dat['central_wavelength'])!=cwave: continue

            # Get flat frames
            if dat['observation_type']=='FLAT' and 'FLAT' in cal_types:
                cals.append(dat)
                continue

            # Get arc frames
            if dat['observation_type']=='ARC' and 'ARC' in cal_types:
                cals.append(dat)
                continue

    return(cals, caljson)

def unpack_tarfile(outtarname):
        basedir = os.path.split(outtarname)[0]
        tar = tarfile.open(outtarname, 'r')
        tar.extractall(basedir)
        tar.close()

        if os.path.exists(basedir+'/md5sums.txt'):
            os.remove(basedir+'/md5sums.txt')
        if os.path.exists(basedir+'/README.txt'):
            os.remove(basedir+'/README.txt')

        # bunzip2 all bz2 files
        for file in glob.glob(basedir + '/*.bz2'):
            os.system('bunzip2 {0}'.format(file))

        # Clean up tar file
        os.remove(outtarname)

def download_file(fileobj, outfilename, archive_url='https://archive.gemini.edu/',
    cookie=None, symlink=''):

    # Color strings for download messages
    green = '\033[1;32;40m'
    red = '\033[1;31;40m'
    end = '\033[0;0m'

    feature = 'download'
    url = os.path.join(archive_url, feature, 'Filename')
    fileobj_name = fileobj['name'].replace('.fits','')
    fileobj_name = fileobj_name.replace('_bias','')
    if fileobj_name.startswith('g'):
        fileobj_name = fileobj_name[1:]

    url = os.path.join(url, fileobj_name)

    if os.path.exists(outfilename):
        print(f'{outfilename} already exists.  Skipping download.')
        return(True)

    message = f'Downloading: {outfilename}'
    sys.stdout.write(message.format(url=url))
    sys.stdout.flush()

    if cookie:
        r = requests.get(url, stream=True, cookies=cookie)
    else:
        r = requests.get(url, stream=True)

    if r.status_code==200:
        basedir = os.path.split(outfilename)[0]
        outtarname = basedir + '/' + fileobj['name'].split('.')[0] + '.tar'

        chunk_size = 256

        with open(outtarname, 'wb') as file:
            for data in r.iter_content(chunk_size):
                file.write(data)

        unpack_tarfile(outtarname)

        if symlink:
            if os.path.exists(outfilename):
                symlinkdir = os.path.split(symlinkname)[0]
                if not os.path.exists(symlinkdir):
                    print(f'\nMaking directory: {symlinkdir}')
                    os.makedirs(symlinkdir)
                os.symlink(outfilename, symlinkname)

        if os.path.exists(outfilename):
            message = '\r' + message
            message += green+' [SUCCESS]'+end+'\n'
            sys.stdout.write(message)
            return(True)
        else:
            message = '\r' + message
            message += red+' [FAILURE]'+end+'\n'
            sys.stdout.write(message)
            return(False)

    message = '\r' + message
    message += red+' [FAILURE]'+end+'\n'
    sys.stdout.write(message)
    return(False)

if __name__=="__main__":

    usage='download_gemini_data.py progids'
    options = add_options(usage=usage)
    programs = options.progids.split(',')
    cookie = load_cookie(options.cookie_file)

    clobber = options.clobber
    outdir = options.outdir
    dates = options.date

    if len(dates)>2:
        raise Exception(f'ERROR: dates should be 0, 1, or 2 arguments.  '+\
            'See download_gemini_date.py -h.')

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    caljson = {}
    for progid in programs:
        data = get_observation_data(progid, cookie=cookie)
        data = mask_object_spectral_observation(data, date=dates)
        for fileobj in data:
            fullfilename = get_full_outname(fileobj, outdir=outdir)
            if os.path.exists(fullfilename) and not clobber:
                print(f'WARNING: {fullfilename} exists.  Continuing...')
                continue
            basedir = os.path.split(fullfilename)[0].replace('science','')
            symlinkname = fullfilename.replace('rawdata/','workspace/')
            symlinkname = symlinkname.replace('science/','')
            check = download_file(fileobj, fullfilename, cookie=cookie,
                symlink=symlinkname)

            print('Checking for cals...')
            cals, caljson = get_associated_cals(fileobj, cookie=cookie,
                caljson=caljson)
            nbias = len([c for c in cals if c['observation_type']=='BIAS'])
            nflat = len([c for c in cals if c['observation_type']=='FLAT'])
            narcs = len([c for c in cals if c['observation_type']=='ARC'])
            delta_days = 1
            # Search for bias and flat 1 day in the future
            cal_types = []
            if nbias < 5: cal_types.append('BIAS')
            if nflat < 5: cal_types.append('FLAT')
            if narcs < 1: cal_types.append('ARC')
            if cal_types:
                print('Checking for additional cals...')
                add_cals, caljson = get_associated_cals(fileobj, cookie=cookie,
                    delta_days=[-4,4], cal_types=cal_types)
                cals.extend(add_cals)

            # Get unique cals
            names = []
            modcals = []
            for c in cals:
                if c['name'] not in names:
                    modcals.append(c) ; names.append(c['name'])
            cals = copy.copy(modcals)

            ncals = len(cals)
            nbias = len([c for c in cals if c['observation_type']=='BIAS'])
            nflat = len([c for c in cals if c['observation_type']=='FLAT'])
            narcs = len([c for c in cals if c['observation_type']=='ARC'])

            m = f'Grabbing {ncals} calibration frames: '
            m += f'{nbias} bias, {nflat} flats, {narcs} arcs'
            print(m)

            for cal in cals:
                fullfilename = get_full_outname(cal, forcedir=basedir,
                    outdir=outdir)
                symlinkname = fullfilename.replace('rawdata/','workspace/')
                symlinkname = symlinkname.replace('cals/','')
                check = download_file(cal, fullfilename, cookie=cookie,
                    symlink=symlinkname)

