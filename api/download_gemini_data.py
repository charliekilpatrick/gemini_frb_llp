import requests
import sys
import os
import shutil
import tarfile
import glob
import copy
import numpy as np

from astropy.time import Time, TimeDelta

programs = ['GS-2021B-LP-204','GS-2022A-LP-204']
cookie_file = '/home/ckilpatrick/scripts/gemini.cookie'
outdir = '/data2/Gemini/rawdata'
clobber = False

archive_url='https://archive.gemini.edu/'

# Color strings for download messages
green = '\033[1;32;40m'
red = '\033[1;31;40m'
end = '\033[0;0m'

def get_observation_data(progid, feature='jsonsummary', cookie=None):
    if cookie:
        r = requests.get(archive_url+feature+'/'+progid, cookies=cookie)
    else:
        r = requests.get(archive_url+feature+'/'+progid)
    if r.status_code==200:
        data = r.json()
        return(data)
    else:
        return([])

def load_cookie(cfile):
    with open(cfile, 'r') as f:
        cookie = f.readline().replace('\n','')
        return(dict(gemini_archive_session=cookie))

def get_full_outname(fileobj, makedirs=True, forcedir=''):
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
def mask_object_spectral_observation(data):
    newlist = []
    for fileobj in data:
        if 'mode' not in fileobj.keys(): continue
        if 'observation_type' not in fileobj.keys(): continue
        mode = fileobj['mode'].lower()
        obstype = fileobj['observation_type'].lower()
        if ((mode=='spectroscopy' or mode=='ls') and (obstype=='object')):
            newlist.append(fileobj)

    return(newlist)

# Query gemini archive for calibration files associated with the input fileobj
def get_associated_cals(fileobj, cookie=None, delta_days=[0.0,0.0],
    cal_types=['BIAS','FLAT','ARC']):
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
        url = archive_url+feature+date
        print(f'Checking {url}')
        r = requests.get(url, cookies=cookie)

        if r.status_code==200:
            data = r.json()
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

    return(cals)

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

def download_file(fileobj, outfilename, cookie=None, symlink=''):

    feature = 'download'
    url = archive_url+feature+'/Filename/'
    fileobj_name = fileobj['name'].replace('.fits','')
    fileobj_name = fileobj_name.replace('_bias','')
    if fileobj_name.startswith('g'):
        fileobj_name = fileobj_name[1:]
    url = url + fileobj_name

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

cookie = load_cookie(cookie_file)

for progid in programs:
    data = get_observation_data(progid, cookie=cookie)
    data = mask_object_spectral_observation(data)
    for fileobj in data:
        fullfilename = get_full_outname(fileobj)
        if os.path.exists(fullfilename) and not clobber:
            print(f'WARNING: {fullfilename} exists.  Continuing...')
            continue
        basedir = os.path.split(fullfilename)[0].replace('science','')
        symlinkname = fullfilename.replace('rawdata/','workspace/')
        symlinkname = symlinkname.replace('science/','')
        check = download_file(fileobj, fullfilename, cookie=cookie,
            symlink=symlinkname)

        print('Checking for cals...')
        cals = get_associated_cals(fileobj, cookie=cookie)
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
            add_cals = get_associated_cals(fileobj, cookie=cookie,
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
            fullfilename = get_full_outname(cal, forcedir=basedir)
            symlinkname = fullfilename.replace('rawdata/','workspace/')
            symlinkname = symlinkname.replace('cals/','')
            check = download_file(cal, fullfilename, cookie=cookie,
                symlink=symlinkname)

