import glob
import os
import sys
import shutil
import requests
import numpy as np
import io

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import unique, vstack, Table
from astropy.time import Time
from astropy import units as u

from pypeit.par.util import parse_pypeit_file
from pypeit.spectrographs.util import load_spectrograph
from pypeit.par import PypeItPar
from pypeit.metadata import PypeItMetaData

import matplotlib.pyplot as plt

inst_map = {'gmos_south': 'gemini_gmos_south_ham'}
param_map = {'gmos_south': ['[calibrations]\n',
'    [[biasframe]]\n',
'        [[[process]]]\n',
'            combine = median\n',
'            use_biasimage = False\n',
'            shot_noise = False\n',
'            use_pixelflat = False\n',
'            use_illumflat = False\n',
'    [[darkframe]]\n',
'        [[[process]]]\n',
'            mask_cr = True\n',
'            use_pixelflat = False\n',
'            use_illumflat = False\n',
'    [[arcframe]]\n',
'        [[[process]]]\n',
'            use_pixelflat = False\n',
'            use_illumflat = False\n',
'    [[tiltframe]]\n',
'        [[[process]]]\n',
'            use_pixelflat = False\n',
'            use_illumflat = False\n',
'    [[pixelflatframe]]\n',
'        [[[process]]]\n',
'            combine = median\n',
'            satpix = nothing\n',
'            use_pixelflat = False\n',
'            use_illumflat = False\n',
'    [[alignframe]]\n',
'        [[[process]]]\n',
'            satpix = nothing\n',
'            use_pixelflat = False\n',
'            use_illumflat = False\n',
'    [[traceframe]]\n',
'        [[[process]]]\n',
'            use_pixelflat = False\n',
'            use_illumflat = False\n',
'    [[illumflatframe]]\n',
'        [[[process]]]\n',
'            satpix = nothing\n',
'            use_pixelflat = False\n',
'            use_illumflat = False\n',
'    [[skyframe]]\n',
'        [[[process]]]\n',
'            mask_cr = True\n',
'            lamaxiter = 3\n',
'            sigclip = 3.0\n',
'            noise_floor = 0.01\n',
'    [[standardframe]]\n',
'        [[[process]]]\n',
'            mask_cr = True\n',
'            lamaxiter = 3\n',
'            sigclip = 3.0\n',
'            noise_floor = 0.01\n',
'    [[wavelengths]]\n',
'        method = full_template\n',
'        lamps = CuI, ArI, ArII\n',
'        rms_threshold = 0.4\n',
'        nsnippet = 1\n',
'    [[slitedges]]\n',
'        fit_order = 3\n',
'        bound_detector = True\n',
'    [[tilts]]\n',
'        tracethresh = 10.0\n',
'[scienceframe]\n',
'    [[process]]\n',
'        mask_cr = True\n',
'        noise_floor = 0.01\n',
'[flexure]\n',
'    spec_method = boxcar\n',
'[sensfunc]\n',
'    multi_spec_det = 1, 2, 3\n',
'    algorithm = IR\n',
'    [[IR]]\n',
'        telgridfile = /home/marley/anaconda2/envs/pypeit/lib/python3.8/site-packages/pypeit/data/telluric/atm_grids/TelFit_LasCampanas_3100_26100_R20000.fits\n'],}

def construct_standard_star_library():
    ''' Library of standard stars '''

    ssl = {
        'bd174708': StandardStar(ra='22:11:31.38', dec='+18:05:34.2'),
        'bd262606': StandardStar(ra='14:49:02.36', dec='+25:42:09.1'),
        'bd284211': StandardStar(ra='21:51:11.02', dec='+28:51:50.4'),
        'bd332642': StandardStar(ra='15:51:59.89', dec='+32:56:54.3'),
        'feige15': StandardStar(ra='01:49:09.49', dec='+13:33:11.8'),
        'feige24': StandardStar(ra='02:35:07.59', dec='+03:43:56.8'),
        'feige25': StandardStar(ra='02:38:37.79', dec='+05:28:11.3'),
        'feige34': StandardStar(ra='10:39:36.74', dec='+43:06:09.2'),
        'feige56': StandardStar(ra='12:06:47.24', dec='+11:40:12.7'),
        'feige92': StandardStar(ra='14:11:31.88', dec='+50:07:04.1'),
        'feige98': StandardStar(ra='14:38:15.75', dec='+27:29:32.9'),
        'feige110':StandardStar(ra='23:19:58.4', dec='-05:09:56.2'),
        'g158100': StandardStar(ra='00:33:54', dec='-12:07:57'), ###
        'g191b2b': StandardStar(ra='05:05:30.62', dec='+52:49:51.9'),
        'gd71': StandardStar(ra='05:52:27.62', dec='+15:53:13.2'),
        'gd248': StandardStar(ra='23:26:07', dec='+16:00:21'), ###
        'hd19445': StandardStar(ra='03:08:25.59', dec='+26:19:51.4'),
        'hd84937':StandardStar(ra='09:48:56.1',dec='+13:44:39.3'),
        'hz43':  StandardStar(ra='13:16:21.85', dec='+29:05:55.4'),
        'hz44': StandardStar(ra='13:23:35.26', dec='+36:07:59.5'),
        'ltt0379':StandardStar(ra='18:36:25.95', dec='-44:18:36.9'),
        'ltt1020':StandardStar(ra='01:54:50.27',dec='-27:28:35.7'),
        'ltt1788': StandardStar(ra='03:48:22.61', dec='-39:08:37.0'),
        'ltt2415': StandardStar(ra='05:56:24.74', dec='-27:51:32.4'),
        'ltt3218': StandardStar(ra='08:41:32.43', dec='-32:56:32.9'),
        'ltt3864': StandardStar(ra='10:32:13.62', dec='-35:37:41.7'),
        'ltt4364': StandardStar(ra='11:45:42.92', dec='-64:50:29.5'),
        'ltt6248': StandardStar(ra='15:38:59.648', dec='-28:35:36.97'),
        'ltt7379': StandardStar(ra='18:36:25.950', dec='-44:18:36.90'),
        }
    return ssl


class StandardStar():
    def __init__(self,ra='',dec=''):
        self.coord = SkyCoord('{} {}'.format(ra,dec),
                                frame='icrs',
                                unit=(u.hourangle, u.deg))

# Downloads an essentially complete list (look at SQL query params) of all
# supernovae in the YSE PZ data base
def download_yse_pz_supernovae():
    url = 'https://ziggy.ucolick.org/yse/explorer/45/download?format=csv'
    data = requests.get(url)

    input_data=io.BytesIO(data.text.encode('utf-8')[6:])
    table = Table.read(input_data, format='csv')

    for key in table.keys():
        table.rename_column(key, key.lower())

    for key in table.keys():
        if 'transient_ra' in key: table.rename(key, 'ra')
        if 'transient_dec' in key: table.rename(key, 'dec')

    for key in table.keys():
        if 'name' in key:
            table.rename_column(key, 'name')\

    table['ra'] = table['ra'].astype(float)
    table['dec'] = table['dec'].astype(float)

    return(table)

def pypeit_setup(reddir, inst):
    if inst in inst_map.keys():
        pinst = inst_map[inst]
    else:
        return(None)

    cmd = f'pypeit_setup -s {pinst} -r {reddir} -d {reddir} -c all'
    print(cmd)
    os.system(cmd)

def parse_pypeit_files(reddir, inst):
    if inst in inst_map.keys():
        pinst = inst_map[inst]
    else:
        return(None)

    spectrograph = load_spectrograph(pinst)
    outdata = {}

    pdirs = glob.glob(os.path.join(reddir, pinst+'*'))
    if len(pdirs)>0:
        for pdir in pdirs:
            pfile = glob.glob(os.path.join(pdir, '*.pypeit'))
            if len(pfile)==1:
                usr_cfg_lines, data_files, frametype, usrdata, setups = \
                    parse_pypeit_file(pfile[0])
                par = PypeItPar.from_cfg_lines(cfg_lines=usr_cfg_lines)

                # Create pypeit metadata object
                fitstbl = PypeItMetaData(spectrograph, par=par,
                    files=data_files, usrdata=usrdata)

                outdata[pfile[0]] = fitstbl

    return(outdata)

def merge_bias_data(setup_data):

    # Grab bias frames if they exist from one of the tables
    subtable = None
    for key in sorted(list(setup_data.keys())):
        table = setup_data[key].table
        mask = table['frametype']=='bias'
        if len(table[mask])>0:
            subtable = table[mask]
            break

    if not subtable:
        return(setup_data)

    for key in sorted(list(setup_data.keys())):
        table = setup_data[key].table
        table = unique(vstack([table, subtable]), keys=['filename'])
        setup_data[key].table = table

    return(setup_data)

def write_out_pypeit_files(reddir, inst, setup_data):

    # Delete old output directories
    inst_name = inst_map[inst]
    outdirs = glob.glob(os.path.join(reddir, inst_name+'*'))
    for outdir in outdirs:
        if os.path.exists(outdir):
            print('Deleting',outdir)
            shutil.rmtree(outdir)

    for key in sorted(list(setup_data.keys())):
        # Clobber previous file
        fulloutfile = os.path.join(reddir, key)
        if os.path.exists(fulloutfile):
            os.remove(fulloutfile)

        outname = os.path.split(key)[1].replace('.pypeit','')
        config = outname.split('_')[-1]
        fitstbl = setup_data[key]

        fitstbl.clean_configurations()
        fitstbl.get_frame_types()

        fitstbl.set_configurations(fitstbl.unique_configurations())
        fitstbl.set_calibration_groups()
        fitstbl.set_combination_groups()

        # Need to reset config variable
        cfg_data = fitstbl.configs['A']
        fitstbl.configs = {config: cfg_data}
        fitstbl['setup']=[config]*len(fitstbl)

        fitstbl.write_pypeit(reddir)

def add_params(reddir, inst, setup_data):

    for key in sorted(list(setup_data.keys())):
        pypeit_file = os.path.join(reddir, key)
        if os.path.exists(pypeit_file):
            f = open('tmp', 'w')
            inst_name = inst_map[inst]
            with open(pypeit_file, 'r') as pf:
                for line in pf:
                    if line.strip().replace(' ','')!='spectrograph='+inst_name:
                        f.write(line)
                    else:
                        f.write(line)
                        for param_line in param_map[inst]:
                            f.write(param_line)
            f.close()

            shutil.move('tmp', pypeit_file)

def run_pypeit(reddir, pfile):
    cmd = f'run_pypeit {pfile} -r {reddir}'
    print(cmd)
    os.system(cmd)

def run_pypeit_for_dir(reddir, inst):
    if not os.path.exists(reddir):
        return(None)

    pypeit_setup(reddir, inst)
    setup_data = parse_pypeit_files(reddir, inst)
    setup_data = merge_bias_data(setup_data)
    write_out_pypeit_files(reddir, inst, setup_data)
    add_params(reddir, inst, setup_data)

    for key in sorted(list(setup_data.keys())):
        pypeit_file = os.path.join(reddir, key)
        if os.path.exists(pypeit_file):
            run_pypeit(reddir, pypeit_file)

def is_standard(file):
    hdu = fits.open(file, mode='readonly')
    coord = SkyCoord(hdu[0].header['RA'], hdu[0].header['DEC'], unit='deg')

    ssl = construct_standard_star_library()
    for i,standardKey in enumerate(ssl):
        standard = ssl[standardKey]
        sep = coord.separation(standard.coord)
        if sep < 3.0 * u.arcsec:
            return True

    return False

def generate_sensfunc(file, outfile):
    cmd = f'pypeit_sensfunc {file} -o {outfile}'
    print(cmd)
    os.system(cmd)

def generate_flux_file(reddir, inst, file, caldir):

    # Get the disperser, decker, and angle for file
    hdu = fits.open(file, mode='readonly')
    hdu2 = fits.open(file.replace('spec1d','spec2d'), mode='readonly')

    t = Time(hdu[0].header['MJD'], format='decimalyear')
    date_str = t.datetime.strftime('ut%y%m%d')

    dispname = hdu[0].header['DISPNAME'].split('_')[0]
    dispname = dispname.split('+')[0]
    dispname = dispname.split('-')[0]
    dispname = dispname.lower()

    decker = hdu[0].header['DECKER'].replace('arcsec','')
    decker = decker.replace('.','')

    angle = str(int(hdu2[0].header['CENTWAVE']))

    cal_pattern = f'{inst}.*.{dispname}.{angle}.sens_1.fits'

    # Check potential cal files
    cal_files = glob.glob(os.path.join(caldir, cal_pattern))
    if len(cal_files)>0:
        cal_files = np.array(cal_files)
        dates = [f.split('.')[1].replace('ut','') for f in cal_files]
        dates = np.array([Time({'year':int('20'+d[0:2]),'month':int(d[2:4]),
            'day':int(d[4:6])}, format='ymdhms') for d in dates])
        absdiff = np.array([np.abs(d-t) for d in dates])

        idx = np.argmin(absdiff)

        cal_file = cal_files[idx]

        flux_file = os.path.basename(file).replace('.fits','.flux')
        flux_file = os.path.join(reddir, flux_file)

        with open(flux_file, 'w') as f:
            f.write('[fluxcalib]\n')
            f.write('  extrap_sens = True\n')
            f.write('flux read \n')
            f.write(f'  {file} {cal_file} \n')
            f.write('flux end \n')

        return(flux_file)

    return('')

def group_science_files(reddir):

    files = glob.glob(os.path.join(reddir,'Science/spec1d*.fits'))
    outdata = {}
    for file in files:
        # Don't need to group standards together
        if is_standard(file): continue
        hdu = fits.open(file, mode='readonly')
        target = hdu[0].header['TARGET']
        if target not in outdata.keys():
            outdata[target]=[file]
        else:
            outdata[target].append(file)

    return(outdata)

def generate_coadd1d_file(files, outfile, outname):

    data = []
    for file in files:
        hdu = fits.open(file, mode='readonly')
        for h in hdu:
            if 'XTENSION' not in h.header.keys(): continue
            if h.header['XTENSION']!='BINTABLE': continue
            colnames = [c.name for c in h.columns]
            if h.name.startswith('SPAT') and 'OPT_FLAM' in colnames:
                data.append([file, h.name])

    if len(data)==0:
        return(None)

    with open(outname, 'w') as f:
        f.write('[coadd1d]\n')
        f.write(f'  coaddfile = \'{outfile}\'\n')
        f.write('\n')
        f.write('coadd1d read\n')
        for d in data:
            file = d[0]
            name = d[1]
            f.write(f'    {file} {name}\n')
        f.write('coadd1d end')

def coadd_1d_files(reddir):

    outdata = group_science_files(reddir)
    for key in outdata.keys():
        tmpfile = outdata[key][0]
        hdu = fits.open(tmpfile, mode='readonly')
        # Generate name for outfile
        t = Time(hdu[0].header['MJD'], format='decimalyear')
        date_str = t.datetime.strftime('ut%y%m%d')

        disps = []
        for file in outdata[key]:
            hdu = fits.open(file, mode='readonly')

            dispname = hdu[0].header['DISPNAME'].split('_')[0]
            dispname = dispname.split('+')[0]
            dispname = dispname.split('-')[0]
            dispname = dispname.lower()

            disps.append(dispname)

        if 'b600' in disps and 'r400' in disps:
            dispname = 'both'
        elif 'b600' in disps:
            dispname = 'blue'
        elif 'r400' in disps:
            dispname = 'red'

        if not os.path.exists(os.path.join(reddir, 'Output')):
            os.makedirs(os.path.join(reddir, 'Output'))

        outname = f'{key}.{date_str}.{dispname}.fits'
        outname = os.path.join(reddir, 'Output/'+outname)
        coaddname = outname.replace('.fits','.coadd')
        generate_coadd1d_file(outdata[key], outname, coaddname)

        if os.path.exists(coaddname):
            par = os.path.join(reddir, 'coadd.par')
            cmd = f'pypeit_coadd_1dspec {coaddname}'
            print(cmd)
            os.system(cmd)

def handle_1d_spec_files(reddir, inst, caldir='/data2/Gemini/cal'):

    scidir = os.path.join(reddir, 'Science')
    if os.path.exists(scidir):
        spec1d_files = glob.glob(os.path.join(scidir, 'spec1d*.fits'))

        for file in spec1d_files:
            # Handle stadnards
            if is_standard(file):
                # Generate sensfunc name
                hdu = fits.open(file, mode='readonly')
                # Also need some keywords in the spec2d file
                hdu2 = fits.open(file.replace('spec1d','spec2d'),
                    mode='readonly')

                t = Time(hdu[0].header['MJD'], format='decimalyear')
                date_str = t.datetime.strftime('ut%y%m%d')

                dispname = hdu[0].header['DISPNAME'].split('_')[0]
                dispname = dispname.split('+')[0]
                dispname = dispname.split('-')[0]
                dispname = dispname.lower()

                decker = hdu[0].header['DECKER'].replace('arcsec','')
                decker = decker.replace('.','')

                angle = str(int(hdu2[0].header['CENTWAVE']))

                outname = f'{inst}.{date_str}.{dispname}.{angle}.sens_1.fits'
                if not os.path.exists(caldir):
                    os.makedirs(caldir)

                outfile = os.path.join(caldir, outname)

                if not os.path.exists(outfile):
                    generate_sensfunc(file, outfile)
                else:
                    print(f'sensfunc {outfile} already exists')
            # Assume source is a science exposure and perform fluxing
            else:
                flux_file = generate_flux_file(reddir, inst, file, caldir)
                if os.path.exists(flux_file):
                    par = os.path.join(reddir, 'flux.par')
                    cmd = f'pypeit_flux_calib {flux_file}'
                    print(cmd)
                    os.system(cmd)

def upload_spectrum_to_yse(file, inst, validate=False):
    script = '/home/marley/photpipe/pythonscripts/uploadTransientData.py'
    args = ['-i',file,'--clobber','--obsgroup','YSE',
        '--permissionsgroup','YSE','--instrument',inst,
        '-s','/home/ckilpatrick/scripts/settings.ini','-e','--spectrum']

    cmdlist = [script]+args
    cmd = ' '.join(cmdlist)

    # validate whether spectrum has been successfully uploaded
    vfile = file.replace('.flm','.success')
    if validate:
        if os.path.exists(vfile):
            return(None)

    try:
        print(cmd)
        #r = subprocess.check_call([cmd])
        os.system(cmd)
        if validate:
            with open(vfile, 'w') as f:
                f.write('success')
    except:
        print('YSE PZ upload for',file,'threw an error!')

def crossmatch_to_yse_pz(ra, dec, yse, radius=2.0/3600.0):

    # Crossmatch to YSE_PZ based on input ra/dec and crossmatch radius in deg
    rad = ((yse['ra']-float(ra))*np.cos(yse['dec'] * np.pi/180.0))**2 + (yse['dec']-float(dec))**2
    mask = rad < radius**2

    if len(yse[mask])==1:
        return(yse[mask]['name'][0])
    elif len(yse[mask])>1:
        new_table = yse[mask]
        mask = np.array([n.startswith('20') for n in new_table['name']])
        if len(new_table[mask])==1:
            return(new_table[mask]['name'][0])

def plot_and_save_1d_spectra(reddir, inst, upload=False, yse=None):
    outdir = os.path.join(reddir, 'Output')
    if not os.path.exists(outdir):
        return(None)

    for file in glob.glob(os.path.join(outdir, '*.fits')):
        hdu = fits.open(file)
        wave = [d[0] for d in hdu[1].data]
        flux = [d[1] for d in hdu[1].data]

        minflux = np.percentile(flux, 0.5)
        maxflux = np.percentile(flux, 99.5)
        ran = maxflux - minflux

        limits = [minflux - 0.1 * ran, maxflux + 0.1 * ran]

        plt.clf()

        print('Saving spectrum plot to',file.replace('.fits','.png'))
        plt.plot(wave, flux)
        plt.ylim(limits)
        plt.savefig(file.replace('.fits','.png'))

        target = hdu[0].header['TARGET'].lower()
        if target.startswith('sn') or target.startswith('at'):
            target = target[2:]

        t = Time(hdu[0].header['MJD'], format='decimalyear')
        date_str = t.datetime.strftime('%Y-%m-%d %H:%M:%S')
        ra = str(hdu[0].header['RA'])
        dec = str(hdu[0].header['DEC'])

        if yse:
            new_target = crossmatch_to_yse_pz(ra, dec, yse)
            if new_target:
                print(f'Crossmatched to YSE-PZ target name:',new_target)
                file = file.replace(target, new_target)
                target = new_target

        if inst=='gmos_south':
            inst = 'GMOS-S'
        elif inst=='gmos_north':
            inst = 'GMOS-N'

        print('Saving spectrum file to',file.replace('.fits','.flm'))
        with open(file.replace('.fits','.flm'), 'w') as f:
            f.write('# wavelength flux fluxerr\n')
            f.write(f'# SNID {target}\n')
            f.write(f'# OBS_DATE {date_str}\n')
            f.write(f'# INSTRUMENT {inst}\n')
            f.write(f'# OBS_GROUP YSE\n')
            f.write(f'# RA {ra}\n')
            f.write(f'# DEC {dec}\n')
            f.write('# GROUPS YSE\n')
            for d in hdu[1].data:
                f.write('%.4f %.4f %.4f \n'%(d[0], d[1], d[2]))

        if upload:
            # Make YSE version
            yse_file = file.replace('.fits','_yse.flm')
            print('Saving spectrum file to',yse_file)
            with open(yse_file, 'w') as f:
                f.write('# wavelength flux\n')
                f.write(f'# SNID {target}\n')
                f.write(f'# OBS_DATE {date_str}\n')
                f.write(f'# INSTRUMENT {inst}\n')
                f.write(f'# OBS_GROUP YSE\n')
                f.write(f'# RA {ra}\n')
                f.write(f'# DEC {dec}\n')
                f.write('# GROUPS YSE\n')
                for d in hdu[1].data:
                    f.write('%.4f %.4f \n'%(d[0], d[1]))

            if os.path.exists(yse_file):
                print(f'Uploading YSE file: {yse_file}')
                upload_spectrum_to_yse(yse_file, 'GMOS-S', validate=True)

reddir = sys.argv[1]
inst = sys.argv[2]
yse_supernovae = download_yse_pz_supernovae()
run_pypeit_for_dir(reddir, inst)
handle_1d_spec_files(reddir, inst)
coadd_1d_files(reddir)
plot_and_save_1d_spectra(reddir, inst, upload=True, yse=yse_supernovae)
