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

# Local dependencies
import params

inst_map = params.inst_map
param_map = params.params

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

    ssl = params.construct_standard_star_library()
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

def plot_and_save_1d_spectra(reddir, inst, upload=False, group='F4'):
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
            f.write(f'# OBJID {target}\n')
            f.write(f'# OBS_DATE {date_str}\n')
            f.write(f'# INSTRUMENT {inst}\n')
            f.write(f'# OBS_GROUP {group}\n')
            f.write(f'# RA {ra}\n')
            f.write(f'# DEC {dec}\n')
            f.write(f'# GROUPS {group}\n')
            for d in hdu[1].data:
                f.write('%.4f %.4f %.4f \n'%(d[0], d[1], d[2]))

        if upload:
            # Make YSE version
            upload_file = file.replace('.fits','_upload.flm')
            print('Saving spectrum file to',upload_file)
            with open(upload_file, 'w') as f:
                f.write('# wavelength flux\n')
                f.write(f'# SNID {target}\n')
                f.write(f'# OBS_DATE {date_str}\n')
                f.write(f'# INSTRUMENT {inst}\n')
                f.write(f'# OBS_GROUP {group}\n')
                f.write(f'# RA {ra}\n')
                f.write(f'# DEC {dec}\n')
                f.write(f'# GROUPS {group}\n')
                for d in hdu[1].data:
                    f.write('%.4f %.4f \n'%(d[0], d[1]))

if __name__=="__main__":
    reddir = sys.argv[1]
    inst = sys.argv[2]
    run_pypeit_for_dir(reddir, inst)
    handle_1d_spec_files(reddir, inst)
    coadd_1d_files(reddir)
    plot_and_save_1d_spectra(reddir, inst, upload=True)
