""" Script to compare images in pairs. One method is simply the ratio while 
the other is a Similarity metrics.
"""

import os
import glob
import socket
import time
import argparse
import logging
import re
from string import ascii_letters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim as ssim
import seaborn
import fitsio

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
)

def comp_ratio(arr1, arr2):
    """ Comparison by simply getting the ratio between thow images. Note this
    method will modify values when the denominator value is zero
    """
    try:
        arr2[np.where(arr2 == 0)] = np.nextafter(0, 1)
        ratio = arr1 / arr2
    except:
        logging.error('Error when calculating ratio')
        ratio = np.nan
    return ratio

def comp_ssim(arr1, arr2, window=3):
    """ Method to compare by calculating the similarity metrics using the 
    Structural Similarity Metrics.
    http://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.
    compare_ssim
    There are more keywords that can be added to the SSIM call. See API
    """
    kw  = {
        'win_size': window,
        'gradient': True,
    }
    mssim, grad = ssim(arr1, arr2, **kw) 
    return mssim, grad

def read_fits(fnm, ext):
    """ Single extension read
    """
    try:
        with fitsio.FITS(fnm) as f:
            x = f[ext].read()
            h = f[ext].read_header()
    except:
        raise
        logging.error('Error reading FITS file {0}, ext={1}'.format(fnm, ext))
        exit()
    return h, x

# -----------------------------------------------------------------------------
# Following 2 functions are used for sorting string with numbers in between
#    
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_key(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]
#
# -----------------------------------------------------------------------------

def aux_main(vlist1=None, vlist2=None, prefix=None, field=None,
             ext1=0, ext2=0):
    """ Auxiliary main for the processing
    """
    # Get the list of file for all of the matching 
    va1, va2 = [glob.glob(x) for x in vlist1]
    vb1, vb2 = [glob.glob(y) for y in vlist2]
    # Sort using the numbers inside the filename    
    va1.sort(key=natural_key)
    va2.sort(key=natural_key)
    vb1.sort(key=natural_key)
    vb2.sort(key=natural_key)
    # Key for ride out of special characters 
    # def make_key(string):
    #      return ''.join(c for c in string if c in set(ascii_letters)).lower()
    # Sort to have same image for comparison
    # va1.sort(key=make_key)
    # va2.sort(key=make_key)
    # vb1.sort(key=make_key)
    # vb2.sort(key=make_key)
    # Checks
    if ((len(va1) != len(va2)) or (len(vb1) != (len(vb2)))):
        logging.error('Regex must generate same amount of files per pair')
        exit()
    # Run the ratio comparison for the first set
    for k in range(len(va1)):
        # Read the files
        h1, x1 = read_fits(va1[k], ext1)
        h2, x2 = read_fits(va2[k], ext1)
        # Apply method
        tmp_ratio = comp_ratio(x1, x2)    
        # Save output, using band for naming
        if (h1['BAND'].strip() != h2['BAND'].strip()):
            t_w = 'BAND is not the same for both files: {0}'.format(h1['BAND'])
            t_w += ' {0}'.format(h2['BAND'])
            logging.warning(t_w)
            aux_band = ''.join(h1['BAND'].strip(), h2['BAND'].strip())
        else: 
            aux_band = h1['BAND'].strip()
        try: 
            outfnm = '{0}_{1}_ratio_c{2:02}.npy'.format(prefix, 
                                                        aux_band, 
                                                        h2['CCDNUM'])
        except:
            logging.info('Naming using iterator instead of CCDNUM')
            outfnm = '{0}_{1}_ratio_id{2:02}.npy'.format(prefix, 
                                                         aux_band, 
                                                         k + 1)
        np.save(outfnm, tmp_ratio)
        logging.info('Saved: {0}'.format(outfnm))
    # Run the SSIM comparison for the second set
    # Notice no border trimming has been applied
    res_val_ssim = []
    aux_band = None
    for j in range(len(vb1)):
        # Read the pair of FITS files
        h1, y1 = read_fits(vb1[j], ext2)
        h2, y2 = read_fits(vb2[j], ext2)
        # Apply SSIM, recovering a matrix of gradient and a single value
        val_ssim, grad_ssim = comp_ssim(y1, y2)
        # Save output, using band for naming the gradient output, and for the
        # SSIM table output
        if (h1['BAND'].strip() != h2['BAND'].strip()):
            t_w = 'BAND is not the same for both files: {0}'.format(h1['BAND'])
            t_w += ' {0}'.format(h2['BAND'])
            logging.warning(t_w)
            aux_band = ''.join(h1['BAND'].strip(), h2['BAND'].strip())
        else: 
            aux_band = h1['BAND'].strip()
        # For storing the gradient 
        try: 
            outnm_grad = '{0}_{1}_grad_c{2:02}.npy'.format(prefix, 
                                                           aux_band, 
                                                           h2['CCDNUM'])
        except:
            raise
            logging.info('Naming using iterator instead of CCDNUM')
            outnm_grad = '{0}_{1}_grad_id{2:02}.npy'.format(prefix, 
                                                            aux_band, 
                                                            j + 1)
        np.save(outnm_grad, grad_ssim)
        logging.info('Saved: {0}'.format(outnm_grad))
        # For storing the one-value statistics, also add the header keywords
        aux_field = [val_ssim]
        for f in field:
            try:
                aux_field.append(h1[f.upper()])
                aux_field.append(h2[f.upper()])
            except:
                logging.warning('Error getting field: {0}'.format(f.upper()))
                aux_field.append(np.nan, np.nan)
        # Store results from this pair
        res_val_ssim.append(aux_field)
    # Output the table containing the one-value SSIM metrics
    col = ['ssim']
    for f in field:
        col.append('{0}_1'.format(f)) 
        col.append('{0}_2'.format(f))
    outnm_ssim = '{0}_{1}_ssim.csv'.format(prefix, aux_band)
    df_out = pd.DataFrame(res_val_ssim, columns=col)
    df_out.to_csv(outnm_ssim, index=False, header=True)
    #
    logging.info('Saved: {0} '.format(outnm_ssim))
    return True

if __name__ == '__main__':
    logging.info('Running in {0}'.format(socket.gethostname()))
    txt0 = 'Script for image comparison using ratio and Similarity metrics'
    txt0 += ' No masking or border trimming is applied so far'
    abc  = argparse.ArgumentParser(description=txt0)
    h1 = 'Regex for files to be used in ratio comparison (use \ for special'
    h1 += ' characters)'
    abc.add_argument('--ratio', help=h1, nargs=2, type=str)
    h2 = 'Regex for files for SSIM analysis (use \ for special characters)'
    abc.add_argument('--ssim', help=h2, nargs=2, type=str)
    pref = 'comp'
    h3 = 'Prefix to be used for output files. Default: {0}'.format(pref)
    abc.add_argument('--pref', help=h3, default=pref)
    field = ['ccdnum', 'band']
    h4 = 'For the SSIM analysis, which keywords to store for the output table.'
    h4 += ' Input space separated keywords to be extracted from the header.'
    h4 += ' Default: {0}'.format(' '.join(field)) 
    abc.add_argument('--field', help=h4, default=field, nargs='+')
    # Parse
    abc = abc.parse_args()
    #
    aux_main(vlist1=abc.ratio, vlist2=abc.ssim, prefix=abc.pref, 
             field=abc.field)
