""" Script to plot the comparison between sets of dome flats, output from 
compare_domeflats.py
"""
import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# For colorbar positioning
from mpl_toolkits.axes_grid1 import make_axes_locatable
# For a more comprehensive description:
# https://matplotlib.org/tutorials/toolkits/axes_grid.html
#
# For ZScale plotting
from astropy.visualization import (MinMaxInterval, SqrtStretch,
                                   ImageNormalize, ZScaleInterval)

def aux_binned(regex1=None, outf='binned.pdf'):
    ''' Plot the binned fp. As they're few bands, create a single plot
    '''
    print('Doing a 2x3 grid for plotting')
    fnm = glob.glob(regex1)
    fig, ax = plt.subplots(2, 3, figsize=(8, 6))
    kw = {
        'origin': 'lower',
        'cmap': 'gray',
    }
    for idx, axis in enumerate(ax.flatten()):
        aux_x = np.load(fnm[idx])
        # Set Zscale for plotting
        im_norm = ImageNormalize(aux_x, 
                                 interval=ZScaleInterval(),
                                 stretch=SqrtStretch())
        im = axis.imshow(aux_x, norm=im_norm, **kw)
        # Create the subgrid for the colorbars
        divider = make_axes_locatable(axis)
        caxis = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=caxis)
        # Remove the axis labels
        axis.axes.get_xaxis().set_visible(False)
        axis.axes.get_yaxis().set_visible(False)
        axis.set_title(os.path.basename(fnm[idx])[:11])
    # Fine tune
    plt.subplots_adjust(left=0.01, bottom=0.03, top=0.96, right=0.9,
                        hspace=0.1, wspace=0.4)
    plt.suptitle('Y6/Y5 ratio for binned focal plane, supercal, norm dflat', 
                 color='b')
    if True:
        plt.savefig(outf, dpi=300, format='pdf')
    else:
        plt.show()    

def aux_ssim(regex2=None, outf='ssim_plot_out.pdf'):
    ''' Plots the results for the SSIM analysis. The range of values goes from 
    0 to 1, 1 being the identical. Addition of noise gets a lower SSIM than
    additon of a constant.
    '''
    listf = glob.glob(regex2)
    # Load all the tables, merging them
    for idx, t in enumerate(listf):
        if (idx == 0):
            df = pd.read_csv(t)
        else:
            df = pd.concat([df, pd.read_csv(t)])
    df.reset_index(drop=True, inplace=True)
    # Drop the blank spaces on the band entries
    df['band_1'] = df['band_1'].map(str.strip)
    df['band_2'] = df['band_2'].map(str.strip)
    
    print('Setting a 3x2 grid')

    fig, ax = plt.subplots(3, 2, sharex=True, sharey=True, figsize=(10, 8))
    colors = {'u': 'dodgerblue', 'g': 'green', 'r': 'red',
              'i': 'peru', 'z': 'blueviolet', 'Y': 'gray'}
    for idx, axis in enumerate(ax.flatten()):
        band = list(colors.keys())[idx]
        dfaux = df.loc[df['band_1'] == band]
        axis.plot(dfaux['ccdnum_1'], dfaux['ssim'], color=colors[band],
                  marker='o', markersize=2, label=band)
        axis.legend()
        axis.grid(b=True, which='both', color='gray', alpha=0.8, 
                  linestyle=':', linewidth=0.5)
    ax[2, 0].set_xlabel('CCD')
    ax[2, 1].set_xlabel('CCD')
    ax[0, 0].set_ylabel('Similarity Metrics')
    ax[1, 0].set_ylabel('Similarity Metrics')
    ax[2, 0].set_ylabel('Similarity Metrics')
    txt = 'Y6 vs Y5 supercal full-size CCD comparison, via Similarity metrics'
    txt += '\nNote: noise produces lower metrics than offset'
    plt.suptitle(txt, color='navy', fontweight='bold')
    #
    plt.subplots_adjust(wspace=0, hspace=0, left=0.1, bottom=0.1,
                        top=0.94, right=0.98)
    if True:
        plt.savefig(outf, dpi=300, format='pdf')
    else:
        plt.show()
    return True

if __name__ == '__main__':
    desc = 'Plot outputs from compre_domeflats.py'
    inx = argparse.ArgumentParser(description=desc)
    h1 = 'Regex for the files to be plotted all together'
    inx.add_argument('--reg1', help=h1)
    h2 = 'Filename (.pdf) for the files to be plotted all together'
    inx.add_argument('--out1', help=h2)
    h3 = 'Regex for the tables containing the SSIM one-value result'
    inx.add_argument('--reg2', help=h3)
    h4 = 'Filename (.pdf) for the plot of the SSIM metrics'
    inx.add_argument('--out2', help=h4)
    
    #
    inx = inx.parse_args()
    
    aux_ssim(regex2=inx.reg2, outf=inx.out2)
    aux_binned(regex1=inx.reg1, outf=inx.out1)
