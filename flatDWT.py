'''flatDWT
Created: September 29, 2016

This script must be able to detect if a flat is acceptable inmediatly after 
exposure
Use dflats tagged as bad in FLAT_QA 
Use cropped pieces of code from flatStat and decam_test

Oct 4th: DMEY as selected wavelet (symmetric, orthogonal, biorthogonal)

Improvements
============
1) Do it for good/bad flats and compare results
2) Be able to use easyaccess instead of despydb
'''


import os
import sys
import socket
import time 
import argparse
import logging
import numpy as np
import scipy.stats
import scipy.signal
import scipy.interpolate
import matplotlib.pyplot as plt
import fitsio
import pywt
import tables
#
# Basic setup for logging
logging.basicConfig(level=logging.INFO)
#
try:
    import despydb.desdbi as desdbi
except:
    e = sys.exc_info()[0]
    logging.error('{0}'.format(e))
    logging.warning('Consider to use easyaccess instead')


class Toolbox():
    '''methods to be inserted in other classes
    '''
    @classmethod
    def detect_outlier(cls, imlayer):
        ''' Estimate Iglewicz and Hoaglin criteria for outlier 
        and replace by NaN
        http://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm
        Formula:
        Z=0.6745(x_i - median(x)) / MAD
        if abs(z) > 3.5, x_i is a potential outlier
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and 
        Handle Outliers", The ASQC Basic References in Quality Control: 
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
        '''
        #from statsmodels.robust import scale #alternative mad calculation
        from scipy.stats import norm
        '''Percent point function (inverse of cdf -- percentiles) of a normal
        continous random variable
        '''
        cte = norm.ppf(0.75) #aprox 0.6745
        '''Flatten the image, to estimate median
        '''
        flat_im = imlayer.ravel()
        #flat_im=flat_im[flat_im!=-1]#exclude '-1' values
        MAD = np.median( np.abs( flat_im-np.median(flat_im) ) )
        #alternative: scale.mad(flat_im, c=1, axis=0, center=np.median)
        Zscore = cte*(flat_im-np.median(flat_im))/MAD
        Zscore = np.abs(Zscore)
        ''' search for outliers and if present replace by -1 
        '''
        imlayer[np.where(np.abs(cte*(imlayer-np.median(flat_im))/MAD)>3.5)]=-1.
        '''
        if len(Zscore[Zscore>3.5])>0:
            for k in range(0, imlayer.shape[0]):
                for m in range(0, imlayer.shape[1]):
                    if np.abs( cte*(imlayer[k, m]-np.median(flat_im))/MAD )>3.5:
                        imlayer[k, m] = np.nan
            return imlayer
        else:
            return imlayer
        '''
        return imlayer

    @classmethod
    def quick_stat(cls, arr_like):
        MAD = np.median( np.abs(arr_like-np.median(arr_like)) )
        print '__________'
        print '* Min | Max | Mean = {0} | {1} | {2}'.format(
            np.min(arr_like), np.max(arr_like), np.mean(arr_like))
        print '* Median | Std | MAD = {0} | {1} | {2}'.format(
            np.median(arr_like), np.std(arr_like), MAD)
        print '* .25 | .5 | .75 = {0} | {1} | {2}'.format(
            np.percentile(arr_like, .25), np.percentile(arr_like, .5), 
            np.percentile(arr_like, .75))
        return False

    @classmethod
    def dwt_library(cls, img_arr):
        '''Display all the availbale DWT (single level) for the input array  
        '''
        t1 = time.time()
        count = 0
        for fam in pywt.families():
            for mothwv in pywt.wavelist(fam):
                for mod in pywt.Modes.modes:
                    print '\tWavelet: {0} / Mode: {1}'.format(mothwv, mod)
                    (c_A, (c_H, c_V, c_D)) = pywt.dwt2(img_arr, 
                                                    pywt.Wavelet(mothwv), 
                                                    mod)
                    count += 1 
                    fig=plt.figure(figsize=(6, 11))
                    ax1=fig.add_subplot(221)
                    ax2=fig.add_subplot(222)
                    ax3=fig.add_subplot(223)
                    ax4=fig.add_subplot(224)
                    ax1.imshow(c_A, cmap='seismic')#'flag'
                    ax2.imshow(c_V, cmap='seismic')
                    ax3.imshow(c_H, cmap='seismic')
                    ax4.imshow(c_D, cmap='seismic')
                    plt.title('Wavelet: {0} / Mode: {1}'.format(mothwv, mod))
                    plt.subplots_adjust(left=.06, bottom=0.05, 
                                        right=0.99, top=0.99, 
                                        wspace=0., hspace=0.)
                    plt.show()
        t2 = time.time()
        print ('\nTotal time in all {1} modes+mother wavelets: {0:.2f}\''
               .format((t2-t1)/60., count))

    @classmethod
    def range_str(cls, head_rng):
        head_rng = head_rng.strip('[').strip(']').replace(':', ', ').split(', ')
        return map(lambda x: int(x)-1, head_rng)

    @classmethod
    def plot_flat(cls, img):
        '''out of the FP points are masked with value -1
        '''
        img = np.ma.masked_where((img==-1), img)
        percn = [np.percentile(img, k) for k in range(1, 101, 1)]
        ind_aux = len(percn) - percn[::-1].index(-1)
        minVal = percn[ind_aux]
        #
        fig = plt.figure(figsize=(12, 6))
        # 
        ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=1, rowspan=1)
        ax2 = plt.subplot2grid((2, 3), (0, 1), colspan=2, rowspan=2)
        ax3 = plt.subplot2grid((2, 3), (1, 0), colspan=1, rowspan=1)
        #ax1 = fig.add_subplot(221, adjustable='box', aspect=1.)
        #ax2 = fig.add_subplot(222, adjustable='box', aspect=1.)
        #ax3 = fig.add_subplot(223)
        #contour filled
        from matplotlib import colors, ticker, cm
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        Y = np.linspace(0, img.shape[0]-1, img.shape[0])
        X = np.linspace(0, img.shape[1]-1, img.shape[1])
        X, Y = np.meshgrid(X, Y)
        cs = ax1.contourf(X, Y, img, locator=ticker.LogLocator(), cmap=cm.PuBu_r, 
                          nchunk=0, lines='solid', origin='lower')
        divider1 = make_axes_locatable(ax1)
        cax1 = divider1.append_axes('left', size='10%', pad=0.9)
        cbar1 = plt.colorbar(cs, cax=cax1)
        #imshow
        img_aux = img #np.ma.masked_where((img<1), img)
        im = ax2.imshow(img_aux, cmap=cm.PuBu_r, vmin=minVal, origin='lower')
        ax2.contour(X, Y, img, locator=ticker.LogLocator(), colors='red', 
                    nchunk=0, linestyles='solid', linewidths=1, origin='lower')
        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes('right', size='10%', pad=0.2)
        cbar2 = plt.colorbar(im, cax=cax2)
        #histogram and univariate
        imgH = np.sort(img.ravel())
        p, x = np.histogram(imgH[np.where(imgH>minVal)], 
                            bins=100)
        #convert bin edges to centers
        x = x[:-1] + (x[1]+x[2])/2.
        f = scipy.interpolate.UnivariateSpline(x, p, s=len(imgH)/20)
        ax3.hist(imgH[np.where(imgH>minVal)], bins=100, 
                 histtype='stepfilled', color='#43C6DB')
        #
        ax2.invert_yaxis
        ax3.plot(x, f(x), 'k-')
        ax3.set_ylim([0, np.max(f(x))])
        plt.tight_layout()
        #plt.subplots_adjust(left=.025, bottom=0.05, right=0.96, top=0.97)
        plt.show()

    @classmethod
    def dbquery(cls, toquery, outdtype, 
                dbsection='db-desoper', help_txt=False):
        '''the personal setup file .desservices.ini must be pointed by desfile
        DB section by default will be desoper
        '''
        desfile = os.path.join(os.getenv('HOME'), '.desservices.ini')
        section = dbsection
        dbi = desdbi.DesDbi(desfile, section)
        if help_txt: help(dbi)
        cursor = dbi.cursor()
        cursor.execute(toquery)
        cols = [line[0].lower() for line in cursor.description]
        rows = cursor.fetchall()
        outtab = np.rec.array(rows, dtype=zip(cols, outdtype))
        return outtab

    @classmethod
    def niterange(cls, niterange, req, att):
        '''Method to ask the DB for some relevant information to run 
        this script over a range of nights. This method also query for
        basic information to be included in the HDF5 table
        '''
        N1, N2 = niterange
        query = "select n.filename, q.factor, q.rms, q.worst,"
        query += "    m.pfw_attempt_id, m.band, m.nite, q.expnum,"
        query += "    pfw.reqnum, i.path"
        query += " from flat_qa q, miscfile m, miscfile n,"
        query += "    file_archive_info i, pfw_attempt pfw"
        query += " where q.filename=m.filename"
        query += "    and n.filename=i.filename"
        query += "    and m.nite between {0} and {1}".format(N1, N2)
        query += "    and m.pfw_attempt_id=pfw.id"
        query += "    and pfw.reqnum={0}".format(req)
        query += "    and pfw.attnum in ({0})".format(','.join(map(str, att)))
        query += "    and m.filetype='compare_dflat_binned_fp'"
        query += "    and m.pfw_attempt_id=n.pfw_attempt_id"
        query += "    and m.expnum=n.expnum"
        query += "    and n.filetype='pixcor_dflat_binned_fp'"
        datatype = ['a100', 'f4', 'f4', 'f4', 'i4', 'a10', 'i4', 'i4', 'i4', 
                    'a100']
        tab = Toolbox.dbquery(query, datatype)
        return tab


class FPSci():
    def __init__(self, folder, parent_root):
        '''Simple method to construct the focal plane array, DECam is 8x12  
        '''
        aux_fp = np.zeros((4096*8, 2048*13), dtype=float)
        max_r, max_c = 0, 0
        #list all on a directory having same root
        for (path, dirs, files) in os.walk(folder):
            for index, item in enumerate(files):   #file is a string
                if (parent_root in item) and not ('fits.fz' in item) :
                    M_header = fitsio.read_header(path+item)
                    M_hdu = fitsio.FITS(path+item)[0]
                    posA = Toolbox.range_str(M_header['detseca'])
                    posB = Toolbox.range_str(M_header['detsecb'])
                    datA = Toolbox.range_str(M_header['dataseca'])
                    datB = Toolbox.range_str(M_header['datasecb'])
                    if posA[1] > max_c: max_c = posA[1]
                    if posA[3] > max_r: max_r = posA[3]
                    if posB[1] > max_c: max_c = posB[1]
                    if posB[3] > max_r: max_r = posB[3]
                    ampA = M_hdu.read()[datA[2]:datA[3]+1, datA[0]:datA[1]+1]
                    ampB = M_hdu.read()[datB[2]:datB[3]+1, datB[0]:datB[1]+1]
                    aux_fp[posA[2]:posA[3]+1, posA[0]:posA[1]+1] = ampA
                    aux_fp[posB[2]:posB[3]+1, posB[0]:posB[1]+1] = ampB
        self.fpSci = aux_fp[:max_r+1, :max_c+1]
        aux_fp = None


class FPBinned():
    def __init__(self, folder, fits):
        '''Simple method to open focal plane binned images
        When a position not belongs to focal plane, the value is -1
        Before return it, add 1 to set outer region to zero value
        '''
        fname = os.path.join(folder, fits)
        M_header = fitsio.read_header(fname)
        M_hdu = fitsio.FITS(fname)[0]
        tmp = M_hdu.read()
        tmp = Toolbox.detect_outlier(tmp)
        tmp += 1.
        self.fpBinned = tmp


class DWT():
    '''methods for discrete WT of one level
    pywt.threshold
    '''
    @classmethod
    def cutlevel(cls):
        return False
    
    @classmethod
    def single_level(cls, img_arr, wvfunction='dmey', wvmode='zero'):
        '''DISCRETE wavelet transform
        Wavelet families available: 'haar', 'db', 'sym', 'coif', 'bior', 
        'rbio', 'dmey'
        http://www.pybytes.com/pywavelets/regression/wavelet.html
        - When flat shows issues, it presents discontinuities in flux
        - To perform the wavelet, border effects must be considered
        - Bumps at the edge are by now flagged but if something
          could be done, it would represent a huge improvement, specially 
          for th SN team
        FOR THE ENTIRE FOCAL PLANE
        --------------------------
        Must fill the interCCD space with zeroes (?) or interpolation.
        Scales must define the refinement scales I will look. The more scales, 
        the slower calculation and better the resolution.
        - MODES: different ways to deal with border effects
        - DWT2/WAVEDEC2 output is a tuple (cA, (cH, cV, cD)) where (cH, cV, cD)
          repeats Nwv times
        Coeffs:
        c_A : approximation (mean of coeffs) coefs
        c_H, c_V, c_D : horizontal detail, vertical, and diagonal coeffs
        '''
        (c_A, (c_H, c_V, c_D)) = pywt.dwt2(img_arr, pywt.Wavelet(wvfunction), 
                                           wvmode)
        #rec_img = pywt.idwt2((c_A, (c_H, c_V, c_D)), WVstr)#, mode='sym')
        '''reduced set of parameters through pywt.threshold()
        with args: soft, hard, greater, less
        To define a set of points (with same dimension as the image), the 
        detailed coeffs must be employed
        '''
        return c_A, c_H, c_V, c_D
        
    @classmethod
    def multi_level(cls, img_arr, wvfunction='dmey', wvmode='zero', Nlev=8):
        '''Wavelet Decomposition in multiple levels, opossed to DWT which is
        the wavelet transform of one level only
        - Nlev: number of level for WAVEDEC2 decomposition
        - WAVEDEC2 output is a tuple (cA, (cH, cV, cD)) where (cH, cV, cD)
          repeats Nwv times
        '''
        c_ml = pywt.wavedec2(img_arr, pywt.Wavelet(wvfunction), 
                             wvmode, level=Nlev)
        aux_shape = []
        for i in range(len(c_ml)):
            if i == 0:
                aux_shape.append(c_ml[i].shape)
            else:
                aux_shape.append(c_ml[i][0].shape)
        cls.cmlshape = aux_shape
        return c_ml


class Coeff(DWT):
    '''method for save results of DWT on a compressed pytables
    '''
    @classmethod
    def set_table(cls, tablename, Nlev):
        '''Method for initialize the file to be filled with the results from
        the DWT decomposition. It works either for 2 or 8 levels of 
        decomposition.
        Descriptions for pytables taken from "Numerical Python: A practical 
        techniques approach for Industry"
        '''
        #create a new pytable HDF5 file handle. This does not represents the
        #root group. To access the root node must use cls.h5file.root
        cls.h5file = tables.open_file(
            tablename, mode='w', 
            title='HDF5 file containing data of dome flat wavelet-behavior.', 
            driver='H5FD_CORE'
            )
        #create groups of the file handle object. Args are: path to the parent
        #group (/), the group name, and optionally a title for the group. The  
        #last is a descriptive attribute can be set to the group.
        group = cls.h5file.create_group(
            '/', 'dwt', 
            title='Created using PyWavelets {0}'.format(pywt.__version__)
            )
        #the file handle is defined, also the group inside it. Under the group
        #we will save the DWT tables. 
        #to create a table with mixed type structure we create a class that
        #inherits from tables.IsDescription class
        if Nlev == 2:
            class Levels(tables.IsDescription):
                c_A = tables.Float32Col(shape=DWT.cmlshape[0])
                c1 = tables.Float32Col(shape=DWT.cmlshape[1])
                c2 = tables.Float32Col(shape=DWT.cmlshape[2])
        elif Nlev == 8:
            class Levels(tables.IsDescription):
                c_A = tables.Float32Col(shape=DWT.cmlshape[0])
                c1 = tables.Float32Col(shape=DWT.cmlshape[1])
                c2 = tables.Float32Col(shape=DWT.cmlshape[2])
                c3 = tables.Float32Col(shape=DWT.cmlshape[3]) 
                c4 = tables.Float32Col(shape=DWT.cmlshape[4])
                c5 = tables.Float32Col(shape=DWT.cmlshape[5])
                c6 = tables.Float32Col(shape=DWT.cmlshape[6])
                c7 = tables.Float32Col(shape=DWT.cmlshape[7])
                c8 = tables.Float32Col(shape=DWT.cmlshape[8])
        else:
            raise ValueError('Method supports 2 or 8 decomposition levels.')
        #with the table structure already defined, the table with DWT results 
        #can be fully created. Args are: a group object or the path to the root
        #node, the table name, the table structure specification, and 
        #optionally the table title. The last is stored as attribute.
        cls.cml_table = cls.h5file.create_table(
            group, 
            'dmeyN2', 
            Levels, 
            title='Mother wavelet:dmey. Levels:2.'
            )

    @classmethod 
    def fill_table(cls, coeff_tuple, Nlev, dict_database):
        '''Method to fill the HDF5 file with the DWT results. It works for 2
        and 8 levels of decomposition.
        '''
        #using .attrs or ._v_attrs we can access different levels in the 
        #HDF5 structure
        cls.cml_table.attrs.DB_INFO = dict_database 
        cml_row = Coeff.cml_table.row
        if Nlev == 2:
            for m in xrange(3):
                cml_row['c_A'] = coeff_tuple[0]
                cml_row['c1'] = coeff_tuple[1][m]
                cml_row['c2'] = coeff_tuple[2][m]
                cml_row.append()
                Coeff.cml_table.flush()
        elif Nlev == 8:
            for m in xrange(3):
                cml_row['c_A'] = coeff_tuple[0]
                cml_row['c1'] = coeff_tuple[1][m]
                cml_row['c2'] = coeff_tuple[2][m]
                cml_row['c3'] = coeff_tuple[3][m]
                cml_row['c4'] = coeff_tuple[4][m]
                cml_row['c5'] = coeff_tuple[5][m]
                cml_row['c6'] = coeff_tuple[6][m]
                cml_row['c7'] = coeff_tuple[7][m]
                cml_row['c8'] = coeff_tuple[8][m]
                cml_row.append()
                Coeff.cml_table.flush()
        else:
            raise ValueError('Method supports 2 or 8 decomposition levels.')
    
    @classmethod
    def close_table(cls):
        Coeff.cml_table.flush()
        Coeff.h5file.close()


if __name__=='__main__':
    logging.info('Running on {0}'.format(socket.gethostname()))
    #
    descr = 'Code to run Discrete (Decimated) Wavelet transform over a set of'
    descr += ' binned_fp DECam images. This code is not optimized or parallel.'
    wav = argparse.ArgumentParser(description=descr)  
    #
    t0 = 'Initial and final night to be considered when looking for images'
    wav.add_argument('night_range', help=t0, type=int, nargs=2)
    t1 = 'Label to be used for this run. Default: \'YN\''
    wav.add_argument('-l', help=t1, default='YN', metavar='')
    t2 = 'Directory where to save the results of the run. Default:'
    t2 += ' /work/devel/fpazch/shelf/dwt_dmeyN2/pixcor/'
    wav.add_argument('-s', help=t2, metavar='', 
                     default='/work/devel/fpazch/shelf/dwt_dmeyN2/pixcor/')
    t3 = 'Level of decomposition to be used in the Wavelet. So far, saving'
    t3 += ' results is only coded for 2 and 8 levels. See UNDECIMATED version'
    t3 += ' for a more flexible coding. Default: 2'
    wav.add_argument('-d', help=t3, metavar='', type=int, default=2)
    t4 = 'Request number (reqnum) of the run to analyze'
    wav.add_argument('-r', help=t4, metavar='', type=int)
    t5 = 'Attempt number (attnum) associated to the reqnum to analyze. It can'
    t5 += ' be given as a integer or as space-separated list'
    wav.add_argument('-a', help=t5, metavar='', type=int)
    #
    wav = wav.parse_args()
    logging.info('Starting at {0}'.format(time.ctime()))
    logging.info('Starting with {0}, {1}'.format(label.upper(), yXn))
    # columns:  q.filename, q.factor, q.rms, q.worst, m.pfw_attempt_id, 
    # m.band, m.nite, q.expnum, i.path
    dflat_tab = Toolbox.niterange(wav.night_range, wav.r, wav.a)
    rootpath = '/archive_data/desarchive'
    outpath = wav.s
    count = 1
    for k in xrange(dflat_tab.shape[0]):
        t1 = time.time()
        #basic information to include in HDF5
        binfo = dict(zip(dflat_tab.dtype.names, dflat_tab[k]))
        dirfile = os.path.join(rootpath, dflat_tab['path'][k])
        bin_fp = FPBinned(dirfile, dflat_tab['filename'][k]).fpBinned
        #for stamps the maximum is Nlev=2
        decLev = wav.d
        c_ml = DWT.multi_level(bin_fp, Nlev=decLev)
        #init table
        fnout = outpath
        fnout += dflat_tab['filename'][k][:dflat_tab['filename'][k].find('pixcor')]
        fnout += label + '.h5'
        Coeff.set_table(fnout, decLev)
        #fill table
        Coeff.fill_table(c_ml, decLev, binfo)
        #close table
        Coeff.close_table()    
        count += 1
        t2 = time.time()
        if (not count%10):
            aux = '{2}--{1}, read+multilevel+store: {0:.2f} s'.format(
                t2-t1, dflat_tab['filename'][k], count
                )
            logging.info(aux)
    logging.info('Ending at {0}'.format(time.ctime()))


    # ====================================
    # Below are the remainders of old runs
    #
    #
    #
    '''
    NAME       MINNITE  MAXNITE   MINEXPNUM  MAXEXPNUM
    ---------------------------------------------------
    SVE1       20120911 20121228     133757      164457
    SVE2       20130104 20130228     165290      182695
    Y1E1       20130815 20131128     226353      258564
    Y1E2       20131129 20140209     258621      284018
    Y2E1       20140807 20141129     345031      382973
    Y2E2       20141205 20150518     383751      438346
    Y3         20150731 20160212     459984      516846
    Y4         20160813 20170212     563912      573912
    '''
    BINNED = True
    CCD = False

    if False: # BINNED:
        #setup samples
        #yXn = [20160813, 20170212] 
        #yXn = [20160813, 20170218]
        #yXn = [20150813, 20160810]
        # yXn = [20161216, 20170223]
        # label = 'y4'
        yXn = [20170815, 20180113]
        label = "y5"
        print '\n===\t===\n\tStarting with {0}, {1}\n'.format(label.upper(), yXn)
        #run for every group and save as H5 table files
        #columns:  q.filename, q.factor, q.rms, q.worst, m.pfw_attempt_id, 
        #m.band, m.nite, q.expnum, i.path
        dflat_tab = Toolbox.niterange(yXn)

        rootpath = '/archive_data/desarchive'
        outpath = '/work/devel/fpazch/shelf/dwt_dmeyN2/pixcor/'
        count = 1
        for k in xrange(dflat_tab.shape[0]):
            t1 = time.time()
            #basic information to include in HDF5
            binfo = dict(zip(dflat_tab.dtype.names, dflat_tab[k]))
            dirfile = os.path.join(rootpath, dflat_tab['path'][k])
            bin_fp = FPBinned(dirfile, dflat_tab['filename'][k]).fpBinned
            #for stamps the maximum is Nlev=2
            decLev = 2
            c_ml = DWT.multi_level(bin_fp, Nlev=decLev)
            #init table
            fnout = outpath
            fnout += dflat_tab['filename'][k][:
                            dflat_tab['filename'][k].find('pixcor')]
            fnout += label + '.h5'
            Coeff.set_table(fnout, decLev)
            #fill table
            Coeff.fill_table(c_ml, decLev, binfo)
            #close table
            Coeff.close_table()    
            count += 1
            t2 = time.time()
            if not count%10:
                print '\n{2}--{1}, read+multilevel+store: {0:.2f}\'\''.format(
                    t2-t1, dflat_tab['filename'][k], count)
    if False:
        '''For a single exposure, CCD by CCD
        '''
        path = '/Users/fco/Code/shipyard_DES/raw_201608_dflat/'
        t1 = time.time()
        whole_fp = FPSci(path, 'DECam_00565285').fpSci
        t2 = time.time()
        print '\telapsed time in filling FP: {0:.2f}\''.format((t2-t1)/60.)
        t1 = time.time()
        print '\tsingle-level DWT'
        c_A, c_H, c_V, c_D = DWT.single_level(whole_fp)
        #SAVE?????
        t2 = time.time()
        print '\n\tElapsed time in DWT the focal plane: {0:.2f}\''.format(
                                                                    (t2-t1)/60.)
        print '\tmulti-level DWT'
        t1 = time.time()
        c_ml = DWT.multi_level(whole_fp, Nlev=8)
        t2 = time.time()
        print '\n\tElapsed time in DWT in 8 levels: {0:.2f}\''.format((t2-t1)
                                                                    /60.)
        #init table
        Coeff.set_table('dwt_ID.h5')
        #fill table
        Coeff.fill_table(c_ml)
        #close table
        Coeff.close_table()
