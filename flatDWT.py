'''flatDWT
Created: September 29, 2016

This script must be able to detect if a flat is acceptable inmediatly after 
exposure
Use dflats tagged as bad in FLAT_QA 
Use cropped pieces of code from flatStat and decam_test

Oct 4th: DMWY as selected wavelet (symmetric,orthogonal,biorthogonal)

STEPS:
1) Do it on a single ccd with all the possibilities --done
2) do it well on FP with all the possibilities --done
3) Do it for good/bad flats and compare results
'''
import os
import sys
import time 
import numpy as np
import scipy.stats
import scipy.signal
import scipy.interpolate
import matplotlib.pyplot as plt
import fitsio
import pywt
import tables
import despydb.desdbi as desdbi

class Toolbox():
    '''methods to be inserted in other 
    '''
    @classmethod
    def detect_outlier(cls,imlayer):
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
        return imlayer
        '''
        if len(Zscore[Zscore>3.5])>0:
            for k in range(0,imlayer.shape[0]):
                for m in range(0,imlayer.shape[1]):
                    if np.abs( cte*(imlayer[k,m]-np.median(flat_im))/MAD )>3.5:
                        imlayer[k,m] = np.nan
            return imlayer
        else:
            return imlayer
        '''

    @classmethod
    def quick_stat(cls,arr_like):
        MAD = np.median( np.abs(arr_like-np.median(arr_like)) )
        print '__________'
        print '* Min | Max | Mean = {0} | {1} | {2}'.format(
            np.min(arr_like),np.max(arr_like),np.mean(arr_like))
        print '* Median | Std | MAD = {0} | {1} | {2}'.format(
            np.median(arr_like),np.std(arr_like),MAD)
        print '* .25 | .5 | .75 = {0} | {1} | {2}'.format(
            np.percentile(arr_like,.25),np.percentile(arr_like,.5),
            np.percentile(arr_like,.75))
        return False

    @classmethod
    def dwt_library(cls,img_arr):
        '''Display all the availbale DWT (single level) for the input array  
        '''
        t1 = time.time()
        count = 0
        for fam in pywt.families():
            for mothwv in pywt.wavelist(fam):
                for mod in pywt.Modes.modes:
                    print '\tWavelet: {0} / Mode: {1}'.format(mothwv,mod)
                    (c_A,(c_H,c_V,c_D)) = pywt.dwt2(img_arr,
                                                    pywt.Wavelet(mothwv),
                                                    mod)
                    count += 1 
                    fig=plt.figure(figsize=(6,11))
                    ax1=fig.add_subplot(221)
                    ax2=fig.add_subplot(222)
                    ax3=fig.add_subplot(223)
                    ax4=fig.add_subplot(224)
                    ax1.imshow(c_A,cmap='seismic')#'flag'
                    ax2.imshow(c_V,cmap='seismic')
                    ax3.imshow(c_H,cmap='seismic')
                    ax4.imshow(c_D,cmap='seismic')
                    plt.title('Wavelet: {0} / Mode: {1}'.format(mothwv,mod))
                    plt.subplots_adjust(left=.06, bottom=0.05,
                                        right=0.99, top=0.99,
                                        wspace=0., hspace=0.)
                    plt.show()
        t2 = time.time()
        print ('\nTotal time in all {1} modes+mother wavelets: {0:.2f}\''
               .format((t2-t1)/60.,count))

    @classmethod
    def range_str(cls,head_rng):
        head_rng = head_rng.strip('[').strip(']').replace(':',',').split(',')
        return map(lambda x: int(x)-1, head_rng)

    @classmethod
    def plot_flat(cls,img):
        '''out of the FP points are masked with value -1
        '''
        img = np.ma.masked_where((img==-1),img)
        percn = [np.percentile(img,k) for k in range(1,101,1)]
        ind_aux = len(percn) - percn[::-1].index(-1)
        minVal = percn[ind_aux]
        #
        fig = plt.figure(figsize=(12,6))
        
        ax1 = plt.subplot2grid((2,3),(0,0),colspan=1,rowspan=1)
        ax2 = plt.subplot2grid((2,3),(0,1),colspan=2,rowspan=2)
        ax3 = plt.subplot2grid((2,3),(1,0),colspan=1,rowspan=1)
        #ax1 = fig.add_subplot(221,adjustable='box',aspect=1.)
        #ax2 = fig.add_subplot(222,adjustable='box',aspect=1.)
        #ax3 = fig.add_subplot(223)
        #contour filled
        from matplotlib import colors,ticker,cm
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        Y = np.linspace(0,img.shape[0]-1,img.shape[0])
        X = np.linspace(0,img.shape[1]-1,img.shape[1])
        X,Y = np.meshgrid(X,Y)
        cs = ax1.contourf(X,Y,img,locator=ticker.LogLocator(),cmap=cm.PuBu_r,
                        nchunk=0,lines='solid',origin='lower')
        divider1 = make_axes_locatable(ax1)
        cax1 = divider1.append_axes('left',size='10%',pad=0.9)
        cbar1 = plt.colorbar(cs,cax=cax1)
        #imshow
        img_aux = img #np.ma.masked_where((img<1),img)
        im = ax2.imshow(img_aux,cmap=cm.PuBu_r,vmin=minVal,origin='lower')
        ax2.contour(X,Y,img,locator=ticker.LogLocator(),colors='red',
                    nchunk=0,linestyles='solid',linewidths=1,origin='lower')
        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes('right',size='10%',pad=0.2)
        cbar2 = plt.colorbar(im,cax=cax2)
        #histogram and univariate
        imgH = np.sort(img.ravel())
        p,x = np.histogram(imgH[np.where(imgH>minVal)],
                        bins=100)
        #convert bin edges to centers
        x = x[:-1] + (x[1]+x[2])/2.
        f = scipy.interpolate.UnivariateSpline(x,p,s=len(imgH)/20)
        ax3.hist(imgH[np.where(imgH>minVal)],bins=100,
                histtype='stepfilled',color='#43C6DB')
        #
        ax2.invert_yaxis
        ax3.plot(x,f(x),'k-')
        ax3.set_ylim([0,np.max(f(x))])
        plt.tight_layout()
        #plt.subplots_adjust(left=.025,bottom=0.05,right=0.96,top=0.97)
        plt.show()
        

    @classmethod
    def dbquery(cls,toquery,outdtype,dbsection='db-desoper',help_txt=False):
        '''the personal setup file .desservices.ini must be pointed by desfile
        DB section by default will be desoper
        '''
        desfile = os.path.join(os.getenv('HOME'),'.desservices.ini')
        section = dbsection
        dbi = desdbi.DesDbi(desfile,section)
        if help_txt: help(dbi)
        cursor = dbi.cursor()
        cursor.execute(toquery)
        cols = [line[0].lower() for line in cursor.description]
        rows = cursor.fetchall()
        outtab = np.rec.array(rows,dtype=zip(cols,outdtype))
        return outtab

    @classmethod
    def group1(cls,niterange,epoch):
        '''this method could perfectly be at the main
        '''
        N1,N2 = niterange
        fact_val = 0.7
        query = "select f.filename,f.factor,f.rms,f.worst,m.pfw_attempt_id,\
        m.band,m.nite,f.expnum,i.path from flat_qa f, miscfile m,\
        file_archive_info i where m.nite>={0} and m.nite<={1} and \
        m.filename=f.filename and i.filename=f.filename and \
        f.factor<{2} and m.filetype='compare_dflat_binned_fp' and \
        rownum<=50".format(N1,N2,fact_val)
        datatype = ['a80','f4','f4','f4','i4','a10','i4','i4','a100']
        #query
        tab = Toolbox.dbquery(query,datatype)
        return tab

    @classmethod
    def group2(cls,niterange,epoch):
        '''this method could perfectly be at the main
        '''
        N1,N2 = niterange
        fact_val = 0.7
        rms_val = 0.0075
        query = "select f.filename,f.factor,f.rms,f.worst,m.pfw_attempt_id,\
        m.band,m.nite,f.expnum,i.path from flat_qa f, miscfile m,\
        file_archive_info i where m.nite>={0} and m.nite<={1} and \
        m.filename=f.filename and i.filename=f.filename and \
        f.factor>{2} and f.rms>{3} and m.filetype='compare_dflat_binned_fp' \
        and rownum<=50".format(N1,N2,fact_val,rms_val)
        datatype = ['a80','f4','f4','f4','i4','a10','i4','i4','a100']
        #query
        tab = Toolbox.dbquery(query,datatype)
        return tab

    @classmethod
    def group3(cls,niterange,epoch):
        '''this method could perfectly be at the main
        '''
        N1,N2 = niterange
        fact_val = 0.7
        rms_val = 0.0075
        query = "select f.filename,f.factor,f.rms,f.worst,m.pfw_attempt_id,\
        m.band,m.nite,f.expnum,i.path from flat_qa f, miscfile m,\
        file_archive_info i where m.nite>={0} and m.nite<={1} and \
        m.filename=f.filename and i.filename=f.filename and \
        f.factor>{2} and f.rms<{3} and m.filetype='compare_dflat_binned_fp' \
        and rownum<=50".format(N1,N2,fact_val,rms_val)
        datatype = ['a80','f4','f4','f4','i4','a10','i4','i4','a100']
        #query
        tab = Toolbox.dbquery(query,datatype)
        return tab


class FPSci():
    def __init__(self,folder,parent_root):
        '''Simple method to construct the focal plane array, DECam is 8x12  
        '''
        aux_fp = np.zeros((4096*8,2048*13),dtype=float)
        max_r,max_c = 0,0
        #list all on a directory having same root
        for (path,dirs,files) in os.walk(folder):
            for index,item in enumerate(files):   #file is a string
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
                    ampA = M_hdu.read()[datA[2]:datA[3]+1,datA[0]:datA[1]+1]
                    ampB = M_hdu.read()[datB[2]:datB[3]+1,datB[0]:datB[1]+1]
                    aux_fp[posA[2]:posA[3]+1,posA[0]:posA[1]+1] = ampA
                    aux_fp[posB[2]:posB[3]+1,posB[0]:posB[1]+1] = ampB
        self.fpSci = aux_fp[:max_r+1,:max_c+1]
        aux_fp = None


class FPBinned():
    def __init__(self,folder,fits):
        '''Simple method to open focal plane binned images
        When a position not belongs to focal plane, the value is -1
        Try masking those values
        '''
        fname = folder+fits
        M_header = fitsio.read_header(fname)
        M_hdu = fitsio.FITS(fname)[0]
        tmp = M_hdu.read()
        tmp = Toolbox.detect_outlier(tmp)
        self.fpBinned = tmp


class DWT():
    '''methods for discrete WT of one level
    pywt.threshold
    '''
    @classmethod
    def cutlevel(cls):
        return False
    
    @classmethod
    def single_level(cls,img_arr,wvfunction='dmey',wvmode='symmetric'):
        '''DISCRETE wavelet transform
        Wavelet families available: 'haar', 'db', 'sym', 'coif', 'bior', 
        'rbio','dmey'
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
        c_H,c_V,c_D : horizontal detail,vertical,and diagonal coeffs
        '''
        (c_A,(c_H,c_V,c_D)) = pywt.dwt2(img_arr,pywt.Wavelet(wvfunction),
                                        wvmode)
        #rec_img = pywt.idwt2((c_A,(c_H,c_V,c_D)),WVstr)#,mode='sym')
        '''reduced set of parameters through pywt.threshold()
        with args: soft, hard, greater, less
        To define a set of points (with same dimension as the image), the 
        detailed coeffs must be employed
        '''
        return c_A,c_H,c_V,c_D
        
    @classmethod
    def multi_level(cls,img_arr,wvfunction='dmey',wvmode='symmetric',Nlev=8):
        '''Wavelet Decomposition in multiple levels, opossed to DWT which is
        the wavelet transform of one level only
        - Nlev: number of level for WAVEDEC2 decomposition
        - WAVEDEC2 output is a tuple (cA, (cH, cV, cD)) where (cH, cV, cD)
          repeats Nwv times
        '''
        c_ml = pywt.wavedec2(img_arr,pywt.Wavelet(wvfunction),
                             wvmode,level=Nlev)
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
    def set_table(cls,str_tname):
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
        cls.h5file = tables.open_file(str_tname,mode='w',
                                    title='DWT multilevel decomposition',
                                    driver='H5FD_CORE')
        #driver_core_backing_store=0)
        #coeff: group name, DWT coeff: brief description
        group = cls.h5file.create_group('/','coeff','DWT coeff')
        #FP: table name, FP wavelet decomposition:ttable title
        cls.cml_table = cls.h5file.create_table(group,'FP',Levels,'Wavedec')

    @classmethod 
    def fill_table(cls,coeff_tuple):
        #fills multilevel DWT with N=8
        cml_row = Coeff.cml_table.row
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
    
    @classmethod
    def close_table(cls):
        Coeff.h5file.close()


if __name__=='__main__':
    '''For exposures belonging to a group. Either CCD by CCD or binned
    Must setup a criteria to decide!
    '''
    BINNED = True
    CCD = False

    if BINNED:
        #setup samples
        t1 = time.time()
        Y4sample = [20160808,20161009] #[20160813,20170212] entire Y4
        #select 50 first occurences
        g1 = Toolbox.group1(Y4sample,'Y4')
        g2 = Toolbox.group2(Y4sample,'Y4')
        g3 = Toolbox.group3(Y4sample,'Y4')
        t2 = time.time()
        print '\telapsed time in grouping {0}'.format((t2-t1)/60.)
       
        #run for every group and save as H5 table files
        rootpth = '/archive_data/desarchive/'
        outpath = '/work/devel/fpazch/shelf/dwt_Y4Binned/'
        for it in xrange(3):
            if it == 0: g = Toolbox.group1(Y4sample,'Y4'); gg = 'g1'
            if it == 1: g = Toolbox.group2(Y4sample,'Y4'); gg = 'g2'
            if it == 2: g = Toolbox.group3(Y4sample,'Y4'); gg = 'g3'
            for k in xrange(g1.shape[0]):
                print 'group {0}, item {0}'.format(it+1,k+1)
                dirfile = rootpath + g['path'][k] + '/'
                bin_fp = FPBinned(dirfile,g['filename'][k]).fpBinned
                t1 = time.time()
                #for stamps the maximum is Nlev=2
                c_ml = DWT.multi_level(bin_fp,Nlev=2)
                t2 = time.time()
                print '\n\tmultilevel: {0:.2f}\''.format((t2-t1)/60.)
                #init table
                fnout = outpath
                fnout += g['filename'][k][:g['filename'][k].find('compare')]
                fnout += 'DWT_dmeyN2_' + gg  + '.h5'
                Coeff.set_table(fnout)
                #fill table
                Coeff.fill_table(c_ml)
                #close table
                Coeff.close_table()    
    
    if False:
        '''For a single binned FP
        '''
        fname = 'D00237866_i_r1999p06_compare-dflat-binned-fp.fits'
        bin_fp = FPBinned('/Users/fco/Code/shipyard_DES/devel/',fname).fpBinned
        c_ml = DWT.multi_level(bin_fp,Nlev=2)
        #print c_ml
        
        
    if False:
        '''For a single exposure, CCD by CCD
        '''
        path = '/Users/fco/Code/shipyard_DES/raw_201608_dflat/'
        t1 = time.time()
        whole_fp = FPSci(path,'DECam_00565285').fpSci
        t2 = time.time()
        print '\telapsed time in filling FP: {0:.2f}\''.format((t2-t1)/60.)

        t1 = time.time()
        print '\tsingle-level DWT'
        c_A,c_H,c_V,c_D = DWT.single_level(whole_fp)
        #SAVE?????
        t2 = time.time()
        print '\n\tElapsed time in DWT the focal plane: {0:.2f}\''.format(
                                                                    (t2-t1)/60.)
        
        print '\tmulti-level DWT'
        t1 = time.time()
        c_ml = DWT.multi_level(whole_fp,Nlev=8)
        t2 = time.time()
        print '\n\tElapsed time in DWT in 8 levels: {0:.2f}\''.format((t2-t1)
                                                                    /60.)
        #init table
        Coeff.set_table('dwt_ID.h5')
        #fill table
        Coeff.fill_table(c_ml)
        #close table
        Coeff.close_table()
        
