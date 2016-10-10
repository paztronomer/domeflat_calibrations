'''flatDWT
Created: September 29, 2016

This script must be able to detect if a flat is acceptable inmediatly after 
exposure
Use dflats tagged as bad in FLAT_QA 
Use cropped pieces of code from flatStat and decam_test

Oct 4th: DMWY as selected wavelet (symmetric,orthogonal,biorthogonal)

STEPS:
1) Do it on a single ccd with all the possibilities --done
2) do it well on FP with all the possibilities
3) Do it for good/bad flats and compare results
'''
import os
import sys
import time 
import numpy as np
import scipy.stats
import scipy.signal
import matplotlib.pyplot as plt
import fitsio
import pywt

class TestCCD():
    '''methods for loading single CCDs, for testing
    '''
    #==========================================================
    # CANNOT USE THIS METHOD!!! ONLY FOR TEST ON SPECIFIC IMAGE
    def opentest(self,filename):
        '''to open fits and assign to a numpy array, it must be done
        inside the open/close of the file, otherwise the data structure
        remains as None.
        '''
        N_ext = 0 
        header = fitsio.read_header(filename)
        '''control point to make sure data are in fact Science Images
        use lowercase
        '''
        print '\n\tTHIS METHOD IS ONLY FOR TEST. OTHERWISE IT WIll FAIL\n'
        if fitsio.FITS(filename)[0].get_extname().lower() != 'sci':
            print '\n\tHeader extension name is not SCI'
            exit(0)
        else:
            data = fitsio.FITS(filename)[0]#[57-1:1080+1,1:4096+1]
        return header,data
    #pass


class Toolbox():
    '''methods to be inserted in other 
    '''
    def detect_outlier(self,imlayer):
        ''' Estimate Iglewicz and Hoaglin criteria for outlier 
        AND REPLACE BY MEDIAN OF VICINITY! Change this because we only
        want the values masked, not to replace them
        http://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm
        Formula:
        Z=0.6745(x_i - median(x)) / MAD
        if abs(z) > 3.5, x_i is a potential outlier
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and 
        Handle Outliers", The ASQC Basic References in Quality Control: 
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
        '''
        from statsmodels.robust import scale #for mad calculation
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

        ''' search for outliers and if present replace by the 
        median of neighbors
        '''
        if len(Zscore[Zscore>3.5])>0:
            for k in range(0,imlayer.shape[0]):
                for m in range(0,imlayer.shape[1]):
                    if np.abs( cte*(imlayer[k,m]-np.median(flat_im))/MAD )>3.5:
                        imlayer[k,m] = np.nan
            return imlayer
        else:
            return imlayer

    def quick_stat(self,arr_like):
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

    def view_dwt(self,img_arr):
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
    
class FPScience():
    '''methods for loading science focal plane (1-62), inherited 
    from crosstalk
    Maybe import crosstalk
    '''
    pass


class FPAll():
    '''methods for loading the entire FP, inherited from crosstalk
    CCDs 1-62, 63-64, 65+74
    '''
    pass


class DWT():
    '''methods for discrete WT of one level
    pywt.threshold
    '''
    def cutlevel():
        return False
    
    
    def single_level(self,img_arr,wvfunction='dmey',wvmode='symmetric'):
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
        
        
    def multi_level(self,img_arr,wvfunction='dmey',wvmode='symmetric',Nlev=3):
        '''Wavelet Decomposition in multiple levels, opossed to DWT which is
        the wavelet transform of one level only
        - Nlev: number of level for WAVEDEC2 decomposition
        - WAVEDEC2 output is a tuple (cA, (cH, cV, cD)) where (cH, cV, cD)
          repeats Nwv times
        '''
        c_ml = pywt.wavedec2(img_arr,pywt.Wavelet(wvfunction),
                             wvmode,level=Nlev)
        print 'Multilevel wavelet decomposition, N={0}'.format(len(c_ml)-1)
        return c_ml


if __name__=='__main__':
    print 'main here'

    #only one ccdnum=21
    path = '/Users/fco/Code/shipyard_DES/raw_201608_dflat/'
    fname = path+'DECam_00565285_21.fits'
    #to route to file on flat_qa, import easyaccess
    #ccd by ccd single epoch flats desarchive/OPS/precal/20160811-r2440/p02/xtalked-dflat

    auxFn = TestCCD()
    header,img = auxFn.opentest(fname)
    dwt_a = DWT()
    dwt_a.single_level(img[:,:])
    dwt_a.multi_level(img[:,:])
