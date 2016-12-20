'''Script to make a mask scalable
Francisco Paz-Chinchon
'''
import os
import sys
import time
import gc
import numpy as np
import scipy
import scipy.interpolate
import statDWT_dflat as statDWT
#setup for display visualization
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#3D plot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


class Mask():
    @classmethod
    def aux_binned_npy(cls,save_mask=False,fname_mask='fp_BinnedMask.npy'):
        '''use a well behaved image to get the outer mask
        '''
        import y4g1g2g3_flatDWT as y4
        p = '/archive_data/desarchive/ACT/fpazch_Y4N/precal/'
        p += '20160811-r2625/p01/binned-fp/D00562955/'
        fn = 'D00562955_i_r2625p01_compare-dflat-binned-fp.fits'
        tmp = y4.FPBinned(p,fn).fpBinned
        #gets the mask, with '--' for masked entries
        tmp2 = np.ma.masked_greater(tmp,-1,copy=True)
        #gets the boolean matrix
        tmp_mask = np.ma.getmask(tmp2)
        if save_mask:
            np.save(fn_mask,tmp_mask)
        return tmp_mask

    @classmethod 
    def scaling(cls,base,target):
        '''Receives a base mask and emulates its shape on the
        target array, using geometric scaling and nearest neighbor 
        interpolation
        '''
        #options: 
        # scipy.interpolate.interpn  
        # scipy.interpolate.RegularGridInterpolator
        # scipy.interpolate.NearestNDInterpolator
        
        #coordiantes of the base mask
        x,y = np.arange(0,base.shape[1]),np.arange(base.shape[0])
        #defines an interpolator object
        intObj = scipy.interpolate.RegularGridInterpolator((y,x),
                                                        base.astype(int))
        #principle: identity matrix scaling with different scale for x and y
        ratio0 = base.shape[0]/np.float(target.shape[0])
        ratio1 = base.shape[1]/np.float(target.shape[1])
        #by multiplying indice by this ratios, will get the equivalent in 
        #the base mask dimension
        xaux,yaux = np.arange(0,target.shape[1]),np.arange(0,target.shape[0])
        xaux,yaux = ratio1*xaux,ratio0*yaux
        xaux,yaux = np.meshgrid(xaux,yaux)
        cooaux = np.array(zip(yaux.ravel(),xaux.ravel()))
        #use nearest neighbor to find values
        fxy = intObj(cooaux,method='nearest').astype(bool)
        Maux = fxy.reshape(target.shape)
        if False:
            fig = plt.figure()
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            ax1.imshow(base,cmap=plt.get_cmap('summer'),origin='lower')
            ax2.imshow(Maux,cmap=plt.get_cmap('summer'),origin='lower')
            plt.show()
        return Maux


class Call():
    @classmethod
    def call1(cls,base_arr,h5coeff,Nlev=2):
        '''Wrapper to call Mask.scaling() method
        '''
        print 'Performing mask'
        if Nlev == 2:
            cA = [row['c_A'] for row in h5coeff.iterrows()]
            c1 = [row['c1'] for row in h5coeff.iterrows()]
            c2 = [row['c2'] for row in h5coeff.iterrows()]
            #in this test, use diagonal coeffs
            cD = [c1[2],c2[2]]
            tmp_mask = [Mask.scaling(base_arr,aux_c) for aux_c in cD]
        return tmp_mask


if __name__ == '__main__':
    mbool = Mask.aux_binned_npy()

    #opening H5 tables from DWT
    fold = '/work/devel/fpazch/shelf/dwt_Y4Binned/'
    for (path,dirs,files) in os.walk(fold):
        for ind,item in enumerate(files):
            if '.h5' in item:
                gc.collect()
                print item
                try:
                    H5tab = statDWT.OpenH5(fold+item)
                    table = H5tab.h5.root.coeff.FP
                    Call.call1(mbool,table)
                finally:
                    H5tab.closetab()
