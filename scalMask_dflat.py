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
        x,y = np.arange(0,base.shape[1]),np.arange(0,base.shape[0])
        #defines an interpolator object
        intObj = scipy.interpolate.RegularGridInterpolator((y,x),
                                                        base.astype(int))
        #principle: identity matrix scaling with different scale for x and y
        #is important to use the value of shape-1, otherwise the maximum
        #value are outside the base/target scaling dimensions
        ratio0 = (base.shape[0]-1.)/np.float(target.shape[0]-1.)
        ratio1 = (base.shape[1]-1.)/np.float(target.shape[1]-1.)
        #by multiplying indice by this ratios, will get the equivalent in 
        #the base mask dimension
        xaux,yaux = np.arange(0,target.shape[1]),np.arange(0,target.shape[0])
        xaux,yaux = ratio1*xaux,ratio0*yaux
        #at this point, just as double check, see if the maximum value is 
        #out of bounds
        #issue: even when are the same number, True is triggered
        if (np.max(xaux) > base.shape[1]-1.): xaux[-1] = base.shape[1]-1.
        if (np.max(yaux) > base.shape[0]-1.): yaux[-1] = base.shape[0]-1.
        #construct the grid for interpolation
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
        #clean the interpolator
        intObj = None
        return Maux
    
    @classmethod
    def cAHV_mask(cls,h5table,savemask=False):
        '''Method to construct a mask and save it. Once the mask is saved as 
        a numpy file, this method can be safely erased
        '''
        cA = [row['c_A'] for row in h5table.iterrows()]
        cHVD1 = [row['c1'] for row in h5table.iterrows()]
        cHVD2 = [row['c2'] for row in h5table.iterrows()]
        cA_model = np.ma.getmask(np.ma.masked_greater(cA[0],1.,copy=True))
        ca_fakeH = np.zeros(cA_model.shape,dtype=np.bool)
        ca_fakeV = np.zeros(cA_model.shape,dtype=np.bool)
        ca_fakeD = np.zeros(cA_model.shape,dtype=np.bool)
        #analyze this coordinates 
        aux = [[30,86],[30,54],[38,46],[46,38],[62,30]]
        #[x0,y0,x1,y1]
        aux_H = [[67,117,70,118],
                [63,133,66,134],[47,125,62,126],[39,117,46,118],
                [31,109,38,110],[23,101,30,102],[23,87,30,88],
                [23,69,30,70],[23,55,30,56],[31,47,38,48],
                [39,39,46,40],[47,31,62,32],[63,23,78,24],
                [71,133,78,134],[79,125,94,126],[95,117,102,118],
                [103,109,110,110],[111,101,118,102],[111,87,118,88],
                [111,69,118,70],[111,55,118,56],[103,47,110,48],
                [95,39,102,40],[79,31,94,32]]
        aux_V = [[71,119,72,134],[65,119,66,134],[63,127,64,134],
                [47,119,48,126],[39,111,40,118],[31,103,32,110],
                [23,87,24,102],[31,71,32,86],[23,55,24,70],
                [31,47,32,54],[39,39,40,46],[47,31,48,38],
                [63,23,64,30],
                [77,23,78,30],[93,31,94,38],[101,39,102,46],
                [109,47,110,54],[117,55,118,70],[109,71,110,86],
                [117,87,118,102],[109,103,110,110],[101,111,102,118],
                [93,119,94,126],[77,127,78,134]]
        aux_D = [[23,63],[23,77],[23,78],[29,63],[29,77],[29,78],
               [30,63],[30,77],[30,78],[31,47],[31,61],[31,62],
               [31,63],[31,77],[31,78],[31,79],[31,93],[31,94],
               [37,47],[37,93],[37,94],[38,47],[38,93],[38,94],
               [39,39],[39,45],[39,46],[39,47],[39,93],[39,94],
               [39,95],[39,101],[39,102],[45,39],[45,101],[45,102],
               [46,39],[46,101],[46,102],[47,31],[47,37],[47,38],
               [47,39],[47,101],[47,102],[47,103],[47,109],[47,110],
               [53,31],[53,109],[53,110],[54,31],[54,109],[54,110],
               [55,23],[55,29],[55,30],[55,31],[55,109],[55,110],
               [55,111],[55,117],[55,118],[69,23],[69,29],[69,30],
               [69,31],[69,109],[69,110],[69,111],[69,117],[69,118],
               [70,23],[70,29],[70,30],[70,31],[70,109],[70,110],
               [70,111],[70,117],[70,118],[71,31],[71,109],[71,110],
               [85,31],[85,109],[85,110],[86,31],[86,109],[86,110],
               [87,23],[87,29],[87,30],[87,31],[87,109],[87,110],
               [87,111],[87,117],[87,118],[101,23],[101,29],[101,30],
               [101,31],[101,109],[101,110],[101,111],[101,117],[101,118],
               [102,23],[102,29],[102,30],[102,31],[102,109],[102,110],
               [102,111],[102,117],[102,118],[103,31],[103,109],[103,110],
               [109,31],[109,37],[109,38],[109,39],[109,101],[109,102],
               [109,103],[109,109],[109,110],[110,31],[110,37],[110,38],
               [110,39],[110,101],[110,102],[110,103],[110,109],[110,110],
               [111,39],[111,101],[111,102],[117,39],[117,45],[117,46],
               [117,47],[117,65],[117,66],[117,67],[117,69],[117,70],
               [117,71],[117,93],[117,94],[117,95],[117,101],[117,102],
               [118,39],[118,45],[118,46],[118,47],[118,65],[118,66],
               [118,67],[118,68],[118,69],[118,70],[118,71],[118,93],
               [118,94],[118,95],[118,101],[118,102],[119,47],[119,65],
               [119,66],[119,71],[119,93],[119,94],[125,47],[125,61],
               [125,62],[125,63],[125,77],[125,78],[125,79],[125,93],
               [125,94],[126,47],[126,61],[126,62],[126,63],[126,77],
               [126,78],[126,79],[126,93],[126,94],[127,63],[127,77],
               [127,78],[133,63],[133,65],[133,66],[133,71],[133,77],
               [133,78],[134,63],[134,64],[134,65],[134,66],[134,71],
               [134,77],[134,78]]
        #change values
        for i in aux:
            cA_model[i[1],i[0]] = False
        #apply corrected mask to fake masks
        ca_fakeH[np.where(cA_model)] = True
        ca_fakeV[np.where(cA_model)] = True
        ca_fakeD[np.where(cA_model)] = True
        #change values for rectangular regions
        for k in xrange(len(aux_H)):
            ca_fakeH[aux_H[k][1]:aux_H[k][3]+1,aux_H[k][0]:aux_H[k][2]+1] =False
        for m in xrange(len(aux_V)):
            ca_fakeV[aux_V[m][1]:aux_V[m][3]+1,aux_V[m][0]:aux_V[m][2]+1] =False
        #change values for poit to point regions
        for p in aux_D:
            ca_fakeD[p[0],p[1]] = False
        #plotting
        plt.close('all')
        plt.imshow(cA_model,cmap='gray',origin='lower',interpolation='none',
                alpha=.7)
        plt.imshow(ca_fakeH,cmap='Blues',origin='lower',interpolation='none',
                alpha=0.4)
        plt.imshow(ca_fakeV,cmap='Oranges',origin='lower',interpolation='none',
                alpha=0.4)
        plt.imshow(ca_fakeD,cmap='Oranges',origin='lower',interpolation='none',
                alpha=0.4)
        plt.show()
        if savemask:
            #save result, this is a good mask
            np.save('cA_mask.npy',cA_model)
            np.save('cH_mask.npy',ca_fakeH)
            np.save('cV_mask.npy',ca_fakeV)
            np.save('cD_mask.npy',ca_fakeD)
        return False


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
    fold = '/work/devel/fpazch/shelf/dwt_dmeyN2/'
    for (path,dirs,files) in os.walk(fold):
        for ind,item in enumerate(files):
            if '.h5' in item:
                gc.collect()
                print item
                try:
                    H5tab = statDWT.OpenH5(fold+item)
                    table = H5tab.h5.root.coeff.FP
                    Mask.cAHV_mask(table)
                    Call.call1(mbool,table)
                finally:
                    H5tab.closetab()
