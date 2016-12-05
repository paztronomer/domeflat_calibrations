'''viewDWT_flat
Created: November 7, 2016

This simple script allows to visual inspect and plot some results
as time-series. The target files are the H5 tables generated from the
flatDWT run
Remember (c_A,(c_H,c_V,c_D))
'''

import os
import sys
import time
import numpy as np
import pandas as pd
#setup for display visualization
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#3D plot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import pywt
import tables
import scipy.signal
import scipy.ndimage
import scipy.misc
import scipy.stats
import sklearn.cluster
import sklearn.metrics

class Toolbox():
    @classmethod
    def rms(cls,arr):
        '''returns RMS for ndarray
        '''
        return np.sqrt(np.mean(np.square(arr.ravel())))
   
    @classmethod
    def corr_random(cls,data):
        '''correlate data with random 2D array
        '''
        auxrdm = np.random.rand(data.shape[0],data.shape[1])
        auxrdm = auxrdm/np.mean(auxrdm)
        corr2d = scipy.signal.correlate2d(data,auxrdm,mode='same',
                                        boundary='symm')
        return corr2d
   
    @classmethod
    def gaussian_filter(cls,data,sigma=1.):
        '''performs Gaussian kernel on image
        '''
        return scipy.ndimage.gaussian_filter(data,sigma=sigma)
    
    @classmethod 
    def cluster_dbscan(cls,points,minsize=3):
        '''friends of friends
        points is a set of coordinates
        DBSCAN.labels_: label for each point, where -1 is set for noise. 
        Includes cores and members not belonging to the core but still on
        the group.
        DBSCAN.core_sample_indices_: list of indices of core samples
        DBSCAN.components_:copy of each core sample found by training
        core_sample_indices_
        '''
        #here we consider the diagonal of a square of side=1 because of
        #the positioning we're using is the grid of points with unity spacing
        diagonal = np.sqrt(2.)
        #real impact of leaf_size has not been unveiled
        db = sklearn.cluster.DBSCAN(eps=diagonal,min_samples=minsize,
                                leaf_size=5, metric='euclidean').fit(points)
        #mask to identify which elements in the cluster belongs 
        #to the inner core and which to the outer region 
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        #exclude those labeled as -1
        n_cluster = np.unique(db.labels_[np.where(db.labels_ != -1)]).shape[0] 
        return n_cluster,db.labels_,core_samples_mask

    @classmethod
    def mask_polygon(cls,arr,vertex):
        '''this simple method makes a mask of the input array, based on the 
        input vertices. If subsampling is needed, simply increase the amount
        of points in x,y meshgrid
        '''
        #from matplotlib.path
        #vertex = [(0,0),(20,50),(100,75)]
        x,y = np.meshgrid(np.arange(arr.shape[1]),np.arange(arr.shape[0]))
        #flatten and ravel has same output (in this case) but flatten returns
        #an actual copy while ravel a view of the original array
        x,y = x.flatten(),y.flatten()
        #generates an array of 2 cols of all the pairs of coordinates
        coord = np.vstack((x,y)).T
        poly = matplotlib.path.Path(vertex)
        gmask = poly.contains_points(coord)
        gmask = gmask.reshape((fp.shape[1],fp.shape[0]))
        return gmask
    
    @classmethod
    def binned_mask_npy(cls):
        '''use a well behaved image to get the outer mask
        '''
        import y4g1g2g3_flatDWT as y4
        p = '/archive_data/desarchive/ACT/fpazch_Y4N/precal/'
        p += '20160811-r2625/p01/binned-fp/D00562955/'
        fn = 'D00562955_i_r2625p01_compare-dflat-binned-fp.fits'
        tmp = y4.FPBinned(p,fn).fpBinned
        tmp2 = np.ma.masked_greater(tmp,-1,copy=True)
        tmp_mask = np.ma.getmask(tmp2)
        np.save('fp_BinnedMask.npy',tmp_mask)
        plt.imshow(tmp2,origin='lower')
        plt.show()
        return False

    @classmethod
    def value_dispers(cls,values):
        '''Method to estimate dispersion in the values of the clustered points.
        Not related to the position, but to the power.
        For the entropy, we'll use the values as unnormalized probabilities
        -mean,median,stdev,variance,rms,median,MAD,entropy
        '''
        xA = scipy.ndimage.mean(values)
        xB = scipy.ndimage.median(values)
        xC = scipy.ndimage.standard_deviation(values)
        xD = scipy.ndimage.variance(values)
        xE = Toolbox.rms(values) 
        xF = np.median(np.abs(values-np.median(values)))
        xG = scipy.stats.entropy(values.ravel())
        return (xA,xB,xC,xD,xE,xF,xG)

    @classmethod
    def spatial_dispers(cls,grid_shape,points):
        '''Methods for estimating spatial (angular and radial) dispersion of 
        clustered. Include PCA.
        '''
        #PCA
        from sklearn.decomposition import PCA
        pca = PCA(n_components=4)
        X_r = pca.fit(points).transform(points)
        #print 'PCA shape: ',X_r.shape
        #center of mass
        pts2col = np.array([[p[0],p[1]] for p in points])
        xcm = np.sum(pts2col[:,1]*1.)/pts2col.shape[0]
        ycm = np.sum(pts2col[:,0]*1.)/pts2col.shape[0] 
        #to measure angular uniformity:
        #would be ideally uniformously distributed, not gaussian
        #but flat. Divide in polar coord of 1deg
        origin = ( (grid_shape[1]-1.)/2.,(grid_shape[0]-1.)/2. )
        theta = np.arctan2(pts2col[:,0]-origin[0],pts2col[:,1]-origin[1])
        #ang_step = [np.pi*np.float(i)/8. for i in xrange(9)]
        ang_step = np.sort(np.concatenate([np.linspace(0,np.pi,180),
                                        np.linspace(0.,-np.pi,180)[1:]]))
        ang_count = []
        for it in xrange(1,len(ang_step)):
            loc = np.where(np.logical_and(theta>ang_step[it-1],
                        theta<=ang_step[it]))
            ang_count.append(theta[loc].shape[0])
        ang_count = np.array(ang_count)
        #some measurements 
        ang_aux1 = Toolbox.rms(ang_count)
        ang_aux2 = np.var(ang_count)
        ang_aux3 = np.mean(ang_count)
        ang_aux4 = np.median(np.abs(ang_count-np.median(ang_count)))
        ang_aux5 = scipy.stats.entropy(ang_count)
        #print '\nrms:{0:.4f}\nS2/S1:{1:.4f}\nMAD:{2:.4f}\nS:{3:.4f}'.format(
        #    ang_aux1,ang_aux2/ang_aux3,ang_aux4,ang_aux5)
        #to measure radial uniformity
        #would be ideal to have a flat amount of points per unit area
        rad = np.sqrt(np.square(pts2col[:,1]-origin[1]) + 
                    np.square(pts2col[:,0]-origin[0]))
        #lets try with 20 division os the radius
        rad_step = np.linspace(0,max(origin),20) 
        rad_count = []
        for rr in xrange(1,len(rad_step)):
            area = np.pi*(np.square(rad_step[rr])-np.square(rad_step[rr-1]))
            pos = np.where(np.logical_and(rad<=rad_step[rr],rad>rad_step[rr-1]))
            rad_count.append(np.float(rad[pos].shape[0])/area)
        rad_count = np.array(rad_count)
        #some measurements 
        rad_aux1 = Toolbox.rms(rad_count)
        rad_aux2 = np.var(rad_count)
        rad_aux3 = np.mean(rad_count)
        rad_aux4 = np.median(np.abs(rad_count-np.median(rad_count)))
        rad_aux5 = scipy.stats.entropy(rad_count)
        #print '\nrms:{0:.4f}\nS2/S1:{1:.4f}\nMAD:{2:.4f}\nS:{3:.4f}'.format(
        #    rad_aux1,rad_aux2/rad_aux3,rad_aux4,rad_aux5)
        return (xcm,ycm,ang_aux1,ang_aux2,ang_aux3,ang_aux4,ang_aux5,
            rad_aux1,rad_aux2,rad_aux3,rad_aux4,rad_aux5) 

class Call():
    @classmethod
    def wrap1(cls,h5table,Nlev=2):
        '''Call the method for clustering.DBSCAN different coefficients,
        through the plotting method
        '''
        if Nlev == 2:
            cA = [row['c_A'] for row in h5table.iterrows()]
            c1 = [row['c1'] for row in h5table.iterrows()]
            c2 = [row['c2'] for row in h5table.iterrows()]
            #filter process
            print '\n\t=== USE |coeff|^2 ==='
            if False:
                print '\t=== performing over c_H ==='
                #for level 1, c1[]
                pos1H = Graph.filter_plot(c1[0]*c1[0],(1,'H'))
                pos2H = Graph.filter_plot(c2[0]*c2[0],(2,'H'))
            if True:
                print '\t=== performing over c_V ==='
                #for level 1, c1[]
                pos1V = Graph.filter_plot(c1[1]*c1[1],(1,'V'))
                pos2V = Graph.filter_plot(c2[1]*c2[1],(2,'V'))
            if False:
                print '\t=== performing over c_D ==='
                #for level 1, c1[]
                pos1D = Graph.filter_plot(c1[2]*c1[2],(1,'D'))
                pos2D = Graph.filter_plot(c2[2]*c2[2],(2,'D'))
        else:
            print '\n\t=== Levels not being N=%d aren\'t still setup ==='%(Nlev)
    
    @classmethod
    def wrap2(cls,dwt_nm,h5table,Nlev=2):
        '''Wrap things to call coeff/values dispersion
        NOTE: when mask for borders are ready, then apply this to c_H,c_v
        '''
        #from name
        aux = dwt_nm[:dwt_nm.find('_DWT')]
        p1 = aux.find('_')
        p2 = aux[p1+1:].find('_') + 1 + p1
        expnum = np.int(aux[1:p1])
        band = aux[p1+1:p2]
        reqnum = np.int(aux[p2+2:p2+6])
        attnum = np.int(aux[p2+8:])
        #DIAGONAL COEFFS
        cA = [row['c_A'] for row in h5table.iterrows()]
        c1 = [row['c1'] for row in h5table.iterrows()]
        c2 = [row['c2'] for row in h5table.iterrows()]
        '''Call for all and plot!!!
        '''
        out_list = [] 
        for it,C2 in enumerate([c1,c2]):
            data = C2[2]*C2[2]
            #       Filter and cluster
            #
            #mask will be good for plotting but not for calculations 
            #fdata = np.ma.masked_less_equal(data,Toolbox.rms(data))
            #plt.imshow(fdata,origin='lower',interpolation='none')
            #plt.show()
            #coordinates of points above RMS
            coo_dat = np.argwhere(data>Toolbox.rms(data))
            #clustering friends to friends
            dat_N,dat_label,dat_mask = Toolbox.cluster_dbscan(coo_dat)
            #clusters in data
            #idx = np.where(np.logical_and(dat_label!=-1,dat_mask))
            #selection to be used
            sel_coo = coo_dat[dat_mask]
            sel_val = data[sel_coo]
            #
            val_stat = Toolbox.value_dispers(sel_val)
            spa_stat = Toolbox.spatial_dispers(data.shape,sel_coo)
            out_list.append((expnum,band,reqnum,attnum,it+1)+val_stat+spa_stat)
            #
            #dat_N,dat_label,dat_mask = None,None,None
        #Write to file, use simple pandas
        cols = ['expnum','band','reqnum','attnum','level']
        cols += ['v.mean','v.med','v.std','v.var','v.rms','v.MAD','v.S']
        cols += ['s.xcm','s.ycm','sa.rms','sa.var','sa.mean','sa.MAD','sa.S']
        cols += ['sr.rms','sr.var','sr.mean','sr.MAD','sr.S']
        df = pd.DataFrame(out_list,columns=cols)
        return df
        
        
class OpenH5():
    '''class oriented to open and close a pytables instance. It's not the same
    to close the instance here inside the same class than outside
    '''
    def __init__(self,fname):
        self.h5 = tables.open_file(fname,driver='H5FD_CORE')
    
    def closetab(self):
        self.h5.close()


class TimeSerie():
    pass
    
    
class Graph():
    @classmethod
    def filter_plot(cls,data,(level,coeff)):
        '''Method for display Gaussian filtered imaage, original image,
        number of clustered points, RMS-cut values 
        '''
        #RMS threshold mask arrays
        fdata = np.ma.masked_less_equal(data,Toolbox.rms(data))
        #
        #gaussian filter over already filterted data
        #sigma of half a pixel increases notoriously the SNR when 
        #compared to sigma=1.
        gauss = Toolbox.gaussian_filter(fdata,sigma=.5)
        #RMS for gaussian filter, because we need isolated points to DBSCAN
        fgauss = np.ma.masked_less_equal(gauss,Toolbox.rms(gauss))
        #
        #coordinates of maked arrays
        coo_dat = np.argwhere(data>Toolbox.rms(data))
        coo_gss = np.argwhere(gauss>Toolbox.rms(gauss))
        #other option:
        #aux = fdata.nonzero()
        #tmp = np.array(zip(*aux))
        #
        #dbscan must be performed over coordinates!
        dat_N,dat_label,dat_mask = Toolbox.cluster_dbscan(coo_dat)
        gss_N,gss_label,gss_mask = Toolbox.cluster_dbscan(coo_gss)
        #
        #define plot
        fig = plt.figure(figsize=(10,5))
        fig.suptitle('DWT level {0}, coefficients {1}'.format(level,coeff),
                    fontsize=16)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        #filtered image and filtered gauss
        im1 = ax1.imshow(fgauss,origin='lower',interpolation='none')
        im2 = ax2.imshow(fdata,origin='lower',interpolation='none')
        #
        """
        #clusters in data
        for k in set(dat_label):
            if k == -1: col = 'k'
            else: col = 'g'
            idx1 = np.where(np.logical_and(dat_label==k,dat_mask))
            idx2 = np.where(np.logical_and(dat_label==k,~dat_mask))
            ax2.scatter(coo_dat[idx1][:,1],coo_dat[idx1][:,0],s=60,c=col,
                    marker='o',alpha=0.2,lw=0.2,edgecolor='none')
            ax2.scatter(coo_dat[idx2][:,1],coo_dat[idx2][:,0],s=40,c=col,
                    marker='o',alpha=.2,lw=0.2,edgecolor='none')
        #clusters in gauss filtered data
        for j in set(gss_label):
            if j == -1: col = 'k'
            else: col = 'r'
            idx1 = np.where(np.logical_and(gss_label==j,gss_mask))
            idx2 = np.where(np.logical_and(gss_label==j,~gss_mask))
            ax1.scatter(coo_gss[idx1][:,1],coo_gss[idx1][:,0],s=60,c=col,
                    marker='o',alpha=0.2,lw=0.2,edgecolor='none')
            ax1.scatter(coo_gss[idx2][:,1],coo_gss[idx2][:,0],s=40,c=col,
                    marker='D',alpha=.2,lw=1,edgecolor='w')
        """
        #left,bottom,width,height 0 to 1
        cb1 = fig.add_axes([0.01,0.2,0.03,0.6]) 
        cb2 = fig.add_axes([0.91,0.2,0.03,0.6])
        fig.colorbar(im1,cax=cb1)
        fig.colorbar(im2,cax=cb2)
        #ax3.set_zlim(-1.01, 1.01)
        #ax3.zaxis.set_major_locator(LinearLocator(10))
        #ax3.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        #fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()

    @classmethod
    def decomp(cls,h5table,Nlev=8):
        '''Displays the coeffs for N=2 decomposition
        '''
        #better to ask column by column c_A,c1,c2
        if Nlev == 2:
            cA = [row['c_A'] for row in h5table.iterrows()]
            c1 = [row['c1'] for row in h5table.iterrows()]
            c2 = [row['c2'] for row in h5table.iterrows()]
            fig = plt.figure(figsize=(10,10))
            #base grid
            outer_grid = gridspec.GridSpec(4,4,wspace=0,hspace=0)
            #define subplots without ticks
            ax1 = plt.subplot(outer_grid[:1,:1])
            plt.xticks(())
            plt.yticks(())
            ax1H = plt.subplot(outer_grid[:1,1:2])
            plt.xticks(())
            plt.yticks(())
            ax1V = plt.subplot(outer_grid[1:2,:1])
            plt.xticks(())
            plt.yticks(())
            ax1D = plt.subplot(outer_grid[1:2,1:2])
            plt.xticks(())
            plt.yticks(())
            ax2H = plt.subplot(outer_grid[:2,2:])
            plt.xticks(())
            plt.yticks(())
            ax2V = plt.subplot(outer_grid[2:,:2])
            plt.xticks(())
            plt.yticks(())
            ax2D = plt.subplot(outer_grid[2:,2:])
            plt.xticks(())
            plt.yticks(())
            #plots
            ax1.imshow(cA[0],origin='lower',cmap='jet')
            ax1.set_aspect(1.05)
            ax1H.imshow(c1[0],origin='lower',cmap='jet')
            ax1H.set_aspect(1.05)
            ax1V.imshow(c1[1],origin='lower',cmap='jet')
            ax1V.set_aspect(1.05)
            ax1D.imshow(c1[2],origin='lower',cmap='jet')
            ax1D.set_aspect(1.05)
            ax2H.imshow(c2[0],origin='lower',cmap='jet')
            ax2V.imshow(c2[1],origin='lower',cmap='jet')
            ax2V.set_aspect(1)
            ax2D.imshow(c2[2],origin='lower',cmap='jet')
            ax2D.set_aspect(1)
            #define spacing (not being gridspec)
            plt.subplots_adjust(left=0.01,bottom=0.01,right=0.99,top=0.99)
            plt.show()
        if Nlev == 8:
            print 'Not yet setup for N=8'

    @classmethod
    def histogram(cls,h5table,Nlev=8):
        '''Besides decomposition plot, shows histograms 
        '''
        if Nlev==2:
            cA = [row['c_A'] for row in h5table.iterrows()]
            c1 = [row['c1'] for row in h5table.iterrows()]
            c2 = [row['c2'] for row in h5table.iterrows()]
            #
            fig = plt.figure(figsize=(20,10))
            #base grid
            outer_grid = gridspec.GridSpec(4,8,wspace=0.1,hspace=0.1)
            #subplot for DWT coeffs
            ax1 = plt.subplot(outer_grid[0:1,0:1])
            plt.xticks(())
            plt.yticks(())
            ax1H = plt.subplot(outer_grid[0:1,1:2])
            plt.xticks(())
            plt.yticks(())
            ax1V = plt.subplot(outer_grid[1:2,0:1])
            plt.xticks(())
            plt.yticks(())
            ax1D = plt.subplot(outer_grid[1:2,1:2])
            plt.xticks(())
            plt.yticks(())
            ax2H = plt.subplot(outer_grid[0:2,2:4])
            plt.xticks(())
            plt.yticks(())
            ax2V = plt.subplot(outer_grid[2:4,0:2])
            plt.xticks(())
            plt.yticks(())
            ax2D = plt.subplot(outer_grid[2:4,2:4])
            plt.xticks(())
            plt.yticks(())
            #subplots for histograms
            hx1 = plt.subplot(outer_grid[0:1,4:6])
            hx2 = plt.subplot(outer_grid[1:2,4:6])
            hx3 = plt.subplot(outer_grid[2:3,4:6])
            hx4 = plt.subplot(outer_grid[3:4,4:6])
            hx5 = plt.subplot(outer_grid[0:1,6:8])
            hx6 = plt.subplot(outer_grid[1:2,6:8])
            hx7 = plt.subplot(outer_grid[2:3,6:8])
            #hx8 = plt.subplot(outer_grid[3:4,6:8])
            #plots
            ax1.imshow(cA[0]*cA[0],origin='lower',cmap='jet')
            ax1.set_aspect(1.05)
            ax1H.imshow(c1[0]*c1[0],origin='lower',cmap='jet')
            ax1H.set_aspect(1.05)
            ax1V.imshow(c1[1]*c1[0],origin='lower',cmap='jet')
            ax1V.set_aspect(1.05)
            ax1D.imshow(c1[2]*c1[2],origin='lower',cmap='jet')
            ax1D.set_aspect(1.05)
            ax2H.imshow(c2[0]*c2[0],origin='lower',cmap='jet')
            ax2V.imshow(c2[1]*c2[1],origin='lower',cmap='jet')
            ax2V.set_aspect(1)
            ax2D.imshow(c2[2]*c2[2],origin='lower',cmap='jet')
            ax2D.set_aspect(1)
            #histograms
            #approximation coeffs
            hx4.hist(cA[0].ravel(),bins=100,histtype='stepfilled',
                    color='#43C6DB')
            hx4.set_title('cA')
            #horizontal coeffs
            hx1.hist(c1[0].ravel(),bins=100,histtype='stepfilled',color='gold')
            hx5.hist(c2[0].ravel(),bins=100,histtype='stepfilled',color='gold')
            #vertical coeffs
            hx2.hist(c1[1].ravel(),bins=100,histtype='stepfilled',
                    color='dodgerblue')
            hx6.hist(c2[1].ravel(),bins=100,histtype='stepfilled',
                    color='dodgerblue')
            #diagonal coeffs
            hx3.hist(c1[2].ravel(),bins=100,histtype='stepfilled',
                    color='dimgrey')
            hx7.hist(c2[2].ravel(),bins=100,histtype='stepfilled',
                    color='dimgrey')
            #define spacing (not being gridspec)
            plt.subplots_adjust(left=0.01,bottom=0.01,right=0.99,top=0.99)
            plt.show()


if __name__=='__main__':
    print 'starting'
    #to setup a mask of the outer region of DECam, based on a well behaved flat
    if False:
        Toolbox.binned_mask_npy()
   
    pathBinned = '/work/devel/fpazch/shelf/dwt_Y4Binned/'
    group = 'g3'; print '\n======== {0} ======== \n'.format(group)
    
    fcounter = 0
    for (path,dirs,files) in os.walk(pathBinned):
        for index,item in enumerate(files):   #file is a string 
            if (group+'.h5' in item):
                print '{0} ___ {1}'.format(item,index+1)
                try:
                    H5tab = OpenH5(pathBinned+item)
                    table = H5tab.h5.root.coeff.FP
                    #Graph.decomp(table,Nlev=2)
                    #Graph.histogram(table,Nlev=2)     
                    #TimeSerie.count_and_tab(table,Nlev=2)
                    #wrap2 calls statistical measurements
                    df_tmp = Call.wrap2(item,table,Nlev=2)
                    #concatenate 
                    if fcounter == 0: df_res = df_tmp
                    else: df_res = pd.concat([df_res,df_tmp])
                    df_res.reset_index()
                    fcounter += 1
                finally:
                    #close open instances
                    H5tab.closetab()
                    table.close()
    #save results to file
    try:
        fnout = 'stat_coeffD_dmeyN2_' + group + '.csv'
        df_res.to_csv(fnout,header=True,index=False)
    except:
        print 'Error in write output DF'
