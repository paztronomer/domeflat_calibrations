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
import gc
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
#internal modules
import y4g1g2g3_flatDWT as y4
import scalMask_dflat as scalMask
#desdm modules
import despydb.desdbi as desdbi

class Toolbox():
    @classmethod
    def polygon_area(cls,x_arr,y_arr):
        '''uses shoelace algorithm or Gauss area formula to get the polygon
        area by sucessive triangulation. Sets of coordinates don't need to be
        sorted
        '''
        A = 0.5*np.abs(np.dot(x_arr,np.roll(y_arr,1))-
                    np.dot(y_arr,np.roll(x_arr,1)))
        return A

    @classmethod
    def rms(cls,arr,maskFP=False,baseMask=np.load('fp_BinnedMask.npy')):
        '''returns RMS for ndarray, is maskFP=True then use a scalable mask
        for inner coeffs. maskFP is set to False as default 
        '''
        if maskFP:
            m1 = scalMask.Mask.scaling(baseMask,arr)
            arr = arr[np.where(m1)]
            outrms = np.sqrt(np.mean(np.square(arr.ravel())))
        else:
            outrms = np.sqrt(np.mean(np.square(arr.ravel())))
        return outrms

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
    def cluster_dbscan(cls,points,minsize=1):
        '''friends of friends
        points is a set of coordinates
        DBSCAN.labels_: label for each point, where -1 is set for noise. 
        Includes cores and members not belonging to the core but still on
        the group.
        DBSCAN.core_sample_indices_: list of indices of core samples
        DBSCAN.components_:copy of each core sample found by training
        core_sample_indices_
        Note:
        minsize=1, because we're using binned focal plane images 
        '''
        #here we consider the diagonal of a square of side=1 because of
        #the positioning we're using is the grid of points with unity spacing
        diagonal = np.sqrt(2.)
        #real impact of leaf_size has not been unveiled
        db = sklearn.cluster.DBSCAN(eps=diagonal,min_samples=minsize,
                                leaf_size=30, metric='euclidean').fit(points)
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
    def mask_join(cls,values_arr,threshold,
                baseMask = np.load('fp_BinnedMask.npy')):
        '''method to combine mask coming from inner region of DECam and 
        and an input value
        Returns the joint maks and the mask of only the peaks (still inside
        the inner DECam region)
        '''
        #apply a composite mask: RMS + INNER AREA
        m1 = scalMask.Mask.scaling(baseMask,values_arr)
        #method for m2 is not ok: need to mask the already masked points
        #The inner decam mask must be applied in all steps
        #do it by yourself
        m2 = np.zeros_like(m1,dtype=bool)
        m2[np.where(np.logical_and(m1,values_arr>threshold))] = True
        #two different masks must be joined into one
        #combine both masks
        mout = np.zeros_like(m1,dtype=bool)
        mout[np.where(np.logical_or(m1,m2))] = True
        return mout,m2
    
    @classmethod
    def prev_value_dispers(cls,values):
        '''Previous Version
        Method to estimate dispersion in the values of the clustered points.
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
    def prev_spatial_dispers(cls,grid_shape,points):
        '''Previous Version
        Methods for estimating spatial (angular and radial) dispersion of 
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
    
    @classmethod
    def value_dispers(cls,selected_pts):
        '''Method to estimate dispersion in the values of the clustered points.
        Not related to the position, but to the power.
        For the entropy, we'll use the values as unnormalized probabilities
        Intput: tuple with one tuple per level containing (values,coord,cluster-
        label,label)
        Returns: tuple with values
        '''
        #using only the second level. When improve the mask, add the 1st level
        level = selected_pts[1]
        #for level in selected_pts:
        #we will use only the Diagonal coeffs. When improve the mask other
        #coeffs will be added
        cD = level[2]
        val = cD[0] 
        nm = cD[2] 
        lab = cD[3]
        #mean,median,stdev,rms,min,max,MAD,S,num_clust,num_pts,ratio(other/core)
        x1 = np.mean(val) 
        x2 = np.median(val)
        x3 = np.std(val)
        x4 = Toolbox.rms(val) 
        x5 = np.min(val)
        x6 = np.max(val)
        x7 = np.median(np.abs(val-np.median(val)))
        x8 = scipy.stats.entropy(val.ravel())
        x9 = np.unique(nm[np.where(nm!=-1)]).shape[0]
        x10 = val.shape[0]
        x11 = (lab.shape[0]-lab[np.where(lab==1)].shape[0])/np.float(
            lab[np.where(lab==1)].shape[0])
        #x12 AndersonDarling! 
        return (x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11)
    
    @classmethod 
    def posit_dispers(cls,selected_pts,coeff_shape):
        '''Method to evaluate the spatial behavior of the detected peaks.
        Intput: tuple with one tuple per level containing (values,coord,cluster-
        label,label). Also a tuple containing the shapes for each of the levels
        of the DWT
        Returns: tuple with values
        '''
        #for now only the Diagonal level=2 coeffs will be used. When the masking 
        #be improved, then we will broad to the others.
        level = selected_pts[1]
        cD = level[2]
        coo = cD[1]
        nm = cD[2]
        lab = cD[3]
        dim2 = coeff_shape[1]
        #define the origin as the center of the array, then calculate the 
        #angle for each of the selected peaks, using the origin as reference
        origin = ((dim2[1]-1)/np.float(2),(dim2[0]-1)/np.float(2))
        theta = np.arctan2(coo[:,0]-origin[0],coo[:,1]-origin[1])
        #then define the borders of the angular pieces. To get the real results
        #negative angles must be employed on the range PI...2PI
        ang_step = (np.concatenate([np.linspace(0,np.pi,19),
                                        np.linspace(0.,-np.pi,19)[::-1]]))
        #print ang_step*180/np.pi
        ang_count = []
        for it in xrange(1,len(ang_step)):
            loc = np.where(np.logical_and(theta>ang_step[it-1],
                        theta<=ang_step[it]))
            ang_count.append(theta[loc].shape[0])
        ang_count = np.array(ang_count)
        #mean,median,stdev,rms,min,max,MAD,S
        y1 = np.mean(ang_count) 
        y2 = np.median(ang_count)
        y3 = np.std(ang_count)
        y4 = Toolbox.rms(ang_count) 
        y5 = np.min(ang_count)
        y6 = np.max(ang_count)
        y7 = np.median(np.abs(ang_count-np.median(ang_count)))
        y8 = scipy.stats.entropy(ang_count.ravel())
        '''PENDING: ADD PCA'''
        return (y1,y2,y3,y4,y5,y6,y7,y8) 
        
        
class Screen():
    @classmethod
    def inner_region(cls,h5table,Nlev=2,minCluster=3):
        '''Method to mask the DWT array and get the inner DECam region. Then it
        gives the positions and values of the selected points.
        It its designed for 2 levels of decomposition.
        minCluster: minimal size of a gruop of points to match conditions to be
        a cluster
        Outputs: 
        1) first output is a tuple of tuples containing arrays, one subset 
        for each level, length of the tuple is the number of DWT levels. Each
        level has: cH,cV,cD. And each coefficient has: (value,coordinates,
        cluster-label,label). The output has 3 levels of depth.
        2) second output is a tuple containing the shapes of the arrays, for 
        each DWT level 
        '''
        gc.collect()
        if Nlev != 2:
            raise ValueError('This method is designed for 2 levels of decomp')
        #iterate over both levels to get the horizontal, vertical, and 
        #diagonal coefficients
        cA = [row['c_A'] for row in h5table.iterrows()]
        cHVD1 = [row['c1'] for row in h5table.iterrows()]
        cHVD2 = [row['c2'] for row in h5table.iterrows()]
        #we will use only the diagonal coefficients in our subsequent analysis
        #because the horizontal and vertical need to the re-masked
        #Create a template mask based on c_A. In this array, values of 
        #points outside the DECam region has values=1.
        cA_model = np.ma.getmask(np.ma.masked_greater(cA[0],1.,copy=True))
        #then scale the c_A template to the current shape of the target array,
        #and mask values below RMS (when calculating RMS inside the inner region 
        #area). Iterate over levels/coeffs.
        innRegion = []
        shapeRegion = []
        for level in [cHVD1,cHVD2]:
            shapeRegion.append(level[0].shape)
            auxlevel = []
            for coeff in level:
                maskA,ptsA = Toolbox.mask_join(
                            coeff,
                            Toolbox.rms(coeff,maskFP=True,baseMask=cA_model),
                            baseMask=cA_model)
                #for clustering, must use coordinates of the mask
                coo = np.argwhere(ptsA)
                #extrema values inside inner region
                minN,maxN = np.min(coeff[coo]),np.max(coeff[coo])         
                #Note: for clustering, will use minimal cluster size=1
                clust_N,clust_label,clust_mask = Toolbox.cluster_dbscan(
                                                coo,minsize=minCluster)
                #to locate the points belonging to clusters (cores) from those
                #considered as part of the cluster but not of the core.
                #Noise (non-core points, low-density regions) are also taken 
                #into account
                #Note that noise is not taken into account (clust_label=-1)
                indCore = np.where(np.logical_and(clust_label!=-1,clust_mask))
                indOut = np.where(np.logical_and(clust_label!=-1,~clust_mask))
                indNoise = np.where(clust_label==-1)
                #for use them on input data:
                # coeff[coo[indCore][:,0],coo[indCore][:,1]]
                # coeff[coo[indOut][:,0],coo[indOut][:,1]]
                # to plot coordinates: coo[indCore][:,1],coo[indCore][:,0]
                #save a list of 3 components: coordinates, cluster-labels, and
                #label
                tmp = np.empty_like(clust_mask,dtype='i4')
                tmp[indCore],tmp[indOut],tmp[indNoise] = 1,0,-1 
                auxlevel.append((coeff[coo[:,0],coo[:,1]],coo,clust_label,tmp))
                #save coord of all points with its label (core=1,outlier=0,
                #noise=-1) because its need to know the amount of cores 
                #(grouping) as a measure of density
            innRegion.append(tuple(auxlevel))
        return tuple(innRegion),tuple(shapeRegion)


class OpenH5():
    '''class oriented to open and close a pytables instance. It's not the same
    to close the instance here inside the same class than outside
    '''
    def __init__(self,fname):
        self.h5 = tables.open_file(fname,driver='H5FD_CORE')
    
    def closetab(self):
        self.h5.close()


class TimeSerie():
    '''when the usuable statistics results, make a time series
    '''
    pass
    
    
class Graph():
    @classmethod
    def filter_plot(cls,dataAVG,data1,data2,coeff,fname,minimalSize=1):
        '''Method for plotting on the same frame the DWT for level 1 and 2,
        belonging to Horizontal, Vertical and Diagonal coefficients.
        Only values inside the inner region od DECam are considered: the mask
        is constructed from c_A selecting values above 1.
        Then the borders are displayed and the inner region is applied through
        mas_join method. Values inside the mask are selected if they are above 
        the RMS of the values of the pixels inside the mask.
        With the above selected points we perform clustering based in DBSCAN,
        using a minimal cluster size of 1. This value is an input.
        Inputs are:
        - data1,data2: arrays of DWT for level=1,2 
        - dataAVG: array of c_A, level=1
        - coeff: string H,V,D 
        - fname: filename of the DWT file
        - minimalSize: minimal size of the cluster to be taken into account. 

        #Erased block:
        #gaussian filter over already filterted data
        #sigma=0.5 pixel increases notoriously the SNR when 
        #compared to sigma=1.
        gauss = Toolbox.gaussian_filter(fdata,sigma=.5) 
        #RMS for gaussian filter, because we need isolated points to DBSCAN
        fgauss = np.ma.masked_less_equal(gauss,Toolbox.rms(gauss))
        coo_gss = np.argwhere(gauss>Toolbox.rms(gauss))
        gss_N,gss_label,gss_mask = Toolbox.cluster_dbscan(coo_gss)
        '''
        #zero-padding is only available for Y4

        #create a template mask based on c_A, where pixel values > 1.
        cA_model = np.ma.getmask(np.ma.masked_greater(dataAVG,1.,copy=True))

        #then use the template to scale it to current size and mask values 
        #below RMS (when calculating RMS inside the unmasked area). Make it for
        #levels 1 and 2.
        mAvg1,pAvg1 = Toolbox.mask_join(
                    data1,
                    Toolbox.rms(data1,maskFP=True,baseMask=cA_model),
                    baseMask=cA_model)
        mAvg2,pAvg2 = Toolbox.mask_join(
                    data2,
                    Toolbox.rms(data2,maskFP=True,baseMask=cA_model),
                    baseMask=cA_model)
        #for clustering, must use coordinates of the mask
        coo1 = np.argwhere(pAvg1)
        coo2 = np.argwhere(pAvg2)
        #extrema values inside inner region
        minN1,maxN1 = np.min(data1[coo1]),np.max(data1[coo1])
        minN2,maxN2 = np.min(data2[coo2]),np.max(data2[coo2])
        
        #Note: for clustering, will use minimal cluster size=1
        gc.collect()
        d1_N,d1_label,d1_mask = Toolbox.cluster_dbscan(coo1,minsize=minimalSize)
        gc.collect()
        d2_N,d2_label,d2_mask = Toolbox.cluster_dbscan(coo2,minsize=minimalSize)
        
        #for plotting usage
        veil1 = np.ma.masked_where(~mAvg1,data1)
        veil2 = np.ma.masked_where(~mAvg2,data2)
        
        plt.close('all')
        fig = plt.figure(figsize=(18,9))
        fig.suptitle('DWT level 1-2, coefficients {0}\n{1}'.format(coeff,fname),
                    fontsize=16)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        #inner and outer region
        im1 = ax1.imshow(veil1,origin='lower',interpolation='none',
                        cmap='gray',alpha=0.08)
        im2 = ax2.imshow(veil2,origin='lower',interpolation='none',
                        cmap='gray',alpha=0.08)
        
        #Clusters and outliers. Colorcode is DWT transform value. Level=1
        idx1 = np.where(np.logical_and(d1_label!=-1,d1_mask))
        idx2 = np.where(np.logical_and(d1_label!=-1,~d1_mask))
        p1 = ax1.scatter(coo1[idx1][:,1],coo1[idx1][:,0],s=30,
                        c=data1[coo1[idx1][:,0],coo1[idx1][:,1]],
                        vmin=minN1,vmax=maxN1,cmap='jet',
                        marker='o',alpha=0.8,lw=0.8,edgecolor='none')
        ax1.scatter(coo1[idx2][:,1],coo1[idx2][:,0],s=60,
                c=data1[coo1[idx2][:,0],coo1[idx2][:,1]],
                vmin=minN1,vmax=maxN1,cmap='jet',
                marker='D',alpha=0.8,lw=0.8,edgecolor='none')
        #Level=2. Note we reuse idx1,2
        idx1 = np.where(np.logical_and(d2_label!=-1,d2_mask))
        idx2 = np.where(np.logical_and(d2_label!=-1,~d2_mask))
        p2 = ax2.scatter(coo2[idx1][:,1],coo2[idx1][:,0],s=30,
                        c=data2[coo2[idx1][:,0],coo2[idx1][:,1]],
                        vmin=minN2,vmax=maxN2,cmap='jet',
                        marker='o',alpha=0.8,lw=0.2,edgecolor='none')
        ax2.scatter(coo2[idx2][:,1],coo2[idx2][:,0],s=60,
                c=data2[coo2[idx2][:,0],coo2[idx2][:,1]],
                vmin=minN2,vmax=maxN2,cmap='jet',
                marker='D',alpha=0.8,lw=0.2,edgecolor='none')

        #left,bottom,width,height 0 to 1
        cb1 = fig.add_axes([0.01,0.2,0.03,0.6]) 
        cb2 = fig.add_axes([0.91,0.2,0.03,0.6])
        fig.colorbar(p1,cax=cb1)
        fig.colorbar(p2,cax=cb2)
        ax1.set_xlim([0,data1.shape[1]-1])
        ax1.set_ylim([0,data1.shape[0]-1])
        ax2.set_xlim([0,data2.shape[1]-1])
        ax2.set_ylim([0,data2.shape[0]-1])
        plt.show()
        return True

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


class Call():
    @classmethod
    def wrap1(cls,h5table,fname,Nlev=2):
        '''Call the method for clustering.DBSCAN different coefficients,
        through the plotting method
        '''
        if Nlev == 2:
            cA = [row['c_A'] for row in h5table.iterrows()]
            c1 = [row['c1'] for row in h5table.iterrows()]
            c2 = [row['c2'] for row in h5table.iterrows()]
            #filter process
            print '\n\t=== USE |coeff|^2 ==='
            if True:
                print '\t=== performing over c_H ==='
                #for level 1, c1[]
                #pos1H = Graph.filter_plot(c1[0]*c1[0],(1,'H'))
                #pos2H = Graph.filter_plot(c2[0]*c2[0],(2,'H'))
                Graph.filter_plot(cA[0],c1[0]*c1[0],c2[0]*c2[0],'H',fname)
            if True:
                print '\t=== performing over c_V ==='
                #for level 1, c1[]
                #pos1V = Graph.filter_plot(c1[1]*c1[1],(1,'V'))
                #pos2V = Graph.filter_plot(c2[1]*c2[1],(2,'V'))
                Graph.filter_plot(cA[1],c1[1]*c1[1],c2[1]*c2[1],'V',fname)
            if True:
                print '\t=== performing over c_D ==='
                #for level 1, c1[]
                #pos1D = Graph.filter_plot(c1[2]*c1[2],(1,'D'))
                #pos2D = Graph.filter_plot(c2[2]*c2[2],(2,'D'))
                Graph.filter_plot(cA[2],c1[2]*c1[2],c2[2]*c2[2],'D',fname)
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
            val_stat = Toolbox.prev_value_dispers(sel_val)
            spa_stat = Toolbox.prev_spatial_dispers(data.shape,sel_coo)
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

    @classmethod
    def wrap3(cls,h5table,table_nm):
        '''Wrapper for mask the inner region of DECam. It its designed ONLY for
        Diagonal coefficients of level=2. When the mask be improved we can 
        switch to other coeffs as well.
        Inputs:
        - h5table: H5 tables of the DWT coeffs
        - table_nm: nmae of the H5 file, from which construct a temporal method
         to get info
        Returns: dataframe with values and column names
        '''
        #first, get the info for the filename. This must be replaced by info on 
        #the header of the HDF5 tables. Then this is only a temporal solution
        fits = table_nm[:table_nm.rfind('_')] + '_compare-dflat-binned-fp.fits'
        #DB information: temporal solution while 
        toquery = "select m.nite,p.reqnum,p.id,f.expnum,m.band, \
                f.factor,f.rms, f.worst \
                from flat_qa f,pfw_attempt p,miscfile m \
                where f.filename='{0}' and m.filename=f.filename and \
                m.pfw_attempt_id=p.id".format(fits)
        #outdtype = ['a80','f4','f4','f4','i4','a10','i4','i4','a100']
        desfile = os.path.join(os.getenv('HOME'),'.desservices.ini')
        section = 'db-desoper'
        dbi = desdbi.DesDbi(desfile,section)
        cursor = dbi.cursor()
        cursor.execute(toquery)
        rows = cursor.fetchall()
        nite = np.int(rows[0][0])
        reqnum,pfw_id,expnum,band,factor,rms,worst = [i for i in rows[0][1:]]
        aux = [nite,reqnum,pfw_id,expnum,band,factor,rms,worst]
        #the statistics of the values and positions for the RMS-selected peaks
        sel,frame = Screen.inner_region(table)
        statVal = Toolbox.value_dispers(sel)
        statPos = Toolbox.posit_dispers(sel,frame)
        #column names where fq. stands for oper.flat_qa origin, v. stands for
        #values, and p. stands for positions
        col = ['nite','reqnum','pfw','expnum','band','fq.factor']
        col += ['fq.rms','fq.worst']   
        col += ['v.avg','v.med','v.std','v.rms','v.min','v.max','v.mad','v.S']
        col += ['v.Ncl','v.Npt','v.Nratio']
        col += ['p.avg','p.med','p.std','p.rms','p.min','p.max','p.mad','p.S'] 
        stat = aux + list(statVal) + list(statPos) 
        df = pd.DataFrame([stat,],columns=col)
        return df


if __name__=='__main__':
    print 'starting'
    #to setup a mask of the outer region of DECam, based on a well behaved flat
    if False:
        Toolbox.binned_mask_npy()
    
    #this is the path to the zero-padded DWT tables
    pathBinned = '/work/devel/fpazch/shelf/dwt_dmeyN2/'
    band = 'u'
    
    '''TO VISUALIZE BY EXPNUM RANGE
    '''
    if False:
        expnum_range = range(601308,601394+1)#(606738,606824+1)
        opencount = 0
        for (path,dirs,files) in os.walk(pathBinned):
            for index,item in enumerate(files):   #file is a string
                expnum = int(item[1:item.find('_')])
                #if not ('_r2625' in item):
                #    print item
                if ('_'+band+'_' in item) and (expnum in expnum_range):
                    opencount += 1
                    print '{0} ___ {1} Iter: {2}'.format(item,index+1,opencount)
                    try:
                        H5tab = OpenH5(pathBinned+item)
                        table = H5tab.h5.root.coeff.FP
                        Call.wrap1(table,item)
                        #Graph.decomp(table,Nlev=2)
                        #Graph.histogram(table,Nlev=2)     
                        #TimeSerie.count_and_tab(table,Nlev=2)
                    finally:
                        #close open instances
                        H5tab.closetab()
                        table.close()

    '''TO SAVE STATISTICS
    '''

    '''NEW TRY
    '''
    #remember to change for each year
    savepath = '/work/devel/fpazch/shelf/stat_dmeyN2/' 
    savename = 'reStat_' + band + '_.csv' 
    if True:
        filler = 0
        for (path,dirs,files) in os.walk(pathBinned):
            for index,item in enumerate(files):   #file is a string
                if ('_'+band+'_' in item):
                    try:
                        H5tab = OpenH5(pathBinned+item)
                        table = H5tab.h5.root.coeff.FP
                        print '\t',item
                        tmp = Call.wrap3(table,item)
                        if filler == 0: df_res = tmp
                        else: df_res = pd.concat([df_res,tmp])
                        df_res.reset_index()
                        filler += 1
                        #sel,frame = Screen.inner_region(table)
                        #Toolbox.value_dispers(sel)
                        #Toolbox.posit_dispers(sel,frame)
                    finally:
                        #close open instances
                        H5tab.closetab()
                        table.close()
        #write oout the table of results
        df_res.to_csv(savename,index=False,header=True)
    
    '''PREVIOUS TRY
    '''
    if False:
        #to make statistics and save to file
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
            fnout = 'stat_cD_' + group + '.csv'
            df_res.to_csv(fnout,header=True,index=False)
        except:
            print 'Error in write output DF'
