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
    def rms(cls,arr,maskFP=False,baseMask=None):
        '''returns RMS for ndarray, is maskFP=True then use a scalable mask
        for inner coeffs. maskFP is set to False as default 
        np.load('fp_BinnedMask.npy')
        '''
        if maskFP:
            m1 = scalMask.Mask.scaling(baseMask,arr)
            arr = arr[np.where(m1)]
        outrms = np.sqrt(np.mean(np.square(arr.ravel())))
        return outrms

    @classmethod
    def uncert(cls,arr,maskFP=False,baseMask=None):
        '''calculates the uncertain in a parameter, as usually used in
        physics
        '''
        if maskFP:
            M = scalMask.Mask.scaling(baseMask,arr)
            arr = arr[np.where(M)]
        ux = np.sqrt(np.mean(np.square(arr.ravel())) + 
                    np.square(np.mean(arr.ravel()))) 
        return ux

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
    def value_dispers(cls,sel_pts):
        '''Method to estimate dispersion in the values of the clustered points.
        Not related to the position, but to the power.
        For the entropy, we'll use the values as unnormalized probabilities
        Intput: tuple with one tuple per level containing 3 elements each level)
        Inside each of the 3 elements (H,V,and D) are:
        1) first element
        - values of RMS and uncertainty for all the coeffs inside DECam,
        - coordinates of selected peaks
        - values of the selected peaks
        - cluster label
        - class of each point (1/0/-1 : core/out/noise)
        Level1-----c_H------RMS/uncert/coord/values/label/class
                |--c_V------RMS/uncert/coord/values/label/class
                |--c_D------RMS/uncert/coord/values/label/class
        Level2-----c_H------RMS/uncert/coord/values/label/class
                |--c_V------RMS/uncert/coord/values/label/class
                |--c_D------RMS/uncert/coord/values/label/class
        2) Also a tuple containing the shapes for each of the levels
        of the DWT
        Returns: tuple with values for each level-coeff combination plus the 
        RMS and uncertainty for all the coeffs inside the mask
        '''
        #number of entries in the output tuple is important for non peak case
        out = []
        nitem = 16
        labcoeff = ['H','V','D']
        for idx1,level in enumerate(sel_pts):
            for idx2,coeff in enumerate(level):
                x0 = idx1 + 1
                x1 = labcoeff[idx2]
                if np.all(np.isnan(coeff[2])):
                    out.append(tuple([x0,x1] + [np.nan]*(nitem-2)))
                elif coeff[3].shape[0] < 2:
                    #for single peak: level,coeff label,and Npeak are non NaN
                    out.append(tuple([x0,x1,coeff[3][0]] + [np.nan]*(nitem-5) +
                             [1,np.nan]))
                else:
                    #level,coeff_label,RMS all,uncert all,mean,median,stdev,rms,
                    #uncert,min,max,MAD,S,num_clust,num_pts,ratio(other/core)
                    x2 = coeff[0]
                    x3 = coeff[1]
                    x4 = np.mean(coeff[3]) 
                    x5 = np.median(coeff[3])
                    x6 = np.std(coeff[3])
                    x7 = Toolbox.rms(coeff[3]) 
                    x8 = Toolbox.uncert(coeff[3])
                    x9 = np.min(coeff[3])
                    x10 = np.max(coeff[3])
                    x11 = np.median(np.abs(coeff[3] - np.median(coeff[3])))
                    x12 = scipy.stats.entropy(coeff[3].ravel())
                    x13 = np.unique(coeff[4][np.where(coeff[4]!=-1)]).shape[0]
                    x14 = coeff[3].shape[0]
                    try:
                        x15 = (coeff[5].shape[0] - 
                            coeff[5][np.where(coeff[5]==1)].shape[0])/np.float(
                            coeff[5][np.where(coeff[5]==1)].shape[0])
                    except:
                        #in case division by zero
                        x15 = np.nan
                    out.append((x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,
                            x14,x15))
                    '''toy histogram
                    n,bins,patches = plt.hist(coeff[3],coeff[3].shape[0]/10, 
                        normed=False,facecolor='green',alpha=0.75)
                    plt.show()
                    '''
        return out
    
    @classmethod 
    def posit_dispers(cls,sel_pts,coeff_shape):
        '''Method to evaluate the spatial behavior of the detected peaks.
        Intput: tuple with one tuple per level containing 3 elements each level.
        Inside each of the 3 elements (H,V,and D) are:
        1) first element
        - values of RMS and uncertainty for all the coeffs inside DECam,
        - coordinates of selected peaks
        - values of the selected peaks
        - cluster label
        - class of each point (1/0/-1 : core/out/noise)
        Level1-----c_H------RMS/uncert/coord/values/label/class
                |--c_V------RMS/uncert/coord/values/label/class
                |--c_D------RMS/uncert/coord/values/label/class
        Level2-----c_H------RMS/uncert/coord/values/label/class
                |--c_V------RMS/uncert/coord/values/label/class
                |--c_D------RMS/uncert/coord/values/label/class
        2) Also a tuple containing the shapes for each of the levels
        of the DWT.
        Returns: tuple with values for each level-coeff combination
        '''
        out = []
        nitem = 11
        labcoeff = ['H','V','D']
        #define the borders of the angular pieces. To get 
        #real results, negative angles must be employed on the range 
        #PI...2PI. Steps of 10 degrees
        ang_step = (np.concatenate([np.linspace(0,np.pi,19),
                np.linspace(0.,-np.pi,19)[::-1]]))
        for idx1,level in enumerate(sel_pts):
            dim = coeff_shape[idx1]
            for idx2,coeff in enumerate(level):
                y0 = idx1+1
                y1 = labcoeff[idx2]
                if np.all(np.isnan(coeff[2])):
                    out.append(tuple([y0,y1] + [np.nan]*(nitem-2)))
                elif (coeff[3].shape[0] < 2):
                    out.append(tuple([y0,y1,coeff[3][0]] + [np.nan]*(nitem-3)))
                else:
                    #define the origin as the center of the array, then 
                    #calculate the angle for each of the selected peaks, using 
                    #the origin as reference
                    coo = coeff[2]
                    origin = ((dim[1]-1)/np.float(2),(dim[0]-1)/np.float(2))
                    theta = np.arctan2(coo[:,0]-origin[0],coo[:,1]-origin[1])
                    ang_count = []
                    for it in xrange(1,len(ang_step)):
                        loc = np.where(np.logical_and(theta>ang_step[it-1],
                                    theta<=ang_step[it]))
                        ang_count.append(theta[loc].shape[0])
                    ang_count = np.array(ang_count)
                    #we have the counts for each angular interval of 10 degrees,
                    #and on this we will perform statistics
                    #mean,median,stdev,rms,min,max,MAD,S
                    y2 = np.mean(ang_count) 
                    y3 = np.median(ang_count)
                    y4 = np.std(ang_count)
                    y5 = Toolbox.rms(ang_count) 
                    y6 = Toolbox.uncert(ang_count)
                    y7 = np.min(ang_count)
                    y8 = np.max(ang_count)
                    y9 = np.median(np.abs(ang_count-np.median(ang_count)))
                    y10 = scipy.stats.entropy(ang_count.ravel())
                    out.append((y0,y1,y2,y3,y4,y5,y6,y7,y8,y9,y10))
                    '''PENDING: ADD PCA ORIENTATION OF PPAL 3 VECTORS'''
        return out

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
    def band_expnum(cls,niterange):
        '''For a niterange, returns the maximum/minumum in EXPNUM and the 
        BAND for the range
        '''
        q1 = "select distinct(m.band) from flat_qa q,miscfile m "
        q1 += "where q.filename=m.filename and "
        q1 += "m.nite>={0} and m.nite<={1} order by m.band".format(
            niterange[0],niterange[1])
        q2 = "select min(q.expnum),max(q.expnum) from flat_qa q,miscfile m "
        q2 += "where q.filename=m.filename and "
        q2 += "m.nite>={0} and m.nite<={1} order by m.band".format(
            niterange[0],niterange[1])
        dtype1 = ['a10']
        dtype2 = ['i4','i4']
        tab1 = Toolbox.dbquery(q1,dtype1)
        tab2 = Toolbox.dbquery(q2,dtype2)
        return tab1['band'][:],[tab2['min(q.expnum)'],tab2['max(q.expnum)']]
            

class Screen():
    @classmethod
    def inner_region(cls,h5table,Nlev=2,minCluster=3):
        '''Method to mask the DWT array and get the inner DECam region. Then it
        gives the positions and values of the selected points.
        It its designed for 2 levels of decomposition.
        minCluster: minimal size of a gruop of points to match conditions to be
        a cluster
        Note: for level-1 we will use horizontal and vertical masks for c_H and 
        c_V. For level-2 and for diagonal, we will use c_A based mask.
        Here we must also consider the scenario of no peak detection, as well
        as the statistics of all the coefficients inside the selected mask.
        Outputs: 
        1) first output is a tuple of tuples containing arrays, one subset 
        for each level, length of the tuple is the number of DWT levels. Each
        level has: cH,cV,cD. And each coefficient has: (value,coordinates,
        cluster-label,label). The output has 3 levels of depth.
        Level1-----c_H------RMS/uncert/coord/values/label/class
                |--c_V------RMS/uncert/coord/values/label/class
                |--c_D------RMS/uncert/coord/values/label/class
        Level2-----c_H------RMS/uncert/coord/values/label/class
                |--c_V------RMS/uncert/coord/values/label/class
                |--c_D------RMS/uncert/coord/values/label/class
        2) second output is a tuple containing the shapes of the arrays, for 
        each DWT level. This will allow us to analyze the spatial distribution
        of peaks
        '''
        gc.collect()
        if Nlev != 2:
            raise ValueError('This method is designed for 2 levels of decomp')
        #iterate over both levels to get the horizontal, vertical, and 
        #diagonal coefficients
        cA = [row['c_A'] for row in h5table.iterrows()]
        cHVD1 = [row['c1'] for row in h5table.iterrows()]
        cHVD2 = [row['c2'] for row in h5table.iterrows()]
        #set of masks, made inside scalMask_dflat script
        mask_A = np.load('/work/devel/fpazch/shelf/cA_mask.npy')
        mask_H = np.load('/work/devel/fpazch/shelf/cH_mask.npy')
        mask_V = np.load('/work/devel/fpazch/shelf/cV_mask.npy')
        mask_D = np.load('/work/devel/fpazch/shelf/cD_mask.npy')
        #then scale the template mask to the current shape of the target array,
        #and mask values below RMS. Iterate over levels/coeffs.
        innRegion = []
        shapeRegion = []
        for ind1,level in enumerate([cHVD1,cHVD2]):
            shapeRegion.append(level[0].shape)
            #this list (later a tuple) will harbor 3 coeffs results per level
            auxlevel = []
            for ind2,coeff in enumerate(level):
                #discriminate for c_H and c_V masks, level:1
                if (ind1 == 0 and ind2 == 0):
                    model = mask_H
                elif (ind1 == 0 and ind2 == 1):
                    model = mask_V
                else:
                    model = mask_A
                #rms and uncertainty of all coefficients
                var1 = Toolbox.rms(coeff,maskFP=True,baseMask=model)
                var2 = Toolbox.uncert(coeff,maskFP=True,baseMask=model)
                #get the scaled mask and the peaks within the condition
                maskN,ptsN = Toolbox.mask_join(
                            coeff,
                            1.*Toolbox.rms(coeff,maskFP=True,baseMask=model),
                            baseMask=model)
                #for clustering, must use coordinates of the mask
                coo = np.argwhere(ptsN)
                #as absence of peaks is a possibility, we must account for it
                #remember that minimal shape of coordinates is [0,2]
                if (coo.shape[0] > 0):
                    #extrema values of the masked peaks
                    minN,maxN = np.min(coeff[coo]),np.max(coeff[coo])         
                    clust_N,clust_label,clust_mask = Toolbox.cluster_dbscan(
                                                    coo,minsize=minCluster)
                    #output from DBSCAN are: 
                    # - number of clusters,
                    # - labels of each input coordinate, 
                    # - mask to separate 3 sets (core/outer/noise) and locate 
                    # the points belonging to clusters (cores) from those
                    # considered as part of the cluster but not of the core.
                    # Noise (non-core points, low-density regions) are also 
                    # taken into account
                    iCore = np.where(np.logical_and(clust_label!=-1,clust_mask))
                    iOut = np.where(np.logical_and(clust_label!=-1,~clust_mask))
                    iNoise = np.where(clust_label==-1)
                    #for use the above indices on input data:
                    # coeff[coo[iCore][:,0],coo[iCore][:,1]]
                    # coeff[coo[iOut][:,0],coo[iOut][:,1]]
                    #to plot coordinates: coo[iCore][:,1],coo[iCore][:,0]
                    #save a list of 3 components: coordinates, cluster-labels, 
                    #and label
                    #
                    #create a list resuming the class of each input point
                    tmp = np.empty_like(clust_mask,dtype='i4')
                    tmp[iCore],tmp[iOut],tmp[iNoise] = 1,0,-1
                    #save a list of tuples where each entry has:
                    # - RMS and uncertainty of all the coeffs in DECam
                    # - coordinate of the input point
                    # - DWT value of the input point
                    # - clustering label for each input point (-1,0,1,2,...)
                    # - clustering class for each input point (1,0,-1)
                    #save coord of all points with its label (core=1,outlier=0,
                    #noise=-1) because its need to know the amount of cores 
                    #(grouping) as a measure of density
                    auxlevel.append((var1,var2,
                                coo,coeff[coo[:,0],coo[:,1]],
                                clust_label,tmp))
                else:
                    #if no peak is selected, then save a tuple of NaN
                    auxlevel.append((var1,var2,np.nan,np.nan,np.nan,np.nan))
            #this list (later tuple) will has 2 items (2 levels of DWT)
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
    def filter_plot(cls,mask_cA,mask_cH,mask_cV,data1,data2,coeff,fname,
                minimalSize=3):
        '''Method for plotting on the same frame the DWT for level 1 and 2,
        belonging to Horizontal, Vertical and Diagonal coefficients.
        Only values inside the inner region od DECam are considered: the masks
        are constructed from c_A selecting values above 1, and modifying it 
        to better suit horizontal and vertical coeffs.
        Then the borders are displayed and the inner region is applied through
        mask_join method. Values inside the mask are selected if they are above 
        the RMS of the values of the pixels inside the mask.
        With the above selected points we perform clustering based in DBSCAN,
        using a minimal cluster size of 3. This value is an input.
        Inputs are:
        - data1,data2: arrays of DWT for level=1,2 
        - mask_cA: mask constructed outside this script. Based on c_A, 
        and correcting some pixels in the border by hand. Boolean array.
        - mask_cH: mask made by hand, based on mask_cA but adding 2 rows of
        pixels on each horizontal edge.
        - mask_cV: mask made by hand, based on mask_cA but adding 2 columns of
        pixels on each vertical edge.
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
        #erode the boolean array and produce same dtype
        #will be used? for diagonal coeffs?
        cA_eros = scipy.ndimage.binary_erosion(mask_cA,
                                            iterations=1).astype(bool)
        if coeff.upper() == 'H': inn_mask = mask_cH
        elif coeff.upper() == 'V': inn_mask = mask_cV
        elif coeff.upper() == 'D': inn_mask = mask_cA
        else: raise ValueError('Coeff string must be H,V or D')
       
        #just for testing values
        #inner mask is only for level:1 because of the lenght scale it traces
        rms1 = Toolbox.rms(data1,maskFP=True,baseMask=mask_cA)
        rms2 = Toolbox.rms(data2,maskFP=True,baseMask=mask_cA)
        aux_m1 = scalMask.Mask.scaling(mask_cA,data1)
        std1 = np.std(data1[np.where(aux_m1)])
        avg1 = np.mean(data1[np.where(aux_m1)])
        aux_m2 = scalMask.Mask.scaling(mask_cA,data2)
        std2 = np.std(data2[np.where(aux_m2)])
        avg2 = np.mean(data2[np.where(aux_m2)])
        print 'RMS ',rms1,rms2,'STD ',std1,std2,'AVG ',avg1,avg2
        print ('uncert: ',np.sqrt(np.square(rms1)-np.square(avg1)),
            np.sqrt(np.square(rms2)-np.square(avg2)),'\n\n')

        #for level:1 use Horizontal and Vertical masks 
        #for level:2 use the Average mask for all the coeffcients
        thres1 = 1.*Toolbox.rms(data1,maskFP=True,baseMask=inn_mask)
        thres2 = 1.*Toolbox.rms(data2,maskFP=True,baseMask=mask_cA)
        mAvg1,pAvg1 = Toolbox.mask_join(data1,thres1,baseMask=inn_mask)
        mAvg2,pAvg2 = Toolbox.mask_join(data2,thres2,baseMask=mask_cA)
        
        #for clustering, must use coordinates of the peaks inside the mask
        coo1 = np.argwhere(pAvg1)
        coo2 = np.argwhere(pAvg2)
        #extrema values inside inner region
        '''deal with no peaks scenario'''
        #conditions for non-peaks above RMS
        if (data1[coo1].ravel().shape[0] > 0): 
            minN1,maxN1 = np.min(data1[coo1]),np.max(data1[coo1])
        else:
            minN1,maxN1 = -1,0
        if (data2[coo2].ravel().shape[0] > 0):
            minN2,maxN2 = np.min(data2[coo2]),np.max(data2[coo2])
        else:
            minN2,maxN2 = -1,0
        #we will distinguish between cores, points belonging to clusters but 
        #not to cores, and noise. That is the reason we're using a cluster 
        #size > 1. So, cluster size=3
        '''deal with no peaks scenario'''
        if (data1[coo1].ravel().shape[0] > 0): 
            gc.collect()
            d1_N,d1_label,d1_mask = Toolbox.cluster_dbscan(coo1,
                                                        minsize=minimalSize)
        if (data2[coo2].ravel().shape[0] > 0):
            gc.collect()
            d2_N,d2_label,d2_mask = Toolbox.cluster_dbscan(coo2,
                                                        minsize=minimalSize)
        #for plotting usage
        #veil1 = np.ma.masked_where(~mAvg1,data1)
        #veil2 = np.ma.masked_where(~mAvg2,data2)
        #then setup plots for both levels at the same time
        plt.close('all')
        fig = plt.figure(figsize=(18,9))
        fig.suptitle('DWT level 1-2, coefficients {0}\n{1}'.format(coeff,fname),
                    fontsize=16)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        #masking inner and outer region
        im1 = ax1.imshow(aux_m1,origin='lower',interpolation='none',
                        cmap='gray',alpha=.1)
        im2 = ax2.imshow(aux_m2,origin='lower',interpolation='none',
                        cmap='gray',alpha=.1)
        #Clusters and outliers. Colorcode is DWT transform value. Level=1
        if (data1[coo1].ravel().shape[0] > 0): 
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
            #to setup the colorboxes and limits
            #left,bottom,width,height 0 to 1
            cb1 = fig.add_axes([0.01,0.2,0.03,0.6]) 
            fig.colorbar(p1,cax=cb1)
        #Level=2. Note we reuse idx1,2
        if (data2[coo2].ravel().shape[0] > 0):
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
            #to setup the colorboxes and limits
            #left,bottom,width,height 0 to 1
            cb2 = fig.add_axes([0.91,0.2,0.03,0.6])
            fig.colorbar(p2,cax=cb2)
        ax1.set_xlim([0,data1.shape[1]-1])
        ax1.set_ylim([0,data1.shape[0]-1])
        ax2.set_xlim([0,data2.shape[1]-1])
        ax2.set_ylim([0,data2.shape[0]-1])
        plt.show()
        
        if False:
            '''Below is the working code on which RMS and selection were made
            inside a inner mask'''
            #erode the boolean array and produce same dtype
            #will be used? for diagonal coeffs?
            cA_eros = scipy.ndimage.binary_erosion(mask_cA,
                                                iterations=1).astype(bool)
            if coeff.upper() == 'H': inn_mask = mask_cH
            elif coeff.upper() == 'V': inn_mask = mask_cV
            elif coeff.upper() == 'D': inn_mask = mask_cA
            else: raise ValueError('Coeff string must be H,V or D')
            rms1 = Toolbox.rms(data1,maskFP=True,baseMask=inn_mask)
            rms2 = Toolbox.rms(data2,maskFP=True,baseMask=inn_mask)
            aux_m1 = scalMask.Mask.scaling(inn_mask,data1)
            std1 = np.std(data1[np.where(aux_m1)])
            avg1 = np.mean(data1[np.where(aux_m1)])
            aux_m2 = scalMask.Mask.scaling(inn_mask,data2)
            std2 = np.std(data2[np.where(aux_m2)])
            avg2 = np.mean(data2[np.where(aux_m2)])
            print 'RMS ',rms1,rms2,'STD ',std1,std2,'AVG ',avg1,avg2
            print ('uncert: ',np.sqrt(np.square(rms1)-np.square(avg1)),
                np.sqrt(np.square(rms2)-np.square(avg2)),'\n\n')
            thres1 = Toolbox.rms(data1,maskFP=True,baseMask=inn_mask)
            thres2 = Toolbox.rms(data2,maskFP=True,baseMask=inn_mask)
            mAvg1,pAvg1 = Toolbox.mask_join(data1,thres1,baseMask=inn_mask)
            mAvg2,pAvg2 = Toolbox.mask_join(data2,thres2,baseMask=inn_mask)
            #for clustering, must use coordinates of the peaks inside the mask
            coo1 = np.argwhere(pAvg1)
            coo2 = np.argwhere(pAvg2)
            #extrema values inside inner region
            minN1,maxN1 = np.min(data1[coo1]),np.max(data1[coo1])
            minN2,maxN2 = np.min(data2[coo2]),np.max(data2[coo2])
            #we will distinguish between cores, points belonging to clusters but 
            #not to cores, and noise. That is the reason we're using a cluster 
            #size > 1. So, cluster size=3
            gc.collect()
            d1_N,d1_label,d1_mask = Toolbox.cluster_dbscan(coo1,minsize=minimalSize)
            gc.collect()
            d2_N,d2_label,d2_mask = Toolbox.cluster_dbscan(coo2,minsize=minimalSize)
            #for plotting usage
            #veil1 = np.ma.masked_where(~mAvg1,data1)
            #veil2 = np.ma.masked_where(~mAvg2,data2)
            #then setup plots for both levels at the same time
            plt.close('all')
            fig = plt.figure(figsize=(18,9))
            fig.suptitle('DWT level 1-2, coefficients {0}\n{1}'.format(coeff,fname),
                        fontsize=16)
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            #masking inner and outer region
            im1 = ax1.imshow(aux_m1,origin='lower',interpolation='none',
                            cmap='gray',alpha=.1)
            im2 = ax2.imshow(aux_m2,origin='lower',interpolation='none',
                            cmap='gray',alpha=.1)
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
            #to setup the colorboxes and limits
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

        if False:
            '''Below is the working method without erosion'''
            #create a template mask based on c_A, where pixel values > 1.
            cAtmp = np.ma.getmask(np.ma.masked_greater(dataAVG,1.,copy=True))
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
            #Note: for clustering, will use minimal cluster size=3
            gc.collect()
            d1_N,d1_label,d1_mask = Toolbox.cluster_dbscan(coo1,minsize=minimalSize)
            gc.collect()
            d2_N,d2_label,d2_mask = Toolbox.cluster_dbscan(coo2,minsize=minimalSize)
            #for plotting usage
            veil1 = np.ma.masked_where(~mAvg1,data1)
            veil2 = np.ma.masked_where(~mAvg2,data2)
            #then setup plots for both levels at the same time
            plt.close('all')
            fig = plt.figure(figsize=(18,9))
            fig.suptitle('DWT level 1-2, coefficients {0}\n{1}'.format(coeff,fname),
                        fontsize=16)
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            #masking inner and outer region
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
            #to setup the colorboxes and limits
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
        #boolean mask made in scalMask_dflat
        npyMask_A = np.load('/work/devel/fpazch/shelf/cA_mask.npy')
        npyMask_H = np.load('/work/devel/fpazch/shelf/cH_mask.npy')
        npyMask_V = np.load('/work/devel/fpazch/shelf/cV_mask.npy')
        npyMask_D = np.load('/work/devel/fpazch/shelf/cD_mask.npy')
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
                Graph.filter_plot(npyMask_A,npyMask_H,npyMask_V,
                                c1[0]*c1[0],c2[0]*c2[0],'H',fname)
            if True:
                print '\t=== performing over c_V ==='
                #for level 1, c1[]
                #pos1V = Graph.filter_plot(c1[1]*c1[1],(1,'V'))
                #pos2V = Graph.filter_plot(c2[1]*c2[1],(2,'V'))
                Graph.filter_plot(npyMask_A,npyMask_H,npyMask_V,
                                c1[1]*c1[1],c2[1]*c2[1],'V',fname)
            if True:
                print '\t=== performing over c_D ==='
                #for level 1, c1[]
                #pos1D = Graph.filter_plot(c1[2]*c1[2],(1,'D'))
                #pos2D = Graph.filter_plot(c2[2]*c2[2],(2,'D'))
                Graph.filter_plot(npyMask_A,npyMask_H,npyMask_V,
                                c1[2]*c1[2],c2[2]*c2[2],'D',fname)
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
        '''Wrapper for mask the inner region of DECam, levels 1 and 2.
        Inputs:
        - h5table: H5 tables of the DWT coeffs
        - table_nm: name of the H5 file, from which construct a temporal method
        to get info
        Returns: dataframe with values and column names
        '''
        #get the dictionary from the HDF5 table and 
        header = table.attrs.DB_INFO
        #the statistics of the values and positions for the RMS-selected peaks
        sel,frame_shape = Screen.inner_region(table)
        statVal = Toolbox.value_dispers(sel)
        statPos = Toolbox.posit_dispers(sel,frame_shape)
        #column names where v. stands for values, p. for positions, and db. for
        #DES database
        col = ['db.nite','db.expnum','db.band','db.reqnum','db.pfw_attempt_id']
        col += ['db.filename','db.factor','db.rms','db.worst']
        col += ['v.level','v.coeff','v.rms_all','v.uncert_all','v.mean',
        'v.median','v.stdev','v.rms','v.uncert','v.min','v.max','v.mad',
        'v.entropy','v.nclust','v.npeak','v.ratio']
        col += ['p.level','p.coeff','p.mean','p.median','p.stdev','p.rms',
        'p.uncert','p.min','p.max','p.mad','p.entropy'] 
        #saving iteratively for each level and each coeff
        out_stat = []
        #as the levels and amount of coefficients is the same for value_dispers
        #and for posit_dispersion, then we can use one as ruler for the other
        for i1 in xrange(len(statVal)):
            tmp = [header['nite'],header['expnum'],header['band'],
                header['reqnum'],header['pfw_attempt_id'],header['filename'],
                header['factor'],header['rms'],header['worst']]
            tmp += statVal[i1]
            tmp += statPos[i1]
            out_stat.append(tmp)
            tmp = []
        df = pd.DataFrame(out_stat,columns=col)
        return df


if __name__=='__main__':
    #to setup a mask of the outer region of DECam, based on a well behaved flat
    if False:
        Toolbox.binned_mask_npy()
    
    #this is the path to the zero-padded DWT tables
    #note that for band is case sensitive
    pathBinned = '/work/devel/fpazch/shelf/dwt_dmeyN2/'

    '''TO SAVE STATISTICS FOR A NITERANGE, ALL AVAILABLE BANDS
    '''
    if True:
        #remember to change-------------
        nite_range = [20160813,20170212]
        tag = 'y4'
        #-------------------------------
        band_range,expnum_range = Toolbox.band_expnum(nite_range)
        savepath = '/work/devel/fpazch/shelf/stat_dmeyN2/' 
        counter = 0
        print "\nStatistics on niterange: {0}. Year: {1}".format(nite_range,tag) 
        for b in band_range:
            gc.collect()
            print '\nStarting with band:{0}\t{1}'.format(b,time.ctime())
            savename = 'qa_' + b + '_' + tag + '_.csv'
            for (path,dirs,files) in os.walk(pathBinned):
                for index,item in enumerate(files):   #file is a string
                    expnum = int(item[1:item.find('_')])
                    if (('_'+b+'_r' in item) and (expnum >= expnum_range[0])
                    and (expnum <= expnum_range[1])):
                        try:
                            H5tab = OpenH5(pathBinned+item)
                            table = H5tab.h5.root.dwt.dmeyN2
                            print '{0}-- {1}'.format(counter+1,item)
                            tmp = Call.wrap3(table,item)
                            if (counter == 0): df_res = tmp
                            else: df_res = pd.concat([df_res,tmp])
                            counter += 1
                        finally:
                            #close open instances
                            H5tab.closetab()
                            table.close()
            df_res.reset_index(drop=True,inplace=True)
            #write oout the table of results
            df_res.to_csv(savename,index=False,header=True)


    '''TO SAVE STATISTICS FOR ALL DWT TABLES IN A EXPNUM RANGE
    '''
    #remember to change for each year
    if False:
        tag = 'yn'
        band = 'g'
        print '\n\tStatistics on all available tables, band: {0}'.format(band)
        savepath = '/work/devel/fpazch/shelf/stat_dmeyN2/' 
        savename = 'qa_' + band + '_' + tag + '.csv'
        filler = 0
        expnum_range = range(606456,606541+1)
        for (path,dirs,files) in os.walk(pathBinned):
            for index,item in enumerate(files):   #file is a string
                expnum = int(item[1:item.find('_')])
                if ('_'+band+'_' in item) and (expnum in expnum_range):
                    try:
                        H5tab = OpenH5(pathBinned+item)
                        table = H5tab.h5.root.dwt.dmeyN2
                        print '\t',item
                        tmp = Call.wrap3(table,item)
                        if (filler == 0): df_res = tmp
                        else: df_res = pd.concat([df_res,tmp])
                        filler += 1
                    finally:
                        #close open instances
                        H5tab.closetab()
                        table.close()
        df_res.reset_index(drop=True,inplace=True)
        #write oout the table of results
        df_res.to_csv(savename,index=False,header=True)
     
    '''TO VISUALIZE BY EXPNUM RANGE
    '''
    if False:
        expnum_range = range(606456,606541+1)#(460179,516693)#(606738,606824+1)
        opencount = 0
        for (path,dirs,files) in os.walk(pathBinned):
            for index,item in enumerate(files):   #file is a string
                expnum = int(item[1:item.find('_')])
                #if not ('_r2625' in item):
                #    print item
                if (('_'+band+'_' in item) and (expnum in expnum_range)):
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
    
    '''PREVIOUS TRY TO SAVE STATISTICS
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
