"""Adapted from my other code:
viewDWT_flat.py
"""

import os
import sys
import time
import gc
import argparse
import numpy as np
import pandas as pd
#setup for display visualization
import matplotlib
matplotlib.use("TkAgg")
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
import astroML
#internal modules
#import y4g1g2g3_flatDWT as y4
#import scalMask_dflat as scalMask

#
# PENDING
# - tabulate statistics
#


class Toolbox():
    @classmethod
    def polygon_area(cls,x_arr,y_arr):
        """uses shoelace algorithm or Gauss area formula to get the polygon
        area by sucessive triangulation. Sets of coordinates do not need to be
        sorted
        """
        A = 0.5*np.abs(np.dot(x_arr,np.roll(y_arr,1))-
                    np.dot(y_arr,np.roll(x_arr,1)))
        return A

    @classmethod
    def rms(cls,arr,maskFP=False,baseMask=None):
        if maskFP:
            m1 = scalMask.Mask.scaling(baseMask,arr)
            arr = arr[np.where(m1)]
        outrms = np.sqrt(np.mean(np.square(arr.ravel())))
        return outrms

    @classmethod
    def uncert(cls,arr,maskFP=False,baseMask=None):
        if maskFP:
            M = scalMask.Mask.scaling(baseMask,arr)
            arr = arr[np.where(M)]
        ux = np.sqrt(np.mean(np.square(arr.ravel())) +
                    np.square(np.mean(arr.ravel())))
        return ux

    @classmethod
    def median_mask(cls,arr,maskFP=False,baseMask=None):
        if maskFP:
            M = scalMask.Mask.scaling(baseMask,arr)
            arr = arr[np.where(M)]
        mm = np.median(arr.ravel())
        return mm

    @classmethod
    def corr_random(cls,data):
        """correlate data with random 2D array
        """
        auxrdm = np.random.rand(data.shape[0],data.shape[1])
        auxrdm = auxrdm/np.mean(auxrdm)
        corr2d = scipy.signal.correlate2d(data,auxrdm,mode="same",
                                        boundary="symm")
        return corr2d

    @classmethod
    def gaussian_filter(cls,data,sigma=1.):
        """performs Gaussian kernel on image
        """
        return scipy.ndimage.gaussian_filter(data,sigma=sigma)

    @classmethod
    def cluster_dbscan(cls,points,minsize=1):
        """friends of friends
        points is a set of coordinates
        DBSCAN.labels_: label for each point, where -1 is set for noise.
        Includes cores and members not belonging to the core but still on
        the group.
        DBSCAN.core_sample_indices_: list of indices of core samples
        DBSCAN.components_:copy of each core sample found by training
        core_sample_indices_
        Note:
        minsize=1, because we"re using binned focal plane images
        """
        #here we consider the diagonal of a square of side=1 because of
        #the positioning we"re using is the grid of points with unity spacing
        diagonal = np.sqrt(2.)
        #real impact of leaf_size has not been unveiled
        db = sklearn.cluster.DBSCAN(eps=diagonal,min_samples=minsize,
                                leaf_size=30, metric="euclidean").fit(points)
        #mask to identify which elements in the cluster belongs
        #to the inner core and which to the outer region
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        #exclude those labeled as -1
        n_cluster = np.unique(db.labels_[np.where(db.labels_ != -1)]).shape[0]
        return n_cluster,db.labels_,core_samples_mask

    @classmethod
    def mask_polygon(cls,arr,vertex):
        """this simple method makes a mask of the input array, based on the
        input vertices. If subsampling is needed, simply increase the amount
        of points in x,y meshgrid
        """
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
    def mask_join(cls,values_arr,threshold,baseM=None):
        """method to combine mask coming from inner region of DECam and
        and an input value
        Returns the joint maks and the mask of only the peaks (still inside
        the inner DECam region)
        """
        aux_fnm = "/Users/fco/Code/des_calibrations/dwt_files/masks/"
        aux_fnm += "s_mask_b88.npy"
        baseMask = np.load(aux_fnm)
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
    def lower_cut(cls,arr,threshold):
        """method to apply a threshold on data, discarding lower values
        Inputs
        - arr: 2D array of values
        - threshold: value to apply as lowercut
        Returns 
        - the mask of the selected values
        """
        aux = np.zeros_like(arr,dtype=bool)
        #aux[np.where(np.logical_and(arr,values_arr>threshold))] = True
        aux[np.where(arr>threshold)] = True
        #res = np.zeros_like(arr,dtype=bool)
        #res[np.where(np.logical_or(arr,aux))] = True
        return aux

    @classmethod
    def value_dispers_undec(cls,sval):
        """Method to estimate dispersion in the values of the clustered points.
        Not related to the position, but to the power.
        For the entropy, we"ll use the values as unnormalized probabilities
        Intputs
        - list of values for a single direction of decomposition, for a single
        level. Only the power values, not further info
        """
        sval = np.array(sval)
        x1 = np.mean(sval)
        x2 = np.median(sval)
        x3 = np.std(sval)
        x4 = Toolbox.rms(sval)
        x5 = Toolbox.uncert(sval)
        x6 = np.min(sval)
        x7 = np.max(sval)
        x8 = np.median(np.abs(sval - np.median(sval)))
        x9 = scipy.stats.entropy(sval.ravel())
        x10 = scipy.stats.skew(sval)
        x11 = scipy.stats.kurtosis(sval,fisher=False,bias=True)
        x12 = sval.shape[0]
        out = [x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12]
        return out

    @classmethod
    def posit_dispers(cls,sel_pts,coeff_shape):
        #scikit learn
        #http://scikit-learn.org/stable/supervised_learning.html#supervised-learning
        #astroml
        #
        
        """Method to evaluate the spatial behavior of the detected peaks.
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
        """

        #returns the location of the 2 focus
        #http://www.astroml.org/modules/generated/astroML.stats.fit_bivariate_normal.html#astroML.stats.fit_bivariate_normal
        x,y = None,None
        astroML.stats.fit_bivariate_normal(x, y, robust=False)
        exit()

        #
        # HERE
        # HERE HERE!
        #
        out = []
        nitem = 11
        labcoeff = ["H","V","D"]
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
                    """PENDING: ADD PCA ORIENTATION OF PPAL 3 VECTORS"""
        return out


class Screen():
    @classmethod
    def inner_region_undec(cls,h5table,mincluster=3):
        """Modified version of  inner_region() for undecimated results
        Returns:
        - list with 4 depth levels 
        [[[(),    ...    , ()],   [],     []  ],[   ] ...]
          thres1, ...  ,thresn   
                 cA_lev1       cA_lev2  cA_lev3
                           cA
        - list with 3 depth levels, 1st AHVD coeffs, 2nd levels 1,2,3, and
        3rd values 1,2,3,4. These inner tuples contains gral information 
        of the overall non-filtered DWT
        """
        gc.collect()
        print h5table.colnames
        ahvd_aux = []
        ahvd_val = []
        #iterate over cA,cH,cV,cD
        for row in h5table.iterrows():
            print row
            lev_aux = []
            lev_val = []
            #iterate over lev1,lev2,lev3
            for col in h5table.colnames:
                M = row[col]
                #overall values
                v1 = Toolbox.rms(M)
                v2 = Toolbox.uncert(M)
                v3 = np.median(M)
                v4 = np.mean(M)
                """we're not using normalized values
                use different thresholds: saving different thresholds results
                will result in a main for better comparison
                """
                thr_aux = []
                #iterate over 1,2,3 times RMS (overall RMS, not only inner)
                for t in [1,2,3]:
                    #after apply threshold, save a boolean mask of selected
                    #values and its coordinates
                    maskN = Toolbox.lower_cut(M,t*v1)
                    coo = np.argwhere(maskN)
                    #get values corresponding to the masked positions
                    valN = M[np.where(maskN)]
                    #absence/few peaks is a possibility, we must account for it
                    #remember that minimal shape of coordinates is [0,2]
                    if (coo.shape[0] > mincluster):
                        cl_N,cl_label,cl_mask = Toolbox.cluster_dbscan(
                            coo,minsize=mincluster)
                        iCore = np.where(np.logical_and(cl_label!=-1,cl_mask))
                        iOut = np.where(np.logical_and(cl_label!=-1,~cl_mask))
                        iNoise = np.where(cl_label==-1)
                        #to call only the coefficients belonging to a single
                        #category: M[coo[iCore][:,0],coo[iCore][:,1]]
                        categ = np.zeros_like(cl_mask)
                        categ[iCore],categ[iOut],categ[iNoise] = 1,0,-1
                        """for each threshold level save:
                        - coordinates of thresholded coeffs:coo
                        - values of the theresholded coeffs:val
                        - categories of the thresholded coeffs:categ
                        - labels of the thresholded coeffs:cl_label
                        """
                        thr_aux.append((coo,valN,categ,cl_label))
                        #thr_aux.append((np.nan,np.nan,np.nan,np.nan))
                    else:
                        pass
                        thr_aux.append((np.nan,np.nan,np.nan,np.nan))
                lev_aux.append(thr_aux)
                lev_val.append((v1,v2,v3,v4))
            ahvd_aux.append(lev_aux)
            ahvd_val.append(lev_val)
        return ahvd_val,ahvd_aux

class OpenH5():
    """class oriented to open and close a pytables instance. It"s not the same
    to close the instance here inside the same class than outside
    """
    def __init__(self,fname):
        self.h5 = tables.open_file(fname,driver="H5FD_CORE")

    def closetab(self):
        self.h5.close()


class Call():
    @classmethod
    def wrap3(cls,h5table,table_nm):
        """Wrapper for mask the inner region of DECam, levels 1 and 2.
        Inputs:
        - h5table: H5 tables of the DWT coeffs
        - table_nm: name of the H5 file, from which construct a temporal method
        to get info
        Returns: dataframe with values and column names
        """
        #get the dictionary from the HDF5 table and
        header = table.attrs.DB_INFO
        #the statistics of the values and positions for the RMS-selected peaks
        sel,frame_shape = Screen.inner_region(table)
        statVal = Toolbox.value_dispers(sel)
        statPos = Toolbox.posit_dispers(sel,frame_shape)
        #column names where v. stands for values, p. for positions, and db. for
        #DES database
        col = ["db.nite","db.expnum","db.band","db.reqnum","db.pfw_attempt_id"]
        col += ["db.filename","db.factor","db.rms","db.worst"]
        col += ["v.level","v.coeff","v.rms_all","v.uncert_all","v.mean",
        "v.median","v.stdev","v.rms","v.uncert","v.min","v.max","v.mad",
        "v.entropy","v.nclust","v.npeak","v.ratio"]
        col += ["p.level","p.coeff","p.mean","p.median","p.stdev","p.rms",
        "p.uncert","p.min","p.max","p.mad","p.entropy"]
        #saving iteratively for each level and each coeff
        out_stat = []
        #as the levels and amount of coefficients is the same for value_dispers
        #and for posit_dispersion, then we can use one as ruler for the other
        for i1 in xrange(len(statVal)):
            tmp = [header["nite"],header["expnum"],header["band"],
                header["reqnum"],header["pfw_attempt_id"],header["filename"],
                header["factor"],header["rms"],header["worst"]]
            tmp += statVal[i1]
            tmp += statPos[i1]
            out_stat.append(tmp)
            tmp = []
        df = pd.DataFrame(out_stat,columns=col)
        return df

    @classmethod
    def wrap_x(cls,h5tab):
        """Wrapper to perform stats over the new H5 files, generated using
        undecimated 2D wavelet decomposition
        """
        header = h5tab.attrs.DB_INFO
        print header
        """option: execute the statistics without masking, because no weird 
        coeffs are generated on level 1 of decomposition. But in level 2 and 3
        must be a little more cautious
        """
        #values of statistics for non-filtered coeffs, and positions/values
        #for filtered coeffs. vX is a list of a tuple per level, clX is a 
        #per level, with 2 more levels: one per threshold and other (a tuple)
        #of the grouping values
        [vA,vH,vV,vD],[clA,clH,clV,clD] = Screen.inner_region_undec(h5tab)
        """apply  the statistics to all the directions (AHVD), levels (1,2,3)
        and thresholds (1-3 x RMS), to the selected points and coordiantes
        """
        #save different files for different thresholds or a 3D array
        #
        #first test for cA and then expand to the other directions
        for lev in clA:
            for threshold in lev:
                coo,val,categ,label = threshold
                #inside each threshold tuple there are 4 elements:
                #- coordinates of thresholded coeffs:coo
                #- values of the theresholded coeffs:val
                #- categories of the thresholded coeffs:categ
                #- labels of the thresholded coeffs:cl_label
                print type(val),val
                if (not isinstance(val,float)):
                    y1 = Toolbox.value_dispers_undec(val)
                    # y2 = Toolbox.posit_dispers()
        #if non nan, run stat
        exit(0)

        #
        # HERE, HERE HERE
        #

        #the statistics of the values and positions for the RMS-selected peaks
        sel,frame_shape = Screen.inner_region(table)
        statVal = Toolbox.value_dispers(sel)
        statPos = Toolbox.posit_dispers(sel,frame_shape)
        #column names where v. stands for values, p. for positions, and db. for
        #DES database
        col = ["db.nite","db.expnum","db.band","db.reqnum","db.pfw_attempt_id"]
        col += ["db.filename","db.factor","db.rms","db.worst"]
        col += ["v.level","v.coeff","v.rms_all","v.uncert_all","v.mean",
        "v.median","v.stdev","v.rms","v.uncert","v.min","v.max","v.mad",
        "v.entropy","v.nclust","v.npeak","v.ratio"]
        col += ["p.level","p.coeff","p.mean","p.median","p.stdev","p.rms",
        "p.uncert","p.min","p.max","p.mad","p.entropy"]
        #saving iteratively for each level and each coeff
        out_stat = []
        #as the levels and amount of coefficients is the same for value_dispers
        #and for posit_dispersion, then we can use one as ruler for the other
        for i1 in xrange(len(statVal)):
            tmp = [header["nite"],header["expnum"],header["band"],
                header["reqnum"],header["pfw_attempt_id"],header["filename"],
                header["factor"],header["rms"],header["worst"]]
            tmp += statVal[i1]
            tmp += statPos[i1]
            out_stat.append(tmp)
            tmp = []
        df = pd.DataFrame(out_stat,columns=col)
        return df


if __name__=="__main__":
    #to setup a mask of the outer region of DECam, based on a well behaved flat
    descr = "To run statistics on HDF5 2D wavelets result tables. Argparse"
    descr += " under construction"
    pre = argparse.ArgumentParser(description=descr)
    #optional
    h0 = "Path to the HDF5 tables. Default: dmey/"
    pre.add_argument("--folder",help=h0,metavar="",default="dmey/")
    rec = pre.parse_args()
    kw_x = vars(rec)
    print type(rec.folder), kw_x

    """As FOR NOW I'll perform the code over all the files inside the folder,
    then no DB query is needed
    """
    msg = "="*80
    msg += "\nRemember this is a testing-code on which different thresholds\n"
    msg += "are being compared. Keep the results storage simple\n"
    msg += "="*80
    print msg
    DEPTH = 0
    replace = "/Users/fco/Code/des_calibrations/dwt_files/"
    replace += "sample_tst20161216t0223/dmey/"
    #for root,dirs,files in os.walk(rec.folder):
    for root,dirs,files in os.walk(replace):
        if root.count(os.sep) >= DEPTH:
            del dirs[:]
        for tab in files:
            try:
                H5tab = OpenH5(os.path.join(root,tab))
                table = H5tab.h5.root.dflat.dmey
                Call.wrap_x(table)
            finally:
                #close open instances
                H5tab.closetab()
                table.close()
    
    
    
    
    
    
    
    
    
    exit(0)

    """Below is the past call"""
    if False:
        Toolbox.binned_mask_npy()

    #this is the path to the zero-padded DWT tables
    #note that for band is case sensitive
    #pathBinned = "/work/devel/fpazch/shelf/dwt_dmeyN2/pixcor/"
    pathBinned = "dmey/"

    """TO SAVE STATISTICS FOR A NITERANGE, ALL AVAILABLE BANDS
    """
    if False:#True:
        #remember to change-------------
        nite_range = [20160813,20170223]
        tag = "y4"
        #using 3RMS as the threshold!
        #-------------------------------
        band_range,expnum_range = Toolbox.band_expnum(nite_range)
        #savepath = "/work/devel/fpazch/shelf/stat_dmeyN2/"
        print "\nStatistics on niterange: {0}. Year: {1}".format(nite_range,tag)
        print "(note:) Threshold=3*RMS"
        for b in band_range:
            counter = 0
            gc.collect()
            print "\nStarting with band:{0}\t{1}".format(b,time.ctime())
            savename = os.path.join(os.path.expanduser("~"),
                                "Result_box/qa_{0}_{1}.csv".format(b,tag))
            for (path,dirs,files) in os.walk(pathBinned):
                for index,item in enumerate(files):   #file is a string
                    expnum = int(item[1:item.find("_")])
                    if (("_"+b+"_r" in item) and (expnum >= expnum_range[0])
                    and (expnum <= expnum_range[1])):
                        try:
                            H5tab = OpenH5(pathBinned+item)
                            table = H5tab.h5.root.dwt.dmeyN2
                            print "{0}-- {1}".format(counter+1,item)
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
            df_res = None

    """TO SAVE STATISTICS FOR ALL DWT TABLES IN A EXPNUM RANGE
    """
    #remember to change for each year
    if False:
        tag = "yn"
        band = "g"
        print "\n\tStatistics on all available tables, band: {0}".format(band)
        #savepath = "/work/devel/fpazch/shelf/stat_dmeyN2/"
        savename = "qa_" + band + "_" + tag + ".csv"
        filler = 0
        expnum_range = range(606456,606541+1)
        for (path,dirs,files) in os.walk(pathBinned):
            for index,item in enumerate(files):   #file is a string
                expnum = int(item[1:item.find("_")])
                if ("_"+band+"_" in item) and (expnum in expnum_range):
                    try:
                        H5tab = OpenH5(pathBinned+item)
                        table = H5tab.h5.root.dwt.dmeyN2
                        print "\t",item
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

    """TO VISUALIZE BY EXPNUM RANGE
    """
    if False:
        #stable period in Y4: sept01 to sept10 570284,572853
        expnum_range = range(587208,587218+1)#range(564659,564661+1)
        #expnum_range = range(606477,606487+1)
        #(606456,606541+1)#(460179,516693)#(606738,606824+1)
        band = "r"
        opencount = 0
        for (path,dirs,files) in os.walk(pathBinned):
            for index,item in enumerate(files):   #file is a string
                expnum = int(item[1:item.find("_")])
                #if not ("_r2625" in item):
                #    print item
                if (("_"+band+"_" in item) and (expnum in expnum_range)):
                    opencount += 1
                    print "{0} ___ {1} Iter: {2}".format(item,index+1,opencount)
                    try:
                        H5tab = OpenH5(pathBinned+item)
                        table = H5tab.h5.root.dwt.dmeyN2
                        Graph.decomp(table,Nlev=2)
                        Call.wrap1(table,item)
                        Graph.histogram(table,Nlev=2)
                        #TimeSerie.count_and_tab(table,Nlev=2)
                    finally:
                        #close open instances
                        H5tab.closetab()
                        table.close()

    """PREVIOUS TRY TO SAVE STATISTICS
    """
    if False:
        #to make statistics and save to file
        fcounter = 0
        for (path,dirs,files) in os.walk(pathBinned):
            for index,item in enumerate(files):   #file is a string
                if (group+".h5" in item):
                    print "{0} ___ {1}".format(item,index+1)
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
            fnout = "stat_cD_" + group + ".csv"
            df_res.to_csv(fnout,header=True,index=False)
        except:
            print "Error in write output DF"
