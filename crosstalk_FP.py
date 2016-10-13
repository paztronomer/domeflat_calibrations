'''Crosstlak

how crosstalk correction is applied? if ccd by ccd, then the code
can be optimized to transform piece-by-piece

remember each keyword can be 80 ASCII characters long

set a tupe with the allowed OBSTYPE

set a tuple of header keywords, and therefore call using this
structure


IMPORTANT!
---------
    - corrections will be made CCD by CCD
    - the script intents to be self sustainable (without imsupport include),
    on a second stage, the functions will be replaced by the imsupport ones
    when imsupport be already in python
    - REMEMBER TO ADD AS MUCH FREEDOM AS POSSIBLE TO THE USER

Methods from imsupport and argutils
'''

import os
import sys
import time
import gc
import numpy as np
import pandas as pd #version 0.18.1
import tables #version 3.2.3.1
import fitsio #version 0.9.8
import matplotlib.pyplot as plt



class Auxiliary():
    def __init__(self):
        gc.collect()
        
    @classmethod
    def dim_str(cls,str_range):
        str_range = str_range.strip('[').strip(']').replace(':',',').split(',')
        return (int(str_range[1]),int(str_range[3]))

    @classmethod
    def range_str(cls,str_range):
        '''return the indices (x-1), as integer 
        '''
        str_range = str_range.strip('[').strip(']').replace(':',',').split(',')
        return map(lambda x: int(x)-1, str_range)


class UpdateHeader():
    '''Updates GAIN, SATURATE LEVEL, and (...) on the header
    '''
    def __init__(self):
        gc.collect()
    
    @classmethod 
    def calc_gain():
        return False

    @classmethod
    def write_param():
        return False


class Correction():
    def __init__(self):
        gc.collect()
        
    @classmethod
    def open_corr(cls,cross_filename):
        '''Receives crosstalk filename and returns DataFrame plus
        a tuple containing the column names
        '''
        columns = ('victim','source','x','x_err','src_nl',
                   'p_coeff_a','p_coeff_b','p_coeff_c')
        tmp_tab = pd.read_csv(cross_filename,sep='\s*',comment='#',
                              names=columns,engine='python')
        return tmp_tab,columns

    @classmethod
    def crosscorr(cls,fp_raw,df_coeff,cols_coeff,df_key,ccd_init=1,ccd_end=62):
        '''Performs crosstalk on CCDs 1-62 (because for those we have
        coefficients)
        Coeficients will be harbored on a NDim array (not pytables), and the
        resulting corrected coeffs will be passed to a list (to not worry about
        dimensions)
        The coeffs matix (NDIM) needs to be carefully constructed

        NOTE: if a subset of CCDs are selected, is better to use a similar
        method but not this
        '''
        print '\t(+)Empty method {0}, on class: {1}.\n\t(+)File: {2}'.format(
            sys._getframe().f_code.co_name,cls.__name__,__file__)
        print 'Initial/Final CCD number: {0}/{1}'.format(ccd_init,ccd_end)
        #look at the entire ccd, not only datasec
        '''make the simplest version and when working, then improve
        this method is so slow! (about 2 minutes) so improve its
        performance! (maybe merge with other method?)
        '''
        t1 = time.time()
        for C in xrange(pixel.shape[1]):
            for R in xrange(pixel.shape[0]):
                #for ccd,AB_amp,pixel in fp_raw:
                #    locout = C*pixel.shape[0]+R
                #    xtalkcorr = 0
                #    #if column in A and AB
                #    #Auxiliary.in_range()
                #    #if column in A and BA
        t2 = time.time()
        print 'option 2: {0}'.format((t2-t1)/60.)

        key_ls = ['detsec','detseca','detsecb',
                'dataseca','datasecb','biasseca','biassecb',
                'preseca','presecb','postseca','postsecb',
                'ccdnum','extname']
        print 'here tic toc'
        return False
        
class ManageCCD():
    def __init__(self):
        gc.collect()
        
    @classmethod
    def split_amp(cls,pix_arr,df_delim,tech_ccd=False):
        '''Receives a set of pixel arrays (with all the available data)
        and a set of delimiters from which we get the boundaries of
        ccd, overscan, postscan, etc
        - DATASEC{A,B} image section
        - BIASSEC{A,B} overscan section
        - PRESEC{A,B} prescan section
        - POSTSEC{A,B} postscan section
        Here the same dictionary as OpenFile() is employed
        Is NOT the same sectioning for all the CCDs. Is the same shape but not
        same sectioning.
        Pandas documentation recommend optimized data access .at, .iat, .loc,
        .iloc and .ix
        It's important that auxiliary list for sectioning only contain
        sectioning info
        NOTE: even when physically CCDs are horizontal (larger axis is columns),
        after readout the longer axis is vertical (rows)
        '''
        aux_slice = ['dataseca','datasecb','biasseca','biassecb',
                     'preseca','presecb','postseca','postsecb']
        aux_slice.sort(key=lambda x: x.lower())

        '''Create a DF to harbor delimiters and shapes for each ccd, per section
        When created, free space from the initial DF. As CCD orientation changes
        from header to readout, I switched del{1,2,3,4}
        '''
        df_ind = pd.DataFrame(columns=['ccdnum','section',
                                       'del1','del2','del3','del4',
                                       'dim1','dim2'])
        for kword in aux_slice:
            for j in xrange(len(df_delim[kword])):
                aux_sl = Auxiliary.range_str(df_delim.loc[j,kword])
                tmp_df = pd.DataFrame({'ccdnum':df_delim.loc[j,'ccdnum'],
                                       'section':kword,
                                       'del1':aux_sl[2],'del2':aux_sl[3],
                                       'del3':aux_sl[0],'del4':aux_sl[1],
                                       'dim1':np.abs(aux_sl[3]+1-aux_sl[2]),
                                       'dim2':np.abs(aux_sl[1]+1-aux_sl[0])},
                                      index=[0])
                df_ind = df_ind.append(tmp_df)
        #to force as integers
        df_ind[['ccdnum','del1','del2','del3','del4',
                'dim1','dim2']] = df_ind[['ccdnum','del1','del2','del3','del4',
                                          'dim1','dim2']].astype(int)
        df_ind = df_ind.reset_index()
        df_ind = df_ind.drop('index', axis=1)
        df_delim = None

        '''Up to here there is a DF (df_ind) storing indices and dimensions for
        the focalplane cropping. Now I need to make the ccd crop and store these
        arrays inside an object
        To define the pytables, all arrays inside a column must has the same
        dimension. I need to store Science and Guide/Focus ccds separately
        * DATASEC{A/B}, BIASSEC{A/B}, and PRESEC{A/B}: all shares the same 
        dimension on dim1 (separately for Sci and Technical ccds), and inside 
        each section shares the same dimension in dim2
        - datasec:(4096,1024),(20148,1024)
        - biassec:(4096,50),(2048,50)
        - presec:(4096,6),(2048,6)
        - postsec:(50,1024)
        The following variables stores the dimensions of each section, after
        verify these dimensions are unique. We could have put these values
        by hand, but this way is more robust
        '''
        #HERE
        #Put a condition for CCDs to be extracted, because following conditions
        #will change
        if (len(df_ind.loc[(df_ind['section']=='dataseca')
                           & (df_ind['ccdnum']<=62),'dim1']
                .drop_duplicates().values) == 1):
            share_D1 = int(df_ind.loc[(df_ind['section']=='dataseca')
                                      & (df_ind['ccdnum']<=62),'dim1']
                           .drop_duplicates().values[0]) #4096
        else:
            print '(!)Error in shared DIM1 (Science)'; exit()
            
        if tech_ccd:
            if (len(df_ind.loc[(df_ind['section']=='dataseca')
                               & (df_ind['ccdnum']>=63),'dim1']
                    .drop_duplicates().values) == 1):
                share_D1_tech = int(df_ind.loc[(df_ind['section']=='dataseca')
                                               & (df_ind['ccdnum']>=63),'dim1']
                                    .drop_duplicates().values[0]) #2048
            else:
                print '(!)Error in shared DIM1 (Technical)'; exit()
        else:
            pass
        
        if ( (len(df_ind.loc[df_ind['section']=='dataseca','dim2']
                  .drop_duplicates().values) == 1) and
             (len(df_ind.loc[df_ind['section']=='biasseca','dim2']
                  .drop_duplicates().values) == 1) and
             (len(df_ind.loc[df_ind['section']=='preseca','dim2']
                  .drop_duplicates().values) == 1) ):
            share_data_D2 = int(df_ind.loc[df_ind['section']=='dataseca','dim2']
                                .drop_duplicates().values[0]) #1024
            share_bias_D2 = int(df_ind.loc[df_ind['section']=='biasseca','dim2']
                                .drop_duplicates().values[0]) #50
            share_pre_D2 = int(df_ind.loc[df_ind['section']=='preseca','dim2']
                               .drop_duplicates().values[0]) #6
        else:
            print '(!)Error in shared DIM2'; exit()


        '''Define pytables using the above dimensions and fill it up
        The config: driver="H5FD_CORE",
                    driver_core_backing_store=0
        prevents to save a copy of the table on disk
        '''
        class Record(tables.IsDescription):
            ccdnum = tables.Int32Col() #32-bit integer
            flag_amp = tables.Int32Col()#StringCol(10) #-1/1: A/B amplifiers
            datasec = tables.Float32Col(shape=(share_D1,share_data_D2))
            biassec = tables.Float32Col(shape=(share_D1,share_bias_D2))
            presec = tables.Float32Col(shape=(share_D1,share_pre_D2))
            postsec = tables.Float32Col(shape=(share_bias_D2,share_data_D2))
            if tech_ccd:
                datasec_tec = tables.Float32Col(shape=(share_D1_tech,
                                                share_data_D2))
                biassec_tec = tables.Float32Col(shape=(share_D1_tech,
                                                share_bias_D2)) 
                presec_tec = tables.Float32Col(shape=(share_D1_tech,
                                                share_pre_D2))
                postsec_tec =tables.Float32Col(shape=(share_bias_D2,
                                                share_data_D2))
        h5file = tables.open_file('fp_prextalk.h5', mode = 'w',
                                  title = 'FP sections',
                                  driver='H5FD_CORE',
                                  driver_core_backing_store=0)
        group = h5file.create_group('/','pixels','CCD sections')
        table = h5file.create_table(group,'preXtalk',Record,'Pre-crosstalk')
        fp_row = table.row

        '''Fill up Science (1-62) & Technical (63-64,69-74) CCDs sections.
        One row for each amplifier, then 2 lines per CCD. It's verbose...
        '''
        non_used_ccd = []
        t1 = time.time()
        for M in xrange(len(pix_arr)):
            ID_pix,AB_pix,ARR_pix = pix_arr[M]
            #cut in pieces for A and B
            if (ID_pix <= 62 and ID_pix >= 1):
                #Amplifier A
                fp_row['ccdnum'] = ID_pix
                fp_row['flag_amp'] = -1
                fp_row['datasec'] = ARR_pix[
                    df_ind.loc[(df_ind['ccdnum']==ID_pix) &
                               (df_ind['section']=='dataseca'),'del1'].values[0]:
                    df_ind.loc[(df_ind['ccdnum']==ID_pix) &
                               (df_ind['section']=='dataseca'),'del2'].values[0]+1,
                    df_ind.loc[(df_ind['ccdnum']==ID_pix) &
                               (df_ind['section']=='dataseca'),'del3'].values[0]:
                    df_ind.loc[(df_ind['ccdnum']==ID_pix) &
                               (df_ind['section']=='dataseca'),'del4'].values[0]+1]
                fp_row['biassec'] = ARR_pix[
                    df_ind.loc[(df_ind['ccdnum']==ID_pix) &
                               (df_ind['section']=='biasseca'),'del1'].values[0]:
                    df_ind.loc[(df_ind['ccdnum']==ID_pix) &
                               (df_ind['section']=='biasseca'),'del2'].values[0]+1,
                    df_ind.loc[(df_ind['ccdnum']==ID_pix) &
                               (df_ind['section']=='biasseca'),'del3'].values[0]:
                    df_ind.loc[(df_ind['ccdnum']==ID_pix) &
                               (df_ind['section']=='biasseca'),'del4'].values[0]+1]
                fp_row['presec'] = ARR_pix[
                    df_ind.loc[(df_ind['ccdnum']==ID_pix) &
                               (df_ind['section']=='preseca'),'del1'].values[0]:
                    df_ind.loc[(df_ind['ccdnum']==ID_pix) &
                               (df_ind['section']=='preseca'),'del2'].values[0]+1,
                    df_ind.loc[(df_ind['ccdnum']==ID_pix) &
                               (df_ind['section']=='preseca'),'del3'].values[0]:
                    df_ind.loc[(df_ind['ccdnum']==ID_pix) &
                               (df_ind['section']=='preseca'),'del4'].values[0]+1]
                fp_row['postsec'] = ARR_pix[
                    df_ind.loc[(df_ind['ccdnum']==ID_pix) &
                               (df_ind['section']=='postseca'),'del1'].values[0]:
                    df_ind.loc[(df_ind['ccdnum']==ID_pix) &
                               (df_ind['section']=='postseca'),'del2'].values[0]+1,
                    df_ind.loc[(df_ind['ccdnum']==ID_pix) &
                               (df_ind['section']=='postseca'),'del3'].values[0]:
                    df_ind.loc[(df_ind['ccdnum']==ID_pix) &
                               (df_ind['section']=='postseca'),'del4'].values[0]+1]
                #make it effective
                fp_row.append()
                #Amplifier B
                fp_row['ccdnum'] = ID_pix
                fp_row['flag_amp'] = 1
                fp_row['datasec'] = ARR_pix[
                    df_ind.loc[(df_ind['ccdnum']==ID_pix) &
                               (df_ind['section']=='datasecb'),'del1'].values[0]:
                    df_ind.loc[(df_ind['ccdnum']==ID_pix) &
                               (df_ind['section']=='datasecb'),'del2'].values[0]+1,
                    df_ind.loc[(df_ind['ccdnum']==ID_pix) &
                               (df_ind['section']=='datasecb'),'del3'].values[0]:
                    df_ind.loc[(df_ind['ccdnum']==ID_pix) &
                               (df_ind['section']=='datasecb'),'del4'].values[0]+1]
                fp_row['biassec'] = ARR_pix[
                    df_ind.loc[(df_ind['ccdnum']==ID_pix) &
                               (df_ind['section']=='biassecb'),'del1'].values[0]:
                    df_ind.loc[(df_ind['ccdnum']==ID_pix) &
                               (df_ind['section']=='biassecb'),'del2'].values[0]+1,
                    df_ind.loc[(df_ind['ccdnum']==ID_pix) &
                               (df_ind['section']=='biassecb'),'del3'].values[0]:
                    df_ind.loc[(df_ind['ccdnum']==ID_pix) &
                               (df_ind['section']=='biassecb'),'del4'].values[0]+1]
                fp_row['presec'] = ARR_pix[
                    df_ind.loc[(df_ind['ccdnum']==ID_pix) &
                               (df_ind['section']=='presecb'),'del1'].values[0]:
                    df_ind.loc[(df_ind['ccdnum']==ID_pix) &
                               (df_ind['section']=='presecb'),'del2'].values[0]+1,
                    df_ind.loc[(df_ind['ccdnum']==ID_pix) &
                               (df_ind['section']=='presecb'),'del3'].values[0]:
                    df_ind.loc[(df_ind['ccdnum']==ID_pix) &
                               (df_ind['section']=='presecb'),'del4'].values[0]+1]
                fp_row['postsec'] = ARR_pix[
                    df_ind.loc[(df_ind['ccdnum']==ID_pix) &
                               (df_ind['section']=='postsecb'),'del1'].values[0]:
                    df_ind.loc[(df_ind['ccdnum']==ID_pix) &
                               (df_ind['section']=='postsecb'),'del2'].values[0]+1,
                    df_ind.loc[(df_ind['ccdnum']==ID_pix) &
                               (df_ind['section']=='postsecb'),'del3'].values[0]:
                    df_ind.loc[(df_ind['ccdnum']==ID_pix) &
                               (df_ind['section']=='postsecb'),'del4'].values[0]+1]
                #make it effective
                fp_row.append()
            elif (tech_ccd and ID_pix >= 63):
                #HERE FILL THE TECHNICAL CCDs
                pass
            else:
                non_used_ccd.append(ID_pix)
        if len(non_used_ccd) > 0:
            non_used_ccd.sort()
            print ('-CCDs will not be crosstalked: {0}'.
                   format(non_used_ccd))
        t2 = time.time()
        print ('\n\t(*)Elapsed time in sectioning FP: {0:.3f}\'\n'.
               format((t2-t1)/60.))
        #h5file.close()
        return table,df_ind

    @classmethod
    def open_file(cls,filename,cross_file):
        '''to open fits and assign to a numpy array, it must be done
        inside the open/close of the file, otherwise the data structure
        remains as None.

        Important points:
        - load the entire FP --done
        - remember to close the main HDU file (M_hdu) --done
        '''
        t1 = time.time()
        #initilization, header type: fitsio.fitslib.FITSHDR
        M_header = fitsio.read_header(filename)
        M_hdu = fitsio.FITS(filename)

        #few descriptive data
        n_ext = M_header['NEXTEND']
        pix_scale = np.mean([M_header['PIXSCAL1'],M_header['PIXSCAL2']])
        date_obs = M_header['DATE-OBS'][:M_header['DATE-OBS'].find('T')]
        fp_size = Auxiliary.dim_str(M_header['detsize'])
        cls.detsize = fp_size

        '''Extensions runs from 0 to 70, where 0 is the non-data
        default extension. The division is:
        - 1 to 62: science / hduname '{N/S}{1-31}'
        - 63 to 70: guider/focus / hduname 'F{N/S}{1-4}'

        To read data: hdu[ext][:,:] or hdu[ext].read()
        To read header keys: hdu[ext].read_header()['key']
        '''

        '''whatever the order of the dictionary/list of columns, the
        DataFrame initializes using a alphabetical sort of the col-names
        The list is sorted using lowercase, because uppercase goes first
        for sorting
        '''
        #data sections and brief description
        #do i need focus/guider ccds for crosstalk correction?
        #fp_data is a list of [int,array]
        #arrays of data has longer axis as rows (vertical)
        #fp_data = [[M_hdu[i].read_header()['ccdnum'],M_hdu[i].read()]
        #           for i in xrange(1,n_ext+1)]
        #cls.fp_data_list = fp_data
        
        key_ls = ['detsec','detseca','detsecb',
                  'dataseca','datasecb','biasseca','biassecb',
                  'preseca','presecb','postseca','postsecb',
                  'ccdnum','extname']
        key_ls.sort(key=lambda x: x.lower())
        #lowercase is only for sorting, not permanent
        cls.df_key = pd.DataFrame(columns=key_ls)

        fp_data = []
        for h in xrange(1,n_ext+1):
            '''here is assumed the sorting pandas make with
            the DF column names, and such sorting must be
            equivalent to key_ls.sort()
            '''
            aux_row = [M_hdu[h].read_header()[kw] for kw in key_ls]
            cls.df_key.loc[len(cls.df_key)] = aux_row
            
            tmp_A = Auxiliary.range_str(M_hdu[h].read_header()['dataseca'])
            tmp_B = Auxiliary.range_str(M_hdu[h].read_header()['datasecb'])
            if tmp_A[0] < tmp_B[0]: AB_order = 1
            else: AB_order = -1
            fp_data.append([M_hdu[h].read_header()['ccdnum'],
                            AB_order,
                            M_hdu[h].read()])
        cls.fp_data_list = fp_data

        t2 = time.time()
        print ('\n\t(*)Elapsed time in loading FITS: {0:.3f}\'\n'.
               format((t2-t1)/60.))

        '''HERE: make crosstalk corrections
        1) open_corr(): open the file with the coeffs
        2) crosscorr(): apply correction CCD by CCD
        '''
        df_cross,cols_cross = Correction.open_corr(cross_file)
        Correction.crosscorr(cls.fp_data_list,df_cross,cols_cross,cls.df_key)

        print '\n================exiting after crosscorr'; exit()

        print ('-fp_data size in the system: {0:.3F} kB'.
               format(sys.getsizeof(cls.fp_data_list)/1024.))
        
        '''Split by amplifier.
        FIX because need to be performed with the crosstalked arrays
        '''
        preXt_tab,df_sec = ManageCCD.split_amp(cls.fp_data_list,cls.df_key)
        cls.preXt = preXt_tab
        preXt_tab = None
        fp_data = None
        gc.collect()

        '''Here: extract overscan
        '''

        #sizes
        print ('-df_key/df_cross size in the system: {0:.3F}/{1:.3F} kB'.
               format(sys.getsizeof(cls.df_key)/1024.,
               sys.getsizeof(df_cross)/1024.))
        print ('-preXt_tab size in the system: {0:.3F} kB'.
               format(sys.getsizeof(cls.preXt)/1024.))

        #close HDU
        M_hdu.close()
        #
        return False

    @classmethod
    def focal_array(cls):
        '''Method to construct a big array containing all the CCDs in its
        spatial position in the detector.
        Even when Binary Search is better, the method 
        Table.will_query_use_indexing(condition_used_in_where) get as result 
        that this query wouldn't be indexed
        '''
        t1 = time.time()
        #iter_row = preXt_tab.row
        iter_row = ManageCCD.preXt
        #aux_fp = np.zeros((fp_size[1],fp_size[0]),dtype=float)
        aux_fp = np.zeros((ManageCCD.detsize[1],ManageCCD.detsize[0]),
                        dtype=float)
        max_r,max_c = 0,0
        #for iter_row in preXt_tab:
        for iter_row in ManageCCD.preXt:
            tmp_ccd = iter_row['ccdnum']
            #pos1a = Auxiliary.range_str(df_key.loc[df_key['ccdnum']==tmp_ccd,
            #                                       'detseca'].values[0])
            #pos1b = Auxiliary.range_str(df_key.loc[df_key['ccdnum']==tmp_ccd,
            #                                       'detsecb'].values[0])
            pos1a = Auxiliary.range_str(ManageCCD.df_key.loc[
                                        ManageCCD.df_key['ccdnum']==tmp_ccd,
                                        'detseca'].values[0])
            pos1b = Auxiliary.range_str(ManageCCD.df_key.loc[
                                        ManageCCD.df_key['ccdnum']==tmp_ccd,
                                        'detseca'].values[0])
            if pos1a[3] > max_r: max_r = pos1a[3]
            if pos1a[1] > max_c: max_c = pos1a[1]
            if pos1b[3] > max_r: max_r = pos1b[3]
            if pos1b[1] > max_c: max_c = pos1b[1]
            aux_fp[pos1a[2]:pos1a[3]+1,
                   pos1a[0]:pos1a[1]+1] = iter_row['datasec']
            aux_fp[pos1b[2]:pos1b[3]+1,
                   pos1b[0]:pos1b[1]+1] = iter_row['datasec']
        #shrink to the effective data size
        aux_fp = aux_fp[:max_r+1,:max_c+1]
        t2 = time.time()
        print ('\n\t(*)Elapsed time filling FP screenshot: {0:.3f}\'\n'.
               format((t2-t1)/60.))
        return aux_fp

        #ToDo:
        #1) extract any keyword value, CCD by CCD --done
        #2) split in amplifiers --done!
        #3) apply crosstalk correction
        #4) bias, trim, etc
        #from the correction file, appears that is applied ccd by ccd


if __name__=='__main__':
    print ()
    t01 = time.time()

    path = '/Users/fco/Code/shipyard_DES/raw_201608_hexa/'
    fname = 'DECam_00565152.fits.fz'
    crossname = 'DECam_20130606.xtalk'

    ManageCCD.open_file(path+fname,path+crossname)
    t02 = time.time(); print '\telapsed: {0:.3f}\''.format((t02-t01)/60.)

    ManageCCD.focal_array()

    print 'EXITING.....'; exit()
    print '\nOpening corrections' #; time.sleep(0.5)
    df_cross,cols_cross = Correction.open_corr(path+crossname)
    t03 = time.time()

    print '\t(*)Total elapsed time: {0:.3f}\'\''.format((t03-t02))

    print df_cross.iloc[10]





'''
Remember: is not traduction, is rewrite and enhance


getlistitems
typedef struct _replacement_list_
remember to print program usage
there are keys not copied from HDU[0]

crosstlak coefficients
the matrix can be simply loaded from raw_text file?

there are a lot of keys to be modified in crosstalk, so the feed list can be feeded form HDU instead of define by hand?

getopt_long_only

switch

this code is very verbose!

populate crosstalk matrices

(SEE COMMENTS ON NOTEBOOK_paper)

Obstype

Avoid verbose using lowercase/uppercase tranform

Intead of populate the code with the checks statements (hdu integrity, neccesary keywords), a module can be created which must do this verbose work

Syntaxis in line 1232?

Malloc() (memory full with garbage). Calloc() (memory full with zerores)

Line 1500: How it works without a parenthesis???

To run crosstalk corrections, the code flags the saturated pixels. It seeems like this steep must be performed before, or simply in other way


DECam_crosstalk DECam_00561867.fits.fz test1 --crostalk -linear


    TESTING SPACE NOTES
    if fitsio.FITS(filename)[0].get_extname().lower() != 'sci':
        print '\n\tHeader extension name is not SCI'
        exit(0)
    else:
        data = fitsio.FITS(filename)[0][:,:]

    with fitsio.FITS(filename) as hdu:
        data = hdu
        #print 'inside ',type(data), data[0], data[0][:,:]
        for iter_data in hdu:
            N_ext += 1
            list_extnm.append(iter_data.get_extname())




'''
