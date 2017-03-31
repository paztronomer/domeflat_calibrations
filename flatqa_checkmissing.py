'''Simple script to detect the missing EXPNUM entries in FLAT_QA, when 
crossmatching with EXPOSURE table. Generates bash files for run the above 
nights.
'''

import os
import sys
import time
import numpy as np
import despydb.desdbi as desdbi

class Toolbox():
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

class Compare():
    @classmethod
    def missing_sh(cls,dates,label,runsite,outfile,runNITE=False,runEXP=True):
        '''Method receives a data range, the jira label (remember this is 
        equivalent to reqnum), the site name on which to run, and the filename
        for the output bash script. Also 2 parameters to decide whether to make
        bash scripts on a nightly or exposure basis.
        - dates : list with 2 items [date1,date2] for which the search is done
        - label : jira summary label, to get the coeffs grouped under the same
        reqnum
        - runsite: string containing the ID of the site (fermigrid,descmp4,etc)
        '''
        #to get extrema of the range for look into flat_qa
        que1 = "select min(e.expnum), max(e.expnum) from exposure e "
        que1 += "where e.obstype='dome flat' "
        que1 += "and nite>={0} and nite<={1} ".format(dates[0],dates[1])
        #to get the list of expnums from exposure table
        que2 = "select e.expnum,e.nite from exposure e "
        que2 += "where obstype='dome flat' and nite>={0} and nite<={1} ".format(
            dates[0],dates[1])
        que2 += "order by e.expnum"
        res1 = Toolbox.dbquery(que1,['i4','i4'])
        res2 = Toolbox.dbquery(que2,['i4','i4'])
        #print res1.dtype.names
        #to get the expnums from flat_qa in the datarange from exposure 
        que3 = "select q.expnum,m.nite "
        que3 += "from flat_qa q, miscfile m, miscfile n, file_archive_info i "
        que3 += "where m.filename=q.filename "
        que3 += "and n.filename=i.filename "
        que3 += "and q.expnum between {0} and {1} ".format(
            res1['min(e.expnum)'][0],res1['max(e.expnum)'][0])
        que3 += "and m.pfw_attempt_id=n.pfw_attempt_id "
        que3 += "and m.expnum=n.expnum "
        que3 += "and m.filetype='compare_dflat_binned_fp'"
        que3 += "and n.filetype='pixcor_dflat_binned_fp'"
        que3 += "order by q.expnum"

        res3 = Toolbox.dbquery(que3,['i4','i4'])
        #compare both and save the missing ones in flat_qa (if any)
        if (res2['expnum'].shape[0] == res3['expnum'].shape[0]):
            raise ValueError('NO MISSING EXPNUM IN FLAT_QA')
        #crossmatch both arrays of EXPNUMs
        aux_expn = np.setxor1d(res2['expnum'],res3['expnum'])
        if runEXP:
            #exposure IDs must be unique, but a double check will be performed
            aux_expn = np.unique(aux_expn)
            #create as many plain text files as EXPNUMs has been found
            for k in aux_expn:
                with open(str(k)+'.csv','w') as plain_file:
                    plain_file.write(str(k))
            #create a unique bash file to run all the above expnum
            with open('expnum_'+outfile,'w') as bashfile:
                bashfile.write("#!/bin/bash\n")
                bashfile.write("#Created on {0}\n".format(time.ctime))
                bashfile.write("echo EXPNUM-based file\n")
                bashfile.write("echo {0}:START AT $(date) $hostname\n".format(
                            runsite))
                for m in aux_expn:
                    flat_txt = str(m)+'.csv'
                    auxN = res2['nite'][np.where(res2['expnum']==m)][0]
                    p1 = "submit_nitelycal.py --db_section db-desoper"
                    p1 += " --campaign fpazch_Y4N --eups_stack firstcut Y4N+2"
                    p1 += " --queue_size 1 "
                    p1 += " --target_site {0} --flatlist {1} ".format(runsite,
                                                                flat_txt)
                    p1 += " --nite {0} --jira_summary='{1}'".format(auxN,label)
                    p1 += " --count\n"
                    bashfile.write(p1)
                bashfile.write("echo {0}:END AT $(date) $hostname\n".format(
                            runsite))
        if runNITE:
            #array of unique nights associated to the above expnum
            aux_nite_tmp = np.array([res2['nite'][np.where(
                    res2['expnum']==m)][0] for m in aux_expn])
            aux_nite_tmp = np.unique(aux_nite_tmp)
            #erase such nites already contained in flat_qa
            aux_nite = np.setxor1d(aux_nite_tmp,res3['nite'][:])
            #write bash for each data range
            with open('nite_'+outfile,'w') as outfile:
                outfile.write("#!/bin/bash\n")
                outfile.write("#Created on {0}\n".format(time.ctime))
                outfile.write("echo NITE-based file\n")
                outfile.write("echo {0}:START AT $(date) $hostname\n".format(
                            runsite))
                for i in aux_nite:
                    p1 = "submit_nitelycal.py --db_section db-desoper" 
                    p1 += " --campaign fpazch_Y4N --eups_stack firstcut Y4N+2" 
                    p1 += " --queue_size 1 "
                    p1 += " --target_site {0} --nite {1}".format(runsite,i)
                    p1 += " --jira_summary='{0}'\n".format(label)
                    outfile.write(p1)
                outfile.write("echo {0}:END AT $(date) $hostname\n".format(
                            runsite))
        if (not runNITE) and (not runEXP):
            raise ValueError('No output file was selected to be created.')

if __name__=='__main__':
    Compare.missing_sh([20150731,20160212],'flatqa_y3','CampusClusterPrecal',
                    'flatqa_y3_missing_EXPNUM.sh',runEXP=True)
    
    '''
    NAME                 MINNITE  MAXNITE   MINEXPNUM  MAXEXPNUM
    -------------------------------- -------- -------- ---------- ----------
    SVE1                 20120911 20121228     133757     164457
    SVE2                 20130104 20130228     165290     182695
    Y1E1                 20130815 20131128     226353     258564
    Y1E2                 20131129 20140209     258621     284018
    Y2E1                 20140807 20141129     345031     382973
    Y2E2                 20141205 20150518     383751     438346
    Y3               20150731 20160212     459984     516846
    Y4               20160813 20170212     563912     573912
    '''
