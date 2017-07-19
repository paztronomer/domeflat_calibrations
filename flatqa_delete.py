'''
Script to call 2 other scripts in DESDM. One for changing the status of the 
files and other to delete the files from disk and from DB
*Remember to source the adequate filetypes before to call it*
'''
import os
import time
import subprocess
import numpy as np
import despydb.desdbi as desdbi
import shlex
import logging

class Utility():
    def dbquery(self,reqnum,user_db,help_txt=False):
        '''This method perform 2 queries, on the first selects the triplet plus
        pfw_attempt_id and in the second it performs a larger one on which
        gets all the filetypes availables for each pfw_attempt_id
        Input:
        - reqnum
        - user_db: user as listed in the DB, from which the corresponding 
        reqnum will be erased
        Returns:
        - structured arrays for both queries
        '''
        ##list all tables in the DB
        ##q_all = "select distinct OBJECT_NAME"
        ##q_all += " from DBA_OBJECTS"
        ##q_all += " where OBJECT_TYPE='TABLE' and OWNER='DES_ADMIN'"
        
        print '\tQuery on req/user: {0}/{1}'.format(reqnum,user_db)
        
        q1 = "select p.unitname,p.attnum,p.reqnum,p.id,p.user_created_by"
        q1 += " from pfw_attempt p"
        q1 += " where p.reqnum={0}".format(reqnum) 
        q1 += " and p.user_created_by='{0}'".format(user_db.upper())
        desfile = os.path.join(os.getenv('HOME'),'.desservices.ini')
        section = 'db-desoper'
        dbi = desdbi.DesDbi(desfile,section)
        if help_txt:
            help(dbi)
        cursor1 = dbi.cursor()
        cursor1.execute(q1)
        key = [item[0].lower() for item in cursor1.description]
        rows = cursor1.fetchall()
        data1 = np.rec.array(rows,dtype=[(key[0],'|S25'),(key[1],'i4'),
                            (key[2],'i4'),(key[3],'i4'),(key[4],'|S25')])
        print '\t# PFW_IDs={0}'.format(len(rows))
        
        aux_fill = []
        pfw_id = np.unique(data1['id'][:]) #or data1.id[i]
        for idx,pfw in enumerate(pfw_id):
            q2 = "select distinct(d.filetype), d.pfw_attempt_id"
            q2 += " from desfile d,file_archive_info i"
            q2 += " where d.pfw_attempt_id={0}".format(pfw)
            q2 += " and d.filename=i.filename"
            q2 += " order by d.pfw_attempt_id"
            cursor2 = dbi.cursor()
            cursor2.execute(q2)
            kw = [item[0].lower() for item in cursor2.description]
            rows2 = cursor2.fetchall()
            aux = '\tworking on pfw_attempt_id:'
            aux += '{0} (number of filetypes={1}, {2} of {3} IDs)'.format(
                pfw,len(rows2),idx+1,pfw_id.shape[0])
            for rr in rows2: 
                aux_fill.append(rr)
        if len(aux_fill) > 0:
            data2 = np.rec.array(aux_fill,dtype=[(kw[0],'a50'),(kw[1],'i4')])
            print '#PFW_IDs / #TOTAL_FILETYPES = {0} / {1}'.format(
                data1.shape[0],data2.shape[0])
            return data1,data2
        else:
            return None

    
    def data_status(self,unitname,reqnum,attnum,db='db-desoper',
                    mark='JUNK',modify=False):
        '''Method to call Michael's script to change the status of the files.
        Only files marked as JUNK can be deleted.
        Inputs:
        - unitname,reqnum,attnum: triplet
        - db
        - mark: 'JUNK' by default
        - modify: tell the script to really mark as JUNK or only display
        current status
        '''
        print '\n----------\n'
        print '\n\tChanging status for unit/req/att: {0}/{1}/{2}'.format(
            unitname,reqnum,attnum)
        try:
            if modify:
                cmd = "datastate.py --unitname {0}".format(unitname)
                cmd += " --reqnum {0}".format(reqnum)
                cmd += " --attnum {0}".format(attnum)
                cmd += " --section {0}".format(db)
                cmd += " --newstate {0}".format(mark)
                cmd += " --dbupdate"
            else:
                cmd = "datastate.py --unitname {0}".format(unitname)
                cmd += " --reqnum {0}".format(reqnum)
                cmd += " --attnum {0}".format(attnum)
                cmd += " --section {0}".format(db)
                cmd += " --newstate {0}".format(mark)
            cmd = shlex.split(cmd)
            ask = subprocess.Popen(cmd,stdout=subprocess.PIPE)
            output,error = ask.communicate()
            ask.wait()
            print '\n',output[output.find('Current'):]
        except:
            e = sys.exc_info()[0]
            print "Error: {0}".format(e)
            aux = 'Error in calling datastate.py \
                with {0} / {1} / {2}'.format(unitname,reqnum,attnum)
            logging.error(aux)
        return True
    
    
    def delete_junk(self,unitname,reqnum,attnum,filetype,pfw_attempt_id,
                    archive='desar2home',exclusion=None,del_opt='yes'):
        ''' 
        This method calls Doug's script, which deletes selected files,
        previously marked as JUNK. When number of files in DB differs from
        those in disk or if there was a problem in the deletion,
        saves the filetype and pfw_attempt_id on a list
        Inputs:
        - unitname,reqnum,attnum: triplet
        - filetype to be processed
        - pfw_attempt_id: redundant qwith the triplet
        - archive
        - exclusion: list of filetypes to be excluded from deletion
        - del_opt: asnwer to the script, options are yes/no/diff/print
        Returns:
        - list of filetypes and pfw_attempt_ids which causes problems in 
        deleting or has different number of entries in disk and in DB
        '''
        print '========DEL unit/req/att/filetype : {0}/{1}/{2}/{3}\n'.format(
            unitname,reqnum,attnum,filetype)

        checkThis = []
        cmd = "delete_files.py --section=db-desoper"
        cmd += " --filetype={0}".format(filetype)
        cmd += " --unitname={0} --reqnum={1} --attnum={2}".format(
            unitname,reqnum,attnum)
        cmd += " --archive={0}".format(archive)
        cmd = shlex.split(cmd)
        if exclusion is None:
            try:
                pB = subprocess.Popen(cmd,shell=False,
                                    stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    universal_newlines=True)
                outM,errM = pB.communicate(input=del_opt)
                pB.wait()
                tmp = outM.replace('=','').replace('\n',' ').split()
                Ndisk = tmp[[i+1 for i,x in enumerate(tmp) 
                        if (x=='disk' and tmp[i-1]=='from')][0]]
                Ndb = tmp[[i+1 for i,x in enumerate(tmp) 
                        if (x=='db' and tmp[i-1]=='from')][0]]
                Ndisk,Ndb = np.int(Ndisk),np.int(Ndb)
                if ((Ndb != Ndisk) or ('No files on disk' in outM)):
                    print (filetype,pfw_attempt_id,'db and disk differs')
                    checkThis.append((filetype,pfw_attempt_id,
                                    'db and disk differs'))
            except:
                e = sys.exc_info()[0]
                print "Error: {0}".format(e)
                print (filetype,pfw_attempt_id,'ERROR')
                checkThis.append((filetype,pfw_attempt_id,
                                'problem deleting this filetype'))
            
        else:
            if filetype not in exclusion:
                try:
                    pC = subprocess.Popen(cmd,shell=False,
                                        stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        universal_newlines=True)
                    outM,errM = pC.communicate(input=del_opt)
                    pC.wait()
                    tmp = outM.replace('=','').replace('\n',' ').split()
                    Ndisk = tmp[[i+1 for i,x in enumerate(tmp) 
                            if (x=='disk' and tmp[i-1]=='from')][0]]
                    Ndb = tmp[[i+1 for i,x in enumerate(tmp) 
                            if (x=='db' and tmp[i-1]=='from')][0]]
                    Ndisk,Ndb = np.int(Ndisk),np.int(Ndb)
                    if ((Ndb != Ndisk) or ('No files on disk' in outM)):
                        print (filetype,pfw_attempt_id,'db and disk differs')
                        checkThis.append((filetype,pfw_attempt_id,
                                        'db and disk differs'))
                except:
                    e = sys.exc_info()[0]
                    print "Error: {0}".format(e)
                    print (filetype,pfw_attempt_id,'ERROR')
                    checkThis.append((filetype,pfw_attempt_id,
                                    'problem deleting this filetype'))
            else:
                print 'Filetype not allowed to be erased: {0}'.format(filetype)
        return checkThis
    
    
    def do_job(self,reqnum,usernm,keep=None,remove=True):
        '''Wrapper to call the change status and deletion
        Inputs:
        - reqnum
        - usernm: string for the username, as listed in DB
        - keep: list of filetypes to be kept (not deleted)
        - remove: to remove or not the files
        '''
        resquery = Utility().dbquery(reqnum,usernm)
        if resquery is None:
            print '\n\t(WARNING) Reqnum {0} has no DB entries\n'.format(reqnum)
        else:
            arr1,arr2 = resquery
            auxOut = []
            #iterate over pfw_attempt_ids
            for r in xrange(arr1.shape[0]):
                att,req = arr1['attnum'][r],arr1['reqnum'][r]
                unit,pfw_id = arr1['unitname'][r],arr1['id'][r]
                Utility().data_status(unit,req,att,modify=remove)
                #iterate over filetypes for the above pfw_attempt_id
                ftype = arr2[np.where(arr2['pfw_attempt_id']==pfw_id)]
                ftype = ftype['filetype']
                ftype = np.unique(ftype)
                for ft in ftype:
                    try:
                        #call the deletion
                        auxOut += Utility().delete_junk(unit,req,att,ft,pfw_id,
                                                    exclusion=keep)
                    except:
                        print '\n\tFailure on deleting {0}/{1}/{2}/{3}'.format(
                            unit,req,att,ft)
            print '\n\nCheck the below Filetypes/PFW_ID/Flags:\n{0}'.format(
                auxOut)
            if len(auxOut) > 0:
                datacheck = np.rec.array(auxOut,dtype=[('filetype','a50'),
                                                    ('pfw_attempt_id','i4'),
                                                    ('flag','a80')])
                tmp = 'checkPixcor_flatDEL_r'+str(reqnum)+'.csv'
                auxname = os.path.join(os.path.expanduser('~'),
                                    'Result_box/logs_flatDelete',tmp)
                np.savetxt(auxname,datacheck,delimiter=',',
                        fmt='%-s50,%d,%-s100',
                        header='filetype,pfw_attempt_id,flag')
        return True


if __name__=='__main__':
    print '\n**********\nRemember to submit from DESSUB or DESAR2\n**********'
    toCall = Utility()
    #reqnum_delete = [2777,2769,2776,2756,2748,2751,2743,2744]
    #reqnum_delete = [2782,2783,2807,2808]
    #reqnum_delete = [2915,2916,2917]
    #reqnum_delete = [2808,2807]
    reqnum_delete = np.loadtxt("DeleteReqnum_20170510.csv",dtype="int")
    #keepFiles = ['pixcor_dflat_binned_fp']
    for req in reqnum_delete:
        print '\n\n========================\n\tREQNUM={0}'.format(req)
        print '\t{0}\n========================'.format(time.ctime())
        toCall.do_job(req,'fpazch') #keepfiles

    print time.ctime()
    print '\n (THE END)'  
