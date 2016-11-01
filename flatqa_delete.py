'''Simple script to change status and the delete files from archive
Remember to source the adequate files before to call it
'''
import os
import time
import subprocess
import numpy as np
import despydb.desdbi as desdbi

class Utility():
    @classmethod
    def dbquery(cls,reqnum,user_db,help_txt=False):
        '''This method perform 2 queries, on the fisrt selects the triplet plus
        pfw_attempt_id and in the second it performs a larger one on which
        gets all the filetypes availables for each pfw_attempt_id
        '''
        #list all tables in the DB
        q_all = "select distinct OBJECT_NAME from DBA_OBJECTS where \
                OBJECT_TYPE='TABLE' and OWNER='DES_ADMIN'"
        
        #input of triplet plus unique pfw_attempt_id (56 rows)
        q1 = "select p.unitname,p.attnum,p.reqnum,p.id,p.user_created_by \
            from pfw_attempt p where p.reqnum={0}".format(reqnum)
        
        desfile = os.path.join(os.getenv('HOME'),'.desservices.ini')
        section = 'db-desoper'
        dbi = desdbi.DesDbi(desfile,section)
        if help_txt:
            help(dbi)
        
        cursor1 = dbi.cursor()
        cursor1.execute(q1)
        key = [item[0].lower() for item in cursor1.description]
        rows = cursor1.fetchall()
        #setup all cols as integers
        data1 = np.rec.array(rows,dtype=[(key[0],'i4'),(key[1],'i4'),
                            (key[2],'i4'),(key[3],'i4'),(key[4],'a25')])

        if ( len(np.unique(data1['user_created_by']))==1 and 
            np.unique(data1['user_created_by'])[0].lower()==user_db ):
            pass
        else:
            print '(!) ERROR: not unique user: {0}'.format(np.unique(data1['user_created_by']))
            exit()

        #a loop is faster than make an unique query
        aux_fill = []
        for i in xrange(data1.shape[0]):
            pfw_id = data1.id[i] #or data1['id'][i]
            print '\tworking on pfw_attempt_id:{0}'.format(pfw_id)
            q2 = 'select distinct(d.filetype),d.pfw_attempt_id from desfile d \
                where d.pfw_attempt_id={0}'.format(pfw_id)
            cursor2 = dbi.cursor()
            cursor2.execute(q2)
            kw = [item[0].lower() for item in cursor2.description]
            rows = cursor2.fetchall()
            for i in rows: 
                #picks every tuple inside the list
                aux_fill.append(i)

        data2 = np.rec.array(aux_fill,dtype=[(kw[0],'a50'),(kw[1],'i4')])
        return data1,data2
    

    @classmethod
    def data_status(cls,unitname,reqnum,attnum,db='db-desoper',
                    mark='JUNK',modify=False):
        try:
            if modify:
                cmd = 'datastate.py --unitname {0} --reqnum {1} --attnum {2} \
                    --section {3} --newstate {4} --dbupdate'.format(unitname,
                                                                    reqnum,
                                                                    attnum,
                                                                    db,mark)
            else:
                cmd = 'datastate.py --unitname {0} --reqnum {1} --attnum {2} \
                    --section {3} --newstate {4}'.format(unitname,reqnum,attnum,
                                                        db,mark)
            ask = subprocess.Popen(cmd.split(),stdout=subprocess.PIPE)
            output,error = ask.communicate()
        except:
            print 'Error in calling datastate.py \
                with {0} / {1} / {2}'.format(unitname,reqnum,attnum)


    @classmethod
    def delete_junk(cls,unitname,reqnum,attnum,filetype,archive='desar2home',
                exclusion=['compare_dflat_binned_fp','xtalked_dflat'],
                del_opt='yes'):
        '''Different options for del_opt: yes/no/diff/print
        '''
        print '\tDeleting unit/req/att/filetype : {0}/{1}/{2}/{3}'.format(
                                                                    unitname,
                                                                    reqnum,
                                                                    attnum,
                                                                    filetype)
        if filetype not in exclusion:
            cmd = 'delete_files.py --archive {0} --filetype {1}\
                --unitname {1} --reqnum {2} --attnum {3}'.format(archive,
                                                                filetype,
                                                                unitname,
                                                                reqnum,attnum) 
            p = subprocess.Popen(cmd.split(),stdin=subprocess.PIPE)
            p.communicate(input=del_opt)
            p.wait()
            time.sleep(30)
        else:
            print 'Filetype not allowed to be erased: {0}'.format(filetype)


    @classmethod
    def do_job(cls,reqnum,usernm):
        arr1,arr2 = Utility.dbquery(reqnum,usernm)
        for r in xrange(arr1.shape[0]):
            att,req = arr1['attnum'][r],arr1['reqnum'][r]
            unit,pfw_id = arr1['unitname'][r],arr1['id'][r]
            ftype = arr2['filetype'][np.where(arr2['pfw_attempt_id']==pfw_id)]
            for ft in ftype:
                try:
                    Utility.delete_junk(unit,req,att,ft)
                except:
                    print 'Failure on deleting {0}/{1}/{2}/{3}'.format(unit,
                                                                    req,att,ft)


if __name__=='__main__':
    print '\n**********\nRemember to submit from DESSUB or DESAR2\n**********'
    Utility.do_job(2625,'fpazch')

    print '\n (THE END)'  
