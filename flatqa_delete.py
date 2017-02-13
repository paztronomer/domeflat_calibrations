'''Simple script to change status and the delete files from archive
Remember to source the adequate files before to call it
'''
import os
import time
import subprocess
import numpy as np
import despydb.desdbi as desdbi

class Utility():
    
    def dbquery(self,reqnum,user_db,help_txt=False):
        '''This method perform 2 queries, on the first selects the triplet plus
        pfw_attempt_id and in the second it performs a larger one on which
        gets all the filetypes availables for each pfw_attempt_id
        '''
        #list all tables in the DB
        q_all = "select distinct OBJECT_NAME"
        q_all += " from DBA_OBJECTS"
        q_all += " where OBJECT_TYPE='TABLE' and OWNER='DES_ADMIN'"
        print '\tQuery on req/user: {0}/{1}'.format(reqnum,user_db)
        #input of triplet plus unique pfw_attempt_id (56 rows)
        q1 = "select p.unitname,p.attnum,p.reqnum,p.id,p.user_created_by"
        q1 += " from pfw_attempt p"
        q1 += " where p.reqnum={0} and p.user_created_by='{1}'".format(reqnum,
                                                                user_db.upper())
        desfile = os.path.join(os.getenv('HOME'),'.desservices.ini')
        section = 'db-desoper'
        dbi = desdbi.DesDbi(desfile,section)
        if help_txt:
            help(dbi)
        
        cursor1 = dbi.cursor()
        cursor1.execute(q1)
        #time.sleep(3)
        key = [item[0].lower() for item in cursor1.description]
        rows = cursor1.fetchall()
        print '\tN={0}'.format(len(rows))
        data1 = np.rec.array(rows,dtype=[(key[0],'i4'),(key[1],'i4'),
                            (key[2],'i4'),(key[3],'i4'),(key[4],'a25')])
        
        if ( len(np.unique(data1['user_created_by']))==1 and 
            np.unique(data1['user_created_by'])[0].lower()==user_db ):
            pass
        else:
            aux = 'ERROR: not unique user: {0}'.format(np.unique(
                                                    data1['user_created_by']))
            raise ValueError(aux)

        #a loop is faster than make an unique query
        aux_fill = []
        for i in xrange(data1.shape[0]):
            pfw_id = data1.id[i] #or data1['id'][i]
            #WARNING : REVIEW THE QUERIES! SOME LOOK NOT GOOD
            q2 = "select distinct(d.filetype), d.pfw_attempt_id"
            q2 += " from desfile d"
            q2 += " where d.pfw_attempt_id={0}".format(pfw_id)
            cursor2 = dbi.cursor()
            cursor2.execute(q2)
            kw = [item[0].lower() for item in cursor2.description]
            rows2 = cursor2.fetchall()
            print '\tworking on pfw_attempt_id:{0} (N={1}, {2} of {3})'.format(
                                    pfw_id,len(rows2),i+1,data1.shape[0])
            if False: Utility.check_files(pfw_id,ft)
            for i in rows2: 
                #picks every tuple inside the list
                aux_fill.append(i)
        data2 = np.rec.array(aux_fill,dtype=[(kw[0],'a50'),(kw[1],'i4')])
        print 'N_q1/N_q2 = {0}/{1}'.format(data1.shape[0],data2.shape[0])
        return data1,data2
    

    def data_status(self,unitname,reqnum,attnum,db='db-desoper',
                    mark='JUNK',modify=False):
        print '\n----------\n'
        print '\n\tChanging status for unit/req/att: {0}/{1}/{2}'.format(
                                                                unitname,
                                                                reqnum,attnum)
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
            ask = subprocess.Popen(cmd.split(),stdout=subprocess.PIPE)
            output,error = ask.communicate()
            ask.wait()
            
            #INFO
            aux_info = "datastate.py"
            aux_info += " --unitname {0}".format(unitname)
            aux_info += " --reqnum {0}".format(reqnum)
            aux_info += " --attnum {0}".format(attnum)
            aux_info += " --section {0}".format(db)
            aux_info += " --newstate {0}".format(mark)
            outMsg,errMsg = subprocess.Popen(aux_info.split(),
                                        stdout=subprocess.PIPE).communicate()
            print '\n',outMsg[outMsg.find('Current'):]
        
        except:
            aux = 'Error in calling datastate.py \
                with {0} / {1} / {2}'.format(unitname,reqnum,attnum)
            raise ValueError(aux)


    def delete_junk(self,unitname,reqnum,attnum,filetype,pfw_attempt_id,
                archive='desar2home',
                exclusion=['pixcor_dflat_binned_fp'],
                del_opt='yes'):
        '''To keep xtalked files: 
        Different options for del_opt: yes/no/diff/print
        setup -v firstcut Y4N+2
        '''
        checkThis = []
        print '========DEL unit/req/att/filetype : {0}/{1}/{2}/{3}\n'.format(
                                                                    unitname,
                                                                    reqnum,
                                                                    attnum,
                                                                    filetype)
        if filetype not in exclusion:
            cmd = 'delete_files.py --section=db-desoper --filetype={1} \
                --unitname={2} --reqnum={3} --attnum={4} --archive={0}'.format(
                                                                archive,
                                                                filetype,
                                                                unitname,
                                                                reqnum,attnum) 
            try:
                #p = subprocess.Popen(cmd.split(),stdin=subprocess.PIPE)
                p = subprocess.Popen(cmd.split(),shell=False,
                                    stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE) 
                outM,errM = p.communicate()#(input=del_opt)
                #check if same amount of files on DB and DISK
                tmp = outM.replace('=','').replace('\n',' ').split()
                Ndisk = np.int(tmp[[i+1 for i,x in enumerate(tmp) 
                                    if (x=='disk' and tmp[i-1]=='from')][0]]) 
                Ndb = np.int(tmp[[i+1 for i,x in enumerate(tmp) 
                                    if (x=='db' and tmp[i-1]=='from')][0]])
                
                if ((Ndb != Ndisk) or ('No files on disk' in outM)):
                    print (filetype,pfw_attempt_id,'db and disk differs')
                    checkThis.append((filetype,pfw_attempt_id,
                                    'db and disk differs'))
                    p = subprocess.Popen(cmd.split(),shell=False,
                                        stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE)
                    p.communicate(input=del_opt)
                else:
                    p = subprocess.Popen(cmd.split(),shell=False,
                                        stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE)
                    p.communicate(input=del_opt)
                #p.communicate(input=del_opt)
                #p.wait()
            except:
                print (filetype,pfw_attempt_id,'not found')
                checkThis.append((filetype,pfw_attempt_id,'not found'))
        else:
            print 'Filetype not allowed to be erased: {0}'.format(filetype)
        return checkThis

    def do_job(self,reqnum,usernm):
        #auxOut = Utility().delete_junk(20160808,2625,1,'xtalked_bias',579388)
        arr1,arr2 = Utility().dbquery(reqnum,usernm)
        auxOut = []
        for r in xrange(arr1.shape[0]):
            att,req = arr1['attnum'][r],arr1['reqnum'][r]
            unit,pfw_id = arr1['unitname'][r],arr1['id'][r]
            Utility().data_status(unit,req,att)
            ftype = arr2['filetype'][np.where(arr2['pfw_attempt_id']==pfw_id)]
            for ft in ftype:
                try:
                    auxOut += Utility().delete_junk(unit,req,att,ft,pfw_id)
                except:
                    print '\n\tFailure on deleting {0}/{1}/{2}/{3}'.format(unit,
                                                                    req,att,ft)
        print '\n\nCheck the below Filetypes/PFW_ID/Flags:\n{0}'.format(auxOut)
        if len(auxOut) > 0:
            datacheck = np.rec.array(auxOut,dtype=[('filetype','a50'),
                                                ('pfw_attempt_id','i4'),
                                                ('flag','a80')])
            auxname = '/home/fpazch/Result_box/logs_flatDelete/'
            auxname += 'checkPixcor_flatDEL_r'+str(reqnum)+'.csv'
            np.savetxt(auxname,
                    datacheck,delimiter=',',fmt='%s,%d,%s',
                    header='filetype,pfw_attempt_id,flag')


if __name__=='__main__':
    print '\n**********\nRemember to submit from DESSUB or DESAR2\n**********'
    toCall = Utility()
    #reqnum_delete = [2777,2769,2776,2756,2748,2751,2743,2744]
    reqnum_delete = [2782,2783,2807,2808]
    for req in reqnum_delete:
        print '\n\n========================\n\tREQNUM={0}'.format(req)
        print '\t{0}\n========================'.format(time.ctime())
        toCall.do_job(req,'fpazch')

    print '\n (THE END)'  
