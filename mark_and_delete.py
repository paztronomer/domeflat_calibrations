import os
import time
import subprocess
import numpy as np
import despydb.desdbi as desdbi
import shlex
import logging

class Utility():
    def query_run(self,reqnum=None,user_db=None,attnum=None,
                help_txt=False):
        """Method to query all or a subset of attnum, for a given reqnum,
        checking if the file exists 
        Input:
        - reqnum: integer
        - attnum: list of integers or "all" string
        - user_db: user as listed in the DB, from which the corresponding
        reqnum will be erased
        Returns:
        - structured array with query
        """
        ##list all tables in the DB
        ##q_all = "select distinct OBJECT_NAME"
        ##q_all += " from DBA_OBJECTS"
        ##q_all += " where OBJECT_TYPE="TABLE" and OWNER="DES_ADMIN""
        if (reqnum is None) or (user_db is None) or (attnum is None):
            print "No enoghn data for query"
            exit(1)
        
        if (attnum == "all"):
            aux_q1 = ""
        elif isinstance(attnum,list):
            aux_q1 = " and a.attnum in (" + ",".join(map(str,attnum)) + ")"

        print "\tQuery on req/user/attnums: {0}/{1}/{2}".format(reqnum,
                                                            user_db.upper(),
                                                            attnum)
        q1 = "select a.unitname,a.attnum,a.reqnum,a.id,a.user_created_by,"
        q1 += " a.data_state,fai.path"
        q1 += " from pfw_attempt a,file_archive_info fai,desfile d"
        q1 += " where a.reqnum={0}".format(reqnum)
        q1 += aux_q1
        q1 += " and a.user_created_by=\'{0}\'".format(user_db.upper())
        q1 += " and a.id=d.pfw_attempt_id"
        q1 += " and fai.filename=d.filename"
        q1 += " order by a.id"
        desfile = os.path.join(os.getenv("HOME"),".desservices.ini")
        section = "db-desoper"
        dbi = desdbi.DesDbi(desfile,section)
        if help_txt:
            help(dbi)
        cursor1 = dbi.cursor()
        cursor1.execute(q1)
        key = [item[0].lower() for item in cursor1.description]
        rows = cursor1.fetchall()
        if (len(rows) > 0):
            data1 = np.rec.array(rows,dtype=[(key[0],"|S25"),(key[1],"i4"),
                                (key[2],"i4"),(key[3],"i4"),(key[4],"|S25"),
                                (key[5],"|S10"),(key[6],"|S200")])
            print "\t# PFW_IDs={0}".format(len(rows))
            return data1
        else:
            print "\t# PFW_IDs={0} ==> NO elements on disk".format(len(rows))
            return None


    def data_status(self,unitname=None,reqnum=None,attnum=None,
                    db="db-desoper",mark="JUNK",modify=False):
        """Method to call Michael"s script to change the status of the files.
        Only files marked as JUNK can be deleted.
        Inputs:
        - unitname,reqnum,attnum: triplet
        - db
        - mark: "JUNK" by default
        - modify: tell the script to really mark as JUNK or only display
        current status
        """
        print "\t--------------------------------------------------"
        print "\tChanging status for unit/req/att: {0}/{1}/{2}".format(
            unitname,reqnum,attnum)
        try:
            cmd = "datastate.py --unitname {0}".format(unitname)
            cmd += " --reqnum {0}".format(reqnum)
            cmd += " --attnum {0}".format(attnum)
            cmd += " --section {0}".format(db)
            cmd += " --newstate {0}".format(mark)
            if modify:
                cmd += " --dbupdate"
            cmd = shlex.split(cmd)
            ask = subprocess.Popen(cmd,stdout=subprocess.PIPE)
            output,error = ask.communicate()
            ask.wait()
            print "\t{0}".format(output)
            print "\t--------------------------------------------------"
        except:
            aux = "Error in calling datastate.py \
                with {0} / {1} / {2}".format(unitname,reqnum,attnum)
            logging.error(aux)
        return True


    def delete_all(self,unitname=None,reqnum=None,attnum=None,
                archive="desar2home",section="db-desoper",
                del_opt="yes"):
        """ 
        This method calls Doug"s script, which deletes ALL the files
        belonging to the triplet, previously marked as JUNK. 
        When number of files in DB differs from
        those in disk or if there was a problem in the deletion,
        pfw_attempt_id on a list
        Inputs:
        - unitname,reqnum,attnum: triplet
        - archive
        - section
        - del_opt: asnwer to the script, options are yes/no/diff/print
        Returns:
        """
        print "Deleting unit/req/att: {0}/{1}/{2}\n".format(
            unitname,reqnum,attnum)
        checkThis = []
        cmd = "delete_files.py" 
        cmd += " --section {0}".format(section)
        cmd += " --unitname {0}".format(unitname)
        cmd += " --reqnum {0}".format(reqnum)
        cmd += " --attnum {0}".format(attnum)
        cmd += " --archive={0}".format(archive)
        cmd = shlex.split(cmd)
        try:
            pB = subprocess.Popen(cmd,shell=False,
                                stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                universal_newlines=True)
            outM,errM = pB.communicate(input=del_opt)
            pB.wait()
            tmp = outM.replace("=","").replace("\n"," ").split()
            Ndisk = tmp[[i+1 for i,x in enumerate(tmp) 
                    if (x=="disk" and tmp[i-1]=="from")][0]]
            Ndb = tmp[[i+1 for i,x in enumerate(tmp) 
                    if (x=="db" and tmp[i-1]=="from")][0]]
            Ndisk,Ndb = np.int(Ndisk),np.int(Ndb)
            if ((Ndb != Ndisk) or ("No files on disk" in outM)):
                print (reqnum,"db and disk differs")
                checkThis.append((reqnum,"db and disk differs"))
        except:
            print (reqnum,"not found")
            checkThis.append((reqnum,"problem deleting reqnum"))
        return checkThis
    
    
    def caller1(self,reqnum=None,attnum=None,user_db=None,
            mark_junk=None,delete=None):
        """Wrapper to call the change status and deletion
        Inputs:
        - reqnum
        - user_db: string for the username, as listed in DB
        - mark_junk: whether to mark as junk or not the run
        - delete: to remove or not the files from disk
        """
        kw0 = {"reqnum":reqnum,"user_db":user_db,"attnum":attnum}
        res = Utility().query_run(**kw0)
        auxOut = []

        if not (res is None):
            #walk through each triplet, removing duplicates (many filenames)
            rq = np.unique(res["reqnum"])[0]
            x0 = np.unique(res["attnum"])
            for att in x0:
                x1 = np.unique(res[np.where(res["attnum"]==att)]["unitname"])
                for uni in x1:
                    kw1 = {"reqnum":rq,"attnum":att,"unitname":uni}
                    kw1["db"] = "db-desoper"
                    if mark_junk:
                        kw1["mark"] = "JUNK"
                        kw1["modify"] = True
                    else:
                        kw1["modify"] = False
                    Utility().data_status(**kw1)
                    kw2 = {"reqnum":rq,"attnum":att,"unitname":uni}
                    if delete:
                        auxOut += Utility().delete_all(**kw2)

            if len(auxOut) > 0:
                datacheck = np.rec.array(auxOut,dtype=[("reqnum","i4"),
                                                    ("flag","a80")])
                tmp = "deleting_issue_r"+str(reqnum)+".csv"
                auxname = os.path.join(os.path.expanduser("~"),
                                    "Result_box/logs_flatDelete",tmp)
                np.savetxt(auxname,datacheck,delimiter=",",
                        fmt="%-s50,%d,%-s100",
                        header="reqnum,flag")
        return True

if __name__ == "__main__":
    """for deletion, must run in desar2... warning!!!
    """
    reqnum_list = [2907,2908,2920]
    #[2863,2864,2865,2866,2867,2868,2869,2870,2871,2872,2873,2874,2875]
    #[2623,2625,2743,2748,2751,2756,2769,2776,2777,2782,2783,2807,2808]
    for rq in [2877]:#reqnum_list:
        kw = {"user_db":"fpazch","attnum":"all","mark_junk":True}
        kw["delete"] = False
        kw["reqnum"] = rq
        Utility().caller1(**kw)
