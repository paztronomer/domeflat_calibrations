import os
import numpy as np
import despydb.desdbi as desdbi

def TestDB(dbsection='db-desoper',help_txt=False):
    datatype = ['a80','i4','a5','i4','f4']
    query = "select i.path,m.pfw_attempt_id,m.band,m.nite,f.expnum from \
            flat_qa f,file_archive_info i, miscfile m where m.nite=20160808 \
            and m.filename=f.filename and i.filename=f.filename and\
            m.filetype='compare_dflat_binned_fp'"
    desfile = os.path.join(os.getenv('HOME'),'.desservices.ini')
    section = dbsection
    print 'check 0'
    dbi = desdbi.DesDbi(desfile,section)
    print 'check 1'
    if help_txt: help(dbi)
    cursor = dbi.cursor()
    cursor.execute(query)
    print 'check 2'
    cols = [line[0].lower() for line in cursor.description]
    rows = cursor.fetchall()
    outtab = np.rec.array(rows,dtype=zip(cols,datatype))
    return outtab

if __name__=='__main__':
    print TestDB()
