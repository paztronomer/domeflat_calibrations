# flat_wise
# ex:bergamota (*devel*)
té con bergamota, backups de devel

Scripts for Discrete Wavelet Transform to be used in dome flats, to deal with crosstalk, and auxiliary for this two main tasks.

**Main:**
* crosstalk_FP.py: perform crosstalk		
* flatDWT.py: discrete Meyer wavelet in 2D, for binned FP		

**Auxiliaries:**
* statDWT_dflat.py: extracts statistics from the DWT matrices
* scalMask_dflat.py: scale masks, using nearest neighbors match when changing dimensions  
* flatqa_delete.py: deletes runs from DB and from disk

**Others:**
* despydb_test.py: just to remember syntaxis of query from despydb
* flatqa_checkmissing.py: to check for missing expnums. Wrong behaviors. Must be fixed
