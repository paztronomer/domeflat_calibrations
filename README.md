# Codes for dome flat analysis, using Discrete Wavelet Transform

Scripts for Discrete Wavelet Transform to be used in dome flats, as well as auxiliary tasks. Written for Python 2.7, but easily adaptable for 3. When doing that, be aware of changes in **pandas** and other modules.

**Main:**
* flatDWT.py: discrete Meyer wavelet in 2D, for binned FP		

**Auxiliaries:**
* statDWT_dflat.py: extracts statistics from the DWT matrices
* scalMask_dflat.py: scale masks, using nearest neighbors match when changing dimensions  
* flatqa_delete.py: deletes runs from DB and from disk

**Others:**
* despydb_test.py: just to remember syntaxis of query from despydb
* flatqa_checkmissing.py: to check for missing expnums. Wrong behaviors. Must be fixed
