"""
A file for constants for the genetic data handling module.
"""

# PYSNTOOLS Constants
PYSNPTOOLS_MISSING_VAL = -127
PYSNPTOOLS_IID_COL = 1

# Genetic Data Constants
ERRLIST_MAXLEN = 10

# Naming constants
RSID = "rsID"
CHR = "chr"
GENPOS = "genpos"
BPOS = "bpos"
A1 = "a1"
A2 = "a2"
BIM_COLS = {0: CHR, 1: RSID, 2: GENPOS, 3: BPOS, 4: A1, 5: A2}

ID = "ID"
FID = "FID"
IID = "IID"
FFID = "FFID"
MFID = "MFID"
SEX = "SEX"
PHEN = "PHEN"
FAM_COLS = {0: FID, 1: IID, 2: FFID, 3: MFID, 4: SEX, 5: PHEN}

BLOCK = "block"

# LD Matrix Output Constants
LDM = "ldm"
VARINFO = "varinfo"
