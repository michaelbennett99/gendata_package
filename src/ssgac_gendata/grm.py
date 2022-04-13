"""
Module to hold classes and functions related to genetic relationship matrices.
"""

import numpy as np
import pandas as pd

class GRM():
    def __init__(
            self, grm_matrix: np.ndarray, ids: pd.DataFrame, n_snps: np.ndarray
        ):
        self.grm = grm_matrix
        self.ids = ids
        self.n_snps = n_snps
    
    def write_bin(out_file: str):
        """
        Write GRM in GCTA .bin format.
        """
        pass
    
    def write_gz(out_file: str):
        """
        Write GRM in GCTA .gz format.

        x.grm.gz has no header line; columns are indices of pairs of individuals
        (row numbers of the test.grm.id), number of non-missing SNPs and the
        estimate of genetic relatedness.

        x.grm.id has no header line; columns are family ID and individual ID.
        """
        pass
