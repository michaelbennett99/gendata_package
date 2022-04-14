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
    
    def write_bin(self, out_file: str):
        """
        Write GRM in GCTA .bin format.
        """
        pass
    
    def write_gz(self, out_file: str):
        """
        Write GRM in GCTA .gz format.

        x.grm.gz has no header line; columns are indices of pairs of individuals
        (row numbers of the test.grm.id), number of non-missing SNPs and the
        estimate of genetic relatedness.

        x.grm.id has no header line; columns are family ID and individual ID.

        Args:
            out_file (str): Path to write GRM to. Do not include final part of
                suffix (i.e. .grm.gz or .grm.id).
        """
        ix_tril = np.tril_indices_from(self.grm)
        
        gz_df = pd.DataFrame(
            data={
                "i": ix_tril[0] + 1, "j": ix_tril[1] + 1,
                "n": np.tril(self.n_snps)[ix_tril], "a": self.grm[ix_tril]
            }
        )
        gz_df.to_csv(
            f"{out_file}.grm.gz",
            sep="\t", index=False, header=False, compression="gzip"
        )

        self.ids.to_csv(f"{out_file}.grm.id", sep="\t", index=False, header=False)
