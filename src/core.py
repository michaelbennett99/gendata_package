"""
A module containing functions and classes to facilitate the importing of
genetic data.
"""

import warnings

from math import floor

import numpy as np
import pandas as pd
import scipy.sparse as sp

from multiprocessing import Pool
from numpy.random.bit_generator import BitGenerator
from numpy.typing import NDArray
from numba import vectorize
from numba.core.types import float64, int64, int32, boolean
from pysnptools.snpreader import Bed, SnpData
from typing import Any, Iterable, Optional, Union, Type

from .constants import *
from .cov import make_cov, make_weighted_cov, make_spwindow_cov, make_window_cov
from .grm import GRM


class AbstractGenoData:
    """An abstract class for other genotype data classes to be based on.

    This class will include methods that can act generally on all possible
    specific forms of genotype data.
    """
    # Add axis constants
    SNP_AXIS = 0
    SAMPLES_AXIS = 1

    SORT_KEYS = [CHR, BPOS]

    COUNT_COL = "count"

    def __init__(
            self,
            genotypes: pd.DataFrame,
            snps: pd.DataFrame,
            samples: pd.DataFrame
        ):
        """Initialises an AbstractGenoData object.

        Args:
            genotypes (pd.DataFrame): Matrix of genotypes, indexed by rsID on
                the rows and IID on the columns.
            snps (pd.DataFrame): Table of SNP information for the genotype
                matrix indexed by rsID.
            samples (pd.DataFrame): Table of sample information for the genotype
                matrix indexed by IID.

        Raises:
            ValueError: Shapes of provided inputs do not align.
            ValueError: rsIDs do not match between genotypes and SNPs.
            ValueError: IIDs do not match between genotypes and samples.
        """
        ## Check shape.
        if genotypes.shape != (len(snps), len(samples)):
            err_mssg = (
                f"Provided genotype array {genotypes.shape} is not "
                f"compatible with length of snp and sample datasets "
                f"({len(snps)}, {len(samples)})"
            )
            raise ValueError(err_mssg)

        # Check SNP match
        if len(genotypes.index.symmetric_difference(snps.index)) != 0:
            err_mssg = "SNPs do not match between genotype and SNP info data."
            raise ValueError(err_mssg)

        # Check sample match
        if len(genotypes.index.symmetric_difference(snps.index)) != 0:
            err_mssg = (
                "Samples do not match between genotype and sample info data.")
            raise ValueError(err_mssg)

        # Check nonzero snps and samples
        if genotypes.count(axis=self.SAMPLES_AXIS).sum() == 0:
            raise ValueError("Genetic data object contains no valid data.")

        # Confirm SNP order
        snps = snps.sort_values(self.SORT_KEYS)

        # Apply right order to genotype df
        genotypes = genotypes.loc[snps.index, samples.index] # type: ignore

        # Set attributes
        self.genotypes = genotypes
        self.snps = snps
        self.samples = samples

    @property
    def n_snps(self) -> int:
        """Return the number of snps.

        Returns:
            int: Number of snps.
        """
        return self.genotypes.shape[self.SNP_AXIS]

    @property
    def n_samples(self) -> int:
        """Return the number of samples.

        Returns:
            int: The number of samples.
        """
        return self.genotypes.shape[self.SAMPLES_AXIS]

    @property
    def rsids(self) -> pd.Index:
        """Return an index of rsids ordered by chromosome and base position.

        Returns:
            pd.Index: Ordered list of all rsids.
        """
        return self.genotypes.index

    @property
    def iids(self) -> pd.Index:
        """Return an index of IIDs. Order is arbitrary.

        Returns:
            pd.Index: List of all iids.
        """
        return self.genotypes.columns

    @property
    def shape(self) -> tuple[int, int]:
        """Return the shape of the genotype data.

        Returns:
            tuple[int, int]: Shape of the genotype data (n_snps, n_samples).
        """
        return self.genotypes.shape

    def _filter_base_snps(self, *rsid_list: str):
        """A base function to filter genotype data by a list of SNPs (indexes).

        Returns:
            Type[self]: A new genetic data object of the same type.
        """
        genotypes = self.genotypes.filter(rsid_list, axis=self.SNP_AXIS) # type: ignore
        snps = self.snps.loc[rsid_list, :]
        return type(self)(genotypes, snps, self.samples)

    def _filter_base_samples(self, *iid_list: str):
        """A base function to filter genotype data by a list of IIDs (indexes).

        Returns:
            Type[self]: A new genetic data object of the same type.
        """
        genotypes = self.genotypes.filter(iid_list, axis=self.SAMPLES_AXIS) # type: ignore
        samples = self.samples.loc[iid_list, :]
        return type(self)(genotypes, self.snps, samples)

    def _filter_rsid(self, *rsid_list: str):
        """Filter snps based on rsid. Keep the listed SNPs.

        Args:
            *rsid_list (str): rsids to keep.

        Returns:
            self: Object of same type as parent filtered by rsid.
        """
        rsid_intersection = set(self.rsids) & set(rsid_list)
        return self._filter_base_snps(*rsid_intersection)

    def _filter_iid(self, *iid_list: str):
        """Filter samples based on iid. Keep the listed IIDs.

        Args:
            *iid_list (str): IIDs to keep.

        Returns:
            self: Object of same type as parent filtered by iid.
        """
        iid_intersection = set(self.iids) & set(iid_list)
        return self._filter_base_samples(*iid_intersection)

    def _filter_chr(self, *chr_list: int):
        """Filter genetic data by chromosome.

        Returns:
            self: Object of same type as parent filtered by chromsome.
        """
        rsids = self.snps.loc[self.snps[CHR].isin(chr_list), :].index.tolist()
        return self._filter_rsid(*rsids)

    def _filter_bpos(
            self, bp_low: Union[int, float], bp_high: Union[int, float]
        ):
        """Filter genetic data by base position.

        Args:
            bp_low (int | float): Bottom of base position interval.
            bp_high (int | float): Top of base position interval.

        Returns:
            self: Object of same type as parent filtered by base position.
        """
        rsids = self.snps.loc[
            self.snps[BPOS].between(bp_low, bp_high), :].index.tolist()
        return self._filter_rsid(*rsids)

    @property
    def mind(self) -> pd.Series:
        """Calculates genotype missingness for each sample.

        Method name is taken from plink.

        Returns:
            pd.Series: A series indexed by IID showing the proportion of
                genotypes that are missing for each sample.
        """
        missing_by_sample = self.genotypes.isna().sum(axis=self.SNP_AXIS)
        mind = missing_by_sample / self.n_snps
        return mind

    def _filter_mind(self, mind_threshold: float):
        """Filter based on mind.

        Args:
            mind_threshold (float): Samples with mind greater than this level
                will be removed.

        Returns:
            Type[self]: Genotype data object filtered by MAF.
        """
        # Select valid SNPs
        condition = self.mind <= mind_threshold
        samples = self.mind[condition].index.tolist()

        # Create objects
        return self._filter_base_samples(*samples)

    @property
    def max_mind(self) -> float:
        """Find the maximum mind.

        Returns:
            float: The value of the maximum mind.
        """
        return self.mind.max()

    @property
    def geno(self) -> pd.Series:
        """Calculates genotype missingness rate for each SNP.

        Method name is taken from plink.

        Returns:
            pd.Series: Genotype missingness rate for each SNP, indexed by rsID.
        """
        missing_by_snp = self.genotypes.isna().sum(axis=self.SAMPLES_AXIS)
        geno = missing_by_snp / self.n_samples
        return geno

    def _filter_geno(self, geno_threshold: float):
        """Filters based on genotype missingness rate.

        Args:
            geno_threshold (float): SNPs with geno greater than this level will
                be removed.

        Returns:
            Type[self]: Genetic data object filtered by geno missingness rate.
        """
        # Select valid SNPs
        condition = self.geno <= geno_threshold
        snps = self.geno[condition].index.tolist()

        # Create objects
        return self._filter_base_snps(*snps)

    @property
    def max_geno(self) -> float:
        """Find the maximum geno.

        Returns:
            float: The value of the maximum geno.
        """
        return self.geno.max()

    def filter(self, **kwargs: Any):
        """Filters genetic data by an arbitrary number of factors.

        This function takes an arbitrary number of keyword arguments. For each
        argument that matches one of a list of defined filtering operations,
        that operation will be performed according to the value of the argument.

        Any keyword arguments that do not match any members of the list will be
        ignored and a warning will be given.

        Returns:
            GenoData: A GenoData object filtered by all applicable filters.
        """
        geno_data = self
        for arg, val in kwargs.items():
            if not isinstance(val, Iterable):
                val = [val]
            try:
                func = getattr(geno_data, f"_filter_{arg}")
            except AttributeError:
                mssg = (f"GenoData does not implement filter for {arg}. "
                        f"Skipping filtering step for {arg}.")
                warnings.warn(mssg, UserWarning)
            else:
                geno_data = func(*val)
        return geno_data

    def _sample_rsid(
            self,
            param: Union[int, float],
            bit_generator: BitGenerator
        ):
        """Filter snps based on rsid. Keep a random subset of SNPs.

        Args:
            param (Union[int, float]): Number (int) or fraction (float) of SNPs
                to keep.
            bit_generator (BitGenerator): Bit generator to use for sampling.

        Returns:
            self: Object of same type as parent sampled by rsid.
        """
        if isinstance(param, int):
            rsids = self.snps.sample(n=param, random_state=bit_generator)
        elif isinstance(param, float):
            rsids = self.snps.sample(frac=param, random_state=bit_generator)
        else:
            raise TypeError(f"{type(param)} is not a valid type for param.")
        rsid_list = rsids.index.tolist()
        return self._filter_base_snps(*rsid_list)

    def _sample_iid(
            self,
            param: Union[int, float],
            bit_generator: BitGenerator
        ):
        """Filter samples based on iid. Keep a random subset of IIDs.

        Args:
            param (Union[int, float]): Number (int) or fraction (float) of IIDs
                to keep.
            bit_generator (BitGenerator): Bit generator to use for sampling.

        Returns:
            self: Object of same type as parent filtered by iid.
        """
        if isinstance(param, int):
            iids = self.samples.sample(n=param, random_state=bit_generator)
        elif isinstance(param, float):
            iids = self.samples.sample(frac=param, random_state=bit_generator)
        else:
            raise TypeError(f"{type(param)} is not a valid type for param.")
        iid_list = iids.index.tolist()
        return self._filter_base_samples(*iid_list)

    def sample(
            self,
            rsid: Optional[Union[int, float]] = None,
            iid: Optional[Union[int, float]] = None,
            seed: Optional[int] = None
        ):
        """Sample randomly from the genetic data.

        Args:
            rsid (Union[int, float]): Number (int) or percentage (float) of SNPs
                to sample. If None, no sampling will occur.
            iid (Union[int, float]): Number (int) or percentage (float) of
                samples to sample. If None, no sampling will occur.
            seed (int): Seed for random number generator. If None, a random
                seed will be generated.

        Returns:
            GenoData: A GenoData object with a random subset of SNPs and
                samples.
        """
        rng = np.random.default_rng(seed).bit_generator
        geno_data = self
        if rsid is not None:
            geno_data = geno_data._sample_rsid(rsid, rng)
        if iid is not None:
            geno_data = geno_data._sample_iid(iid, rng)
        return geno_data

    def flip_snps(self, *rsids: str):
        """Placeholder for flip snp methods in subclasses.

        Raises:
            NotImplementedError: This method is not implemented for an abstract
                class.
        """
        raise NotImplementedError

    def _split_rsid(
            self, blocks_dict: Union[pd.Series, dict[str, int]]
        ) -> dict[int, list[str]]:
        """Split genetic data into blocks according to a list of rsIDs.

        Args:
            blocks_dict (Union[pd.Series, dict[str, int]]): Mapping from rsID
                to block number. If a Series, the index must be rsIDs and the
                values must be block numbers. If a dict, the keys must be rsIDs
                and the values must be block numbers.

        Raises:
            ValueError: If each rsID is not mapped to a block.

        Returns:
            dict[int, Type[AbstractGenoData]]: Mapping from block number to
                GenoData object.
        """
        # Map from rsid to block
        ref_snps = self.snps.copy()
        ref_snps[BLOCK] = ref_snps.index.map(blocks_dict)
        # Check for non-mapped snps
        not_mapped = ref_snps[BLOCK].isna()
        if not_mapped.any():
            snps = ref_snps[not_mapped].index.tolist()
            if len(snps) < ERRLIST_MAXLEN:
                mssg = f"SNPs {snps} were not mapped to a block."
            else:
                mssg = f"{len(snps)} SNPs were not mapped to a block."
            raise ValueError(mssg)
        # Create separate genetic data object for each block
        block_dict = {}
        for block, block_variants in ref_snps.groupby(BLOCK):
            snps_list = block_variants.index.tolist()
            block_dict[block] = self.filter(rsid=snps_list)
        return block_dict


class IntGenoData(AbstractGenoData):
    """A class to hold and perform basic operations on integer genotype data.
    """
    @property
    def af(self) -> pd.Series:
        """Calculate the allele frequency of each SNP.

        Returns:
            pd.Series: Series of allele frequencies indexed by rsID.
        """
        # Calculate allele frequency
        allele_count = self.genotypes.sum(axis=self.SAMPLES_AXIS, skipna=True)
        # Note: pandas count ignores NA values by default
        slot_count = 2 * self.genotypes.count(axis=self.SAMPLES_AXIS)
        af = allele_count / slot_count
        return af

    @staticmethod
    def to_maf(af: float) -> float:
        """Convert a biallelic allele frequency to a minor allele frequency.

        Args:
            af (float): Allele frequency.

        Raises:
            ValueError: If af is not in [0, 1].

        Returns:
            float: Minor allele frequency.
        """
        if 0 <= af <= 0.5:
            maf = af
        elif af <= 1:
            maf = 1 - af
        else:
            raise ValueError("Allele frequency not in [0, 1].")
        return maf

    @property
    def maf(self) -> pd.Series:
        """Calculate the minor allele frequency for each SNP.

        Returns:
            pd.Series: Series of minor allele frequencies indexed by rsID.
        """
        return self.af.map(self.to_maf)

    def _filter_maf(self, maf_threshold: float):
        """Filter SNPs by a minor allele frequency threshold.

        Args:
            maf_threshold (float): SNPs with MAF below this value should be
                removed.

        Returns:
            Type[IntGenoData]: New genodata object filtered by MAF.
        """
        # Select valid SNPs
        condition = self.maf > maf_threshold
        snps = self.maf[condition].index.tolist()

        # Create objects
        return self._filter_base_snps(*snps)

    @property
    def min_maf(self) -> float:
        """Find the minimim minor allele frequency.

        Returns:
            float: The minimum minor allele frequency.
        """
        return self.maf.min()

    @property
    def n_hom1(self) -> pd.Series:
        """Number of samples homozygous in the reference allele.

        Returns:
            pd.Series: Count of homozygous (A1) samples by rsID.
        """
        return (self.genotypes == 0).sum(skipna=True, axis=self.SAMPLES_AXIS)

    @property
    def n_hom2(self) -> pd.Series:
        """Number of samples homozygous in the alternate allele.

        Returns:
            pd.Series: Count of homozygous (A2) samples by rsID.
        """
        return (self.genotypes == 2).sum(skipna=True, axis=self.SAMPLES_AXIS)

    @property
    def n_het(self) -> pd.Series:
        """Number of samples heterozygous.

        Returns:
            pd.Series: Count of heterozygous samples by rsID.
        """
        return (self.genotypes == 1).sum(skipna=True, axis=self.SAMPLES_AXIS)

    @staticmethod
    @vectorize([
        float64(int32, int32, int32, boolean),
        float64(int64, int64, int64, boolean)])
    def _hwe_test(het: int, rare: int, n: int, midp: bool) -> float:
        """Calculate the HWE p-value for a single SNP.

        The method for calculation of the HWE p-value is taken from Wigginton,
        JE, Cutler, DJ, and Abecasis, GR (2005) A Note on Exact Tests of
        Hardy-Weinberg Equilibrium. American Journal of Human Genetics. 76: 000
        - 000.

        The midp correction is from Graffelman and Moreno (2013) The min p-value
        in exact tests for Hardy-Weinberg equilibrium. Statistical Applications
        in Genetics and Molecular Biology. 12(4): 433 - 448.

        This function is vectorised, so can take numpy arrays as inputs.

        Args:
            het (int): Number of heterozygous samples observed.
            rare (int): Number of rare allele observed.
            n (int): Total number of alleles observed.
            midp (bool): Apply midp adjustment.

        Returns:
            float: HWE p-value.
        """
        p_array = np.zeros(1 + rare)

        midpoint = floor(rare * ((2 * n) - rare) / (2 * n))
        if midpoint % 2 != rare % 2:
            midpoint += 1

        p_array[midpoint] = 1

        # Calculate probabilities from midpoint down
        curr_hets = midpoint
        curr_homr = (rare - midpoint) / 2
        curr_homc = n - curr_hets - curr_homr

        while curr_hets >= 2:
            p_array[curr_hets - 2] = (
                (p_array[curr_hets] * curr_hets * (curr_hets - 1))
                / (4 * (curr_homr + 1) * (curr_homc + 1))
            )

            # 2 fewer heterozygotes -> add 1 rare homozygote, 1 cmmn homozygote
            curr_hets -= 2
            curr_homr += 1
            curr_homc += 1

        # Calculate probabilities from midpoint up
        curr_hets = midpoint
        curr_homr = (rare - midpoint) / 2
        curr_homc = n - curr_hets - curr_homr

        while curr_hets <= rare - 2:
            p_array[curr_hets + 2] = (
                (p_array[curr_hets] * 4 * curr_homr * curr_homc)
                / ((curr_hets + 2) * (curr_hets + 1))
            )

            curr_hets += 2
            curr_homr -= 1
            curr_homc -= 1

        # P-value calculation
        target = p_array[het]

        halfsum = np.sum(p_array[p_array <= target])
        fullsum = halfsum + np.sum(p_array[p_array > target])

        if midp:
            return np.minimum(1.0, (halfsum - (target/2)) / fullsum)
        return np.minimum(1.0, halfsum / fullsum)

    def hwe(self, midp: bool) -> pd.Series:
        """Calculate the exact HWE p-values for each SNP.

        Returns:
            pd.Series: HWE p-values indexed by rsID.
        """
        hom1, hom2, het = (
            x.to_numpy() for x in [self.n_hom1, self.n_hom2, self.n_het])
        n_genotypes = hom1 + hom2 + het

        # Get number of rare homoozygotes and alleles
        n_homr = np.minimum(hom1, hom2)
        n_rare = het + (2 * n_homr)

        p_val = self._hwe_test(het, n_rare, n_genotypes, midp) # type: ignore
        return pd.Series(p_val, index=self.rsids)

    def _filter_hwe(self, hwe_threshold: float):
        """Remove variants with HWE p-values below a certain threshold.

        Args:
            hwe_threshold (float): The p-value threshold.

        Returns:
            GenoData: GenoData objected filtered by HWE p-value.
        """
        hwe = self.hwe(False)
        rsids = hwe[hwe > hwe_threshold].index.tolist()
        return self._filter_base_snps(*rsids)

    def _filter_hwe_midp(self, hwe_threshold: float):
        """Reomve variants with HWE mid p-values below a certain threshold.

        Args:
            hwe_threshold (float): The mid p-value thershold.

        Returns:
            GenoData: GenoData object filtered by HWE mid-p value.
        """
        hwe = self.hwe(True)
        rsids = hwe[hwe > hwe_threshold].index.tolist()
        return self._filter_base_snps(*rsids)

    def flip_snps(self, *rsids: str):
        """Flip A1 and A2 for the selected SNPs.

        Returns:
            Type[IntGenoData]: GenoData object.
        """
        # Flip genotypes
        genotypes = self.genotypes.copy()
        genotypes.loc[rsids, :] = 2 - genotypes.loc[rsids, :]

        # Flip info files
        snps = self.snps.copy()
        snps.loc[rsids, [A1, A2]] = (
            snps.loc[rsids, [A2, A1]].to_numpy().reshape(-1, 2))

        return type(self)(genotypes, snps, self.samples)

    def standardised(self):
        """Standardise genotypes.

        Returns:
            StdGenoData: A standardised genotype data object.
        """
        mean_geno = self.genotypes.mean(axis=self.SAMPLES_AXIS, skipna=True)
        stderr_geno = self.genotypes.std(axis=self.SAMPLES_AXIS, skipna=True)
        if (stderr_geno == 0).any():
            mssg = (
                "Division by zero. Apply HWE and/or MAF filters. "
                f"Bad SNPs: {stderr_geno[stderr_geno == 0].index.tolist()}"
            )
            raise FloatingPointError(mssg)

        demean_geno = self.genotypes.sub(mean_geno, axis=self.SNP_AXIS)
        std_geno = demean_geno.div(stderr_geno, axis=self.SNP_AXIS)

        return StdGenoData(std_geno, self.snps, self.samples)

    standardized = standardised

    def save(self, out: str):
        """
        Save the integer genotype data to a .bed/.bim/.fam fileset.
        """
        # Make iids
        iids = self.samples.index.tolist()
        fiids = [[iid, iid] for iid in iids]

        # Make snps
        snps = self.snps.index.tolist()

        # Get positions
        pos = np.empty((len(snps), 3), dtype="object")
        pos[:, 1] = np.nan
        pos[:, [0, 2]] = self.snps.loc[:, [CHR, BPOS]].to_numpy().reshape(-1, 2)

        # Get Values
        val = self.genotypes.to_numpy().T

        snpdata = SnpData(fiids, snps, val, pos)
        Bed.write(out, snpdata, _require_float32_64=False, num_threads=1)

        # Write BIM file
        self.snps.reset_index(drop=False)\
            .loc[:, list(BIM_COLS.values())]\
            .to_csv(f"{out}.bim", sep="\t", index=False, header=False)

        # Write FAM file
        self.samples.reset_index(drop=False)\
            .loc[:, list(FAM_COLS.values())]\
            .to_csv(f"{out}.fam", sep="\t", index=False, header=False)

class StdGenoData(AbstractGenoData):
    """A class to hold and perform operations on standardised genotype data.
    """
    def flip_snps(self, *rsids: str):
        """Flip A1 and A2 for the selected SNPs.

        Returns:
            Type[StdGenoData]: StdGenoData object.
        """
        rsids = list(rsids) # type: ignore
        # Flip genotypes
        genotypes = self.genotypes
        trans_constant = (
            genotypes.max(axis=self.SAMPLES_AXIS)
            + genotypes.min(axis=self.SAMPLES_AXIS))
        genotypes.loc[rsids, :] = (
            genotypes.loc[rsids, :].rsub(
                trans_constant[rsids], axis=self.SNP_AXIS))

        # Flip info files
        snps = self.snps.copy()
        snps.loc[rsids, [A1, A2]] = (
            snps.loc[rsids, [A2, A1]].to_numpy().reshape(-1, 2))

        return type(self)(genotypes, snps, self.samples)

    def _calculate_ldm_window(
            self,
            chrom: int,
            window: int,
            sparse: bool = True,
            tol: float = 0.001
        ) -> dict[str, Union[np.ndarray, sp.spmatrix, pd.DataFrame]]:
        """Calculate a sparse windowed LD matrix from genotype data.

        Args:
            chrom (int): The chromosome for which to calculate the sparse LD
                matrix.
            window (int): The width of the window in bases.
            sparse (bool): Whether to return a sparse matrix.
            tol (float): The tolerance below which elements are set to zero.

        Returns:
            dict[str, Union[np.ndarray, sp.spmatrix, pd.DataFrame]]:
                A dictionary containing the LD matrix and variant information
                for each block.
        """
        geno_data = self.filter(chr=chrom)
        bpos = geno_data.snps[BPOS].to_numpy().reshape(-1)
        geno_array = np.ascontiguousarray(geno_data.genotypes.to_numpy())
        missing = geno_data.geno.to_numpy().reshape(-1) != 0

        if sparse:
            ldm_i, ldm_j, ldm_val, rows_remove = make_spwindow_cov(
                geno_array, bpos, window, missing, tol)
            rows_keep = list(set(range(geno_data.n_snps)) - set(rows_remove))
            coo_mat = sp.coo_matrix((ldm_val, (ldm_i, ldm_j)))
            coo_mat.eliminate_zeros()
            mat = coo_mat.tocsr()[np.ix_(rows_keep, rows_keep)]
            varinfo = geno_data.snps.iloc[rows_keep].reset_index(drop=False)
        else:
            mat = make_window_cov(geno_array, bpos, window, missing)
            varinfo = geno_data.snps.reset_index(drop=False)
        print(f"Chromosome {chrom} finished.")
        return {LDM: mat, VARINFO: varinfo} # type: ignore

    def calculate_ldm_window(
            self,
            window: Optional[int] = None,
            n_cores: Optional[int] = None,
            sparse: bool = True,
            tol: float = 0.001
        ) -> dict[int, sp.csr_matrix]:
        """Calculate a sparse windowed LD matrix in CSR format.

        Note: These calculations are quite fast, so parallelisation may not
        always be necessary.

        Args:
            window (int): Set LD correlations to zero for all values separated
                by a distance of greater than window.
            n_cores (Optional[int]): Number of cores to use for parallelisation.
                If None, defaults to the number of cores available.
            sparse (bool): Whether to make sparse matrices.
            tol (float): Tolerance for sparse matrix construction.

        Returns:
            dict[int, sp.csr_matrix]: Dictionary containing a sparse LD matrix
                for each chromosome.
        """
        chromosomes = self.snps[CHR].unique().tolist()

        args_list = [(chrom, window, sparse, tol) for chrom in chromosomes]

        if n_cores is not None: # use as many cores as required up to n_cores
            n_cores = min(n_cores, len(args_list))

        with Pool(n_cores) as p:
            res = p.starmap(self._calculate_ldm_window, args_list)
        spmatrix_dict = dict(zip(chromosomes, res))

        return spmatrix_dict # type: ignore

    def _calculate_ldm(self) -> dict[str, Union[np.ndarray, pd.DataFrame]]:
        """Calculate a full LD matrix.

        Remove SNPs that are perfectly correlated with another SNP as these
        give no extra information and will cause the LD matrix to be singular.

        Returns:
            dict[str, Union[np.ndarray, pd.DataFrame]]: A dictionary containing
                the LD matrix and variant information.
        """
        geno_array = np.ascontiguousarray(self.genotypes.to_numpy())
        mat, _, rows_remove = make_cov(geno_array)
        rows_keep = list(set(range(self.n_snps)) - rows_remove)
        mat = mat[np.ix_(rows_keep, rows_keep)]
        varinfo = self.snps.iloc[rows_keep].reset_index(drop=False)
        return {LDM: mat, VARINFO: varinfo}

    @staticmethod
    def _calculate_ldm_static(
            block: Type["StdGenoData"]
        ) -> dict[str, Union[np.ndarray, pd.DataFrame]]:
        """
        Static method to calculate a block of the LD matrix. Used to parallelise
        LD matrix calculations.

        Args:
            block (Type[StdGenoData]): A block of genetic data.

        Returns:
            dict[str, Union[np.ndarray, pd.DataFrame]]: A dictionary containing
                the LD matrix and variant information.
        """
        return block._calculate_ldm() # type: ignore

    def calculate_ldm_blocks(
            self,
            block_map: Union[pd.Series, dict[str, int]],
            n_cores: Optional[int] = None,
        ) -> dict[int, dict[str, Union[np.ndarray, pd.DataFrame]]]:
        """Calculate a block-wise LD matrix, where the each block ends at one
        of the listed terminal rsids.

        Args:
            block_map (list[str]): Mapping of rsids to block numbers.
            n_cores (Optional[int]): Number of cores to use for parallelisation.
                If None, defaults to the number of cores available.

        Returns:
            dict[int, dict[str, Union[np.ndarray, pd.DataFrame]]]:
                A dictionary containing an LD matrix dictionary for each block.
        """
        blocks = self._split_rsid(block_map)
        blocks_list = list(blocks.keys())
        args_list = list(blocks.values())

        if n_cores is not None: # use as many cores as required up to n_cores
            n_cores = min(n_cores, len(args_list))

        print("Starting subprocesses.")

        with Pool(n_cores) as p:
            res = p.map(self._calculate_ldm_static, args_list) # type: ignore
        ldm_dict = dict(zip(blocks_list, res))
        return ldm_dict


    def _calculate_grm(
            self,
            individuals: Optional[list[str]] = None,
            weights: Optional[pd.Series] = None
        ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate a full GRM.

        Args:
            individuals (list[str]): List of individual IDs to include in the
                GRM.
            weights (dict[str, float]): Weights to apply to the GRM.

        Returns:
            np.ndarray: The GRM.
        """
        if individuals is not None:
            gendata = self.filter(iid=individuals)
        else: # use all individuals
            gendata = self
        if weights is None:
            weights = np.ones(gendata.n_snps) # type: ignore
        else:
            weights = weights.loc[
                gendata.snps.index
            ].values.reshape(-1) # type: ignore
        geno_array = np.ascontiguousarray(gendata.genotypes.to_numpy().T)
        grm, non_missing, _ = make_weighted_cov(geno_array, weights=weights) # type: ignore
        return grm, non_missing

    def calculate_grm(
            self,
            individuals: Optional[list[str]] = None,
            weights: Optional[dict[str, float]] = None
        ) -> GRM:
        """Calculate a full GRM.

        Args:
            individuals (list[str]): List of individual IDs to include in the
                GRM.
            weights (dict[str, float]): Weights to apply to the GRM.

        Returns:
            GRM: The GRM.
        """
        grm, non_missing = self._calculate_grm(individuals, weights) # type: ignore
        ids = self.samples.reset_index(drop=False).iloc[:, [0, 1]]
        return GRM(grm, ids, non_missing)


def merge(*genotype_data: Type[AbstractGenoData]) -> Type[AbstractGenoData]:
    """Merge sets of genotype data for different chromosomes/loci from the same
    set of samples.

    Raises:
        TypeError: If genetic data inputs are not all of the same object type.
        ValueError: The set of SNPs overlaps across genetic data inputs.
        ValueError: The set of samples is not the same across all genetic data
            inputs.

    Returns:
        Type[AbstractGenoData]: Merged genetic data.
    """
    # Checks
    distinct_types = set(type(genodata_obj) for genodata_obj in genotype_data)
    if len(distinct_types) != 1:
        mssg = (
            "Genetic data are not all of same type and so cannot be merged. "
            f"Types present: {distinct_types}."
        )
        raise TypeError(mssg)
    genodata_type = type(genotype_data[0])

    # Merge snps
    geno_snps = [genodata_obj.snps for genodata_obj in genotype_data] # type: ignore
    try:
        snps = pd.concat(geno_snps, verify_integrity=True)
    except ValueError as err:
        mssg = "The set of SNPs for each geno data object should not overlap."
        raise ValueError(mssg) from err

    # Merge samples
    geno_iids = set(
        frozenset(genodata_obj.iids) for genodata_obj in genotype_data) # type: ignore
    if len(geno_iids) != 1:
        iid_lens = [len(iid_i) for iid_i in geno_iids]
        mssg = (
            "All geno data objects should have the same set of samples. "
            f"Distinct sample counts: {iid_lens}."
        )
        raise ValueError(mssg)
    samples = genotype_data[0].samples # type: ignore

    # Merge geno
    geno_geno = [genodata_obj.genotypes for genodata_obj in genotype_data] # type: ignore
    genotypes = pd.concat(geno_geno)

    return genodata_type(genotypes, snps, samples) # type: ignore


def elementwise_isin(
        array: np.ndarray,
        match_list: Optional[list] = None
    ) -> tuple[np.ndarray, np.ndarray]:
    """Apply an isin operation elementwise for an array with respect to a list.

    Args:
        array (np.ndarray): An array representing a vector.
        match_list (Optional[list], optional): A list of array elements to
            match with. Defaults to None.

    Returns:
        np.ndarray: A subarray of array containing the intersection of array and
            match list.
        np.ndarray: A 1D array of the indices for all elements which match an
            element of match list.
    """
    if match_list is not None:
        condition = np.isin(array, match_list)
        vals = array[condition]
        indices = np.argwhere(condition).reshape(-1)
    else:
        vals = array
        indices = np.arange(len(array))
    return vals, indices


def _read_bed(
        path: str,
        rsids: Optional[list] = None,
        individuals: Optional[list] = None,
        num_threads: Optional[int] = 1
    ) -> IntGenoData:
    """Read raw genotypes into an annotated data frame.

    Args:
        path (str): Path (without extension) to bed/bim/fam fileset.
        rsids (Optional[list], optional): Filter SNPs to this set of rsIDs.
            Defaults to None, in which case no filtering will occur.
        individuals (Optional[list], optional): Filters samples to this list of
            individuals. Defaults to None, in which case no filtering will
            occur.
        num_threads (Optional[int], optional): Specifies the number of threads
            to use when reading bed files.

    Returns:
        IntGenoData: Integer Genotype data object.
    """
    bed_file = Bed(path, count_A1=False)

    snps = bed_file.sid
    samples = bed_file.iid[:, PYSNPTOOLS_IID_COL]

    snp_vals, snp_indices = elementwise_isin(snps, rsids)
    sample_vals, sample_indices = elementwise_isin(samples, individuals)

    # Read genotype data
    geno_data = bed_file[sample_indices, snp_indices].read(
        dtype="int8", num_threads=num_threads, _require_float32_64=False).val.T # type: ignore
    geno_data_df = pd.DataFrame(geno_data, index=snp_vals, columns=sample_vals)
    geno_data_df[geno_data_df == PYSNPTOOLS_MISSING_VAL] = pd.NA

    # Read variant information
    snp_data = pd.read_csv(
        f"{path}.bim", delim_whitespace=True,
        usecols=BIM_COLS.keys(), header=None # type: ignore
    )
    snp_data = snp_data.rename(
        columns=BIM_COLS).set_index(RSID).loc[snp_vals]

    # Read sample infomration
    sample_data = pd.read_csv(
        f"{path}.fam", delim_whitespace=True,
        usecols=FAM_COLS.keys(), header=None, # type: ignore
        dtype={ix: str for ix, col in FAM_COLS.items() if ID in col}
    )
    sample_data = sample_data.rename(
        columns=FAM_COLS).set_index(IID).loc[sample_vals]
    return IntGenoData(geno_data_df, snp_data, sample_data)


def read_bed(
        paths: Union[str, list[str]],
        rsids: Optional[list] = None,
        individuals: Optional[list] = None,
        num_threads: Optional[int] = 1
    ) -> IntGenoData:
    """Read raw genotypes into an annotated data frame.

    Can take a direct path to a .bed/.bim/.fam file or a list of paths to
    a set of .bed/.bim/.fam files.

    Args:
        paths (Union[str, list[str]]): Paths to files containing paths to
            .bed/.bim/.fam filesets to load together.
        rsids (Optional[list], optional): Filter SNPs to this set of rsIDs.
            Defaults to None, in which case no filtering will occur.
        individuals (Optional[list], optional): Filters samples to this list of
            individuals. Defaults to None, in which case no filtering will
            occur.
        num_threads (Optional[int], optional): Specifies the number of threads
            to use when reading bed files.

    Returns:
        Type[AbstractGenoData]: Annotated genotype data object.
    """
    if isinstance(paths, str):
        paths = [paths]
    geno_data_list = [
        _read_bed(path, rsids, individuals, num_threads) for path in paths
    ]
    if len(geno_data_list) > 1:
        geno_data = merge(*geno_data_list) # type: ignore
    else:
        geno_data = geno_data_list[0]
    return geno_data # type: ignore
