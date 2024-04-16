#/usr/bin/env python3
"""
A command line script to make a GRM in python.
"""

import argparse

from pandas import read_csv, Series

from gendata.core import read_bed
from gendata.utils import read_1col_text

def process_args() -> argparse.Namespace:
    """
    Process command line arguments.
    """
    def read_weights(path: str) -> Series:
        """
        Read weights from file.

        :param path: Path to file containing weights.
        :type path: str

        :return: Series of weights.
        :rtype: Series
        """
        return read_csv(
            path, sep="\t", index_col=0, header=None).squeeze().abs()

    parser = argparse.ArgumentParser(
        description="Make a GRM from a BED file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--bfile", type=str, required=True,
        help="Path to plink fileset."
    )
    parser.add_argument(
        "--extract", type=read_1col_text, default=None,
        help="Path to text file containing list of SNPs to extract."
    )
    parser.add_argument(
        "--weights", type=read_weights, default=None,
        help="Path to file containing weights for each SNP."
    )
    parser.add_argument(
        "--out", type=str, required=True,
        help="Path to which to write GRM."
    )
    return parser.parse_args()

def main(args: argparse.Namespace):
    gendata = read_bed(args.bfile, rsids=args.extract)
    std_gendata = gendata.standardised()
    grm = std_gendata.calculate_grm(weights=args.weights)
    grm.write_gz(args.out)

if __name__ == "__main__":
    args = process_args()
    main(args)
