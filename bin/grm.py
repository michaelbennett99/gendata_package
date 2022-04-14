#/usr/bin/env python3

"""
A command line script to make a GRM in python.
"""

import argparse

from ssgac_gendata.core import read_bed
from ssgac_gendata.utils import read_1col_text

def process_args() -> argparse.Namespace:
    """
    Process command line arguments.
    """
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
        "--out", type=str, required=True, help="Path to which to write GRM."
    )
    return parser.parse_args()

def main(args: argparse.Namespace):
    gendata = read_bed(args.bfile, rsids=args.extract)
    std_gendata = gendata.standardised()
    grm = std_gendata.calculate_grm()
    grm.write_gz(args.out)

if __name__ == "__main__":
    args = process_args()
    main(args)
