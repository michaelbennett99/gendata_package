#!/usr/bin/env python3

def read_1col_text(path: str) -> list[str]:
    """
    Read lines from a one column text file.

    :param path: Path to file containing one column of entries.
    :type path: str

    :return: List of variants.
    :rtype: list[str]
    """
    with open(path, "r") as f:
        variants = [line.strip() for line in f]
    return variants
