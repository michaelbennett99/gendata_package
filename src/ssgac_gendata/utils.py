#!/usr/bin/env python3

def read_1col_text(path: str) -> list[str]:
    """
    Read lines from a one column text file.

    Args:
        path (str): Path to file containing one column of entries.

    Returns:
        list[str]: List of variants.
    """
    with open(path, "r") as f:
        variants = [line.strip() for line in f]
    return variants
