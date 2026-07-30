#!/usr/bin/env python3
"""
Compute the Median Normalized Squared Error (MdNSE) metric for the
J0030 phase-channel model file and append it as a 7th column.

Input file columns (whitespace-separated):
    1. channel number
    2. rotational phase bin
    3. observed counts (y)
    4. best-fit model counts (y_hat)
    5. NS-surface-only model component
    6. phase-independent background

Output: same columns plus
    7. per-row contribution to MdNSE = (y_i - y_hat_i)^2 / median(y_hat)^2

The sum of column 7 equals the overall MdNSE statistic:
    MdNSE = sum_i (y_i - y_hat_i)^2 / median(y_hat)^2
"""

import argparse
from pathlib import Path

import numpy as np


def compute_mdnse(infile: Path, outfile: Path, group_by_channel: bool = False) -> None:
    # Load the data; assumes whitespace-delimited numeric columns.
    data = np.loadtxt(infile)
    if data.ndim != 2 or data.shape[1] < 6:
        raise ValueError(
            f"Expected at least 6 columns in {infile}, got shape {data.shape}"
        )

    channel = data[:, 0]
    observed = data[:, 2]   # y
    model    = data[:, 3]   # y_hat

    if group_by_channel:
        # Compute median(y_hat) per channel, then per-row contribution.
        contrib = np.zeros_like(observed)
        for ch in np.unique(channel):
            mask = channel == ch
            med = np.median(model[mask])
            if med == 0:
                raise ZeroDivisionError(
                    f"median(y_hat) is zero for channel {ch}; cannot normalize."
                )
            contrib[mask] = (observed[mask] - model[mask]) ** 2 / med ** 2
        total = contrib.sum()
        print(f"Per-channel MdNSE computed. Global sum of contributions = {total:.6g}")
    else:
        med = np.median(model)
        if med == 0:
            raise ZeroDivisionError(
                "median(y_hat) is zero; cannot normalize MdNSE."
            )
        contrib = (observed - model) ** 2 / med ** 2
        total = contrib.sum()
        print(f"median(y_hat) = {med:.6g}")
        print(f"Total MdNSE   = {total:.6g}")

    # Stack new column and write out.
    out = np.column_stack([data, contrib])

    # Choose a sensible format: integers for cols 1-2, floats elsewhere.
    # Use generic %.8g to be safe across columns.
    header = (
        "channel  phase_bin  observed  model  ns_surface  background  mdnse_contrib"
    )
    np.savetxt(outfile, out, fmt="%.8g", header=header)
    print(f"Wrote {outfile} ({out.shape[0]} rows, {out.shape[1]} columns).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i", "--input",
        type=Path,
        default=Path("data/j0030_phase_channel_model.txt"),
        help="Input data file (default: data/j0030_phase_channel_model.txt)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output file (default: <input stem>_mdnse.txt next to input)",
    )
    parser.add_argument(
        "--by-channel",
        action="store_true",
        help="Compute median(y_hat) per channel instead of globally.",
    )
    args = parser.parse_args()

    outfile = args.output or args.input.with_name(
        args.input.stem + "_mdnse.txt"
    )
    compute_mdnse(args.input, outfile, group_by_channel=args.by_channel)


if __name__ == "__main__":
    main()