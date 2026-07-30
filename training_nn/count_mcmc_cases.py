"""
count_mcmc_cases.py

Reads all mcmc_vac_*.dat files in a directory and reports:
  - Number of files found
  - Rows per file
  - Total rows (= total cases) across all files
"""

from pathlib import Path
import argparse


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_DIR = Path(
    "/Users/jraynau1/Workspace/CLionProjects/ray_tracing_training_nn"
    "/output_data/j0740/mcmc_samples"
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def count_cases(samples_dir: Path, pattern: str = "mcmc_vac_*.dat",
                verbose: bool = True) -> dict:
    """
    Count total rows across all matching files in samples_dir.

    Returns a dict with:
        'files'        : list of Path objects found
        'rows_per_file': dict mapping filename -> row count
        'total_rows'   : int, sum of all rows
        'total_files'  : int
    """
    files = sorted(samples_dir.glob(pattern))

    if not files:
        print(f"No files matching '{pattern}' found in {samples_dir}")
        return {"files": [], "rows_per_file": {}, "total_rows": 0, "total_files": 0}

    rows_per_file = {}
    total_rows    = 0

    for f in files:
        with open(f, "r") as fh:
            n_rows = sum(1 for line in fh if line.strip())  # skip blank lines
        rows_per_file[f.name] = n_rows
        total_rows += n_rows

        if verbose:
            print(f"  {f.name:40s}  {n_rows:>8d} rows")

    return {
        "files"         : files,
        "rows_per_file" : rows_per_file,
        "total_rows"    : total_rows,
        "total_files"   : len(files),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Count total MCMC cases across all mcmc_vac_*.dat files."
    )
    parser.add_argument(
        "samples_dir",
        type=Path,
        nargs="?",
        default=DEFAULT_DIR,
        help="Directory containing mcmc_vac_*.dat files.",
    )
    parser.add_argument(
        "--pattern", type=str, default="mcmc_vac_*.dat",
        help="Glob pattern for files to include (default: mcmc_vac_*.dat).",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Only print the summary, not per-file counts.",
    )
    args = parser.parse_args()

    if not args.samples_dir.exists():
        print(f"ERROR: directory not found: {args.samples_dir}")
        raise SystemExit(1)

    print(f"Scanning: {args.samples_dir}")
    print(f"Pattern:  {args.pattern}")
    print("-" * 55)

    result = count_cases(
        samples_dir = args.samples_dir,
        pattern     = args.pattern,
        verbose     = not args.quiet,
    )

    print("-" * 55)
    print(f"Files found   : {result['total_files']}")
    print(f"Total rows    : {result['total_rows']:,}")
    print(f"(= total cases: {result['total_rows']:,} "
          f"= {result['total_files']} files × "
          f"{result['total_rows'] // result['total_files'] if result['total_files'] > 0 else 0} rows/file avg)")


if __name__ == "__main__":
    main()