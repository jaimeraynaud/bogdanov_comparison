import itertools
import re
import shutil
from pathlib import Path

import numpy as np
import xarray


def set_up_default_logger():
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


import logging
logger = logging.getLogger(__name__)


def get_memory_mapped_file_contents(file_handle):
    import mmap
    return mmap.mmap(file_handle.fileno(), 0, access=mmap.ACCESS_READ)


class DatasetVariableName:
    INPUT = 'input'
    OUTPUT = 'output'


def jaime_format_file_to_xarray_zarr(
        input_dir: Path,              # CHANGE 1: replaced single input_path with input_dir (directory)
        output_path: Path,
        n_files: int,                 # CHANGE 2: added — number of files to read from directory
        n_params: int = 18,           # CHANGE 3: renamed from input_size to n_params — total params in file
        input_size: int = 16,         # CHANGE 4: added — actual input size after excluding distance and N_H
        n_phase_bins: int = 32,       # CHANGE 5: added — phase bin dimension of output
        n_energy_total: int = 700,    # CHANGE 6: added — total energy channels in file
        n_energy_keep: int = 400,     # CHANGE 7: added — energy channels to keep
        zarr_chunk_axis0_size: int = 20,  # CHANGE 8: reduced from 1000 — output is 200x larger (64->12800 values)
) -> None:
    set_up_default_logger()
    if output_path.exists():
        shutil.rmtree(output_path)

    # CHANGE 9: get first n_files files from directory, sorted for reproducibility
    input_paths = sorted(input_dir.glob('mcmc_vac_*.dat'))[:n_files]
    if len(input_paths) == 0:
        raise ValueError(f'No mcmc_vac_*.dat files found in {input_dir}')
    if len(input_paths) < n_files:
        logger.warning(f'Requested {n_files} files but only {len(input_paths)} found in {input_dir}')
    logger.info(f'Reading {len(input_paths)} files from {input_dir}')

    # CHANGE 10: input_indices excludes 16th and 17th values (0-indexed: 15 and 16)
    # which are distance (fixed at 1.0 kpc) and N_H (fixed at 0.0)
    input_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17]

    # CHANGE 11: OUTPUT encoding is now 3D — added n_phase_bins dimension
    # Original: {'chunks': (zarr_chunk_axis0_size, output_size)}
    # New:      {'chunks': (zarr_chunk_axis0_size, n_phase_bins, n_energy_keep)}
    encoding = {
        DatasetVariableName.INPUT: {'dtype': 'float32', 'chunks': (zarr_chunk_axis0_size, input_size)},
        DatasetVariableName.OUTPUT: {'dtype': 'float32', 'chunks': (zarr_chunk_axis0_size, n_phase_bins, n_energy_keep)},
    }

    total_rows_processed = 0  # CHANGE 12: track total rows across all files for logging

    # CHANGE 13: outer loop iterates over all input files
    for file_index, input_path in enumerate(input_paths):
        logger.info(f'Processing file {file_index + 1}/{len(input_paths)}: {input_path.name}')

        with input_path.open() as file_handle:
            file_contents = get_memory_mapped_file_contents(file_handle)
            value_iterator = re.finditer(rb'[^\s]+', file_contents)
            input_set = []
            output_set = []

            for index in itertools.count():
                # CHANGE 14: read all n_params (18) values but only keep input_indices (16 values)
                # Original read input_size values directly — here we read all 18 and filter
                all_params = []
                try:
                    all_params.append(float(next(value_iterator).group(0)))
                except StopIteration:
                    # CHANGE 15: flush remaining rows at end of EACH FILE
                    # Original only flushed at end of single file — now needed per file in the loop
                    if len(input_set) != 0:
                        partial_dataset = xarray.Dataset(data_vars={
                            DatasetVariableName.INPUT: (['index', 'input'], input_set),
                            DatasetVariableName.OUTPUT: (['index', 'phase', 'energy'], output_set),
                            # CHANGE 16: OUTPUT dims are now ['index', 'phase', 'energy'] — 3D
                            # Original: ['index', 'output'] — 2D
                        })
                        if not output_path.exists():
                            partial_dataset.to_zarr(output_path, encoding=encoding)
                        else:
                            partial_dataset.to_zarr(output_path, append_dim='index')
                        total_rows_processed += len(input_set)
                        logger.info(f'File {file_index + 1}/{len(input_paths)} done. '
                                    f'Total rows processed: {total_rows_processed}')
                        input_set = []
                        output_set = []
                    break  # move to next file

                for _ in range(n_params - 1):
                    all_params.append(float(next(value_iterator).group(0)))
                inputs = [all_params[i] for i in input_indices]  # select 16 values, skip distance and N_H

                # CHANGE 17: skip loglikelihood — same as original, no change needed here
                _ = float(next(value_iterator).group(0))

                # CHANGE 18: read n_phase_bins * n_energy_total values instead of flat output_size
                # Original: flat list of output_size values
                # New: read n_phase_bins * n_energy_total values, reshape to (n_phase_bins, n_energy_total),
                #      then keep only first n_energy_keep energy channels -> (n_phase_bins, n_energy_keep)
                flat_outputs = []
                for _ in range(n_phase_bins * n_energy_total):
                    flat_outputs.append(float(next(value_iterator).group(0)))
                output_2d = np.array(flat_outputs, dtype=np.float32).reshape(n_phase_bins, n_energy_total)
                output_2d = output_2d[:, :n_energy_keep]  # (32, 400)

                input_set.append(inputs)
                output_set.append(output_2d)

                # CHANGE 19: flush interval reduced from 100000 to 10000
                # Original: % 100000 — output was 64 values, RAM use was 25.6 MB per flush
                # New:      % 10000  — output is 12800 values, RAM use ~500 MB per flush
                if (index + 1) % 10000 == 0:
                    # CHANGE 20: OUTPUT dataset variable now has 3 dims ['index', 'phase', 'energy']
                    # Original: ['index', 'output'] — 2D
                    # New:      ['index', 'phase', 'energy'] — 3D
                    partial_dataset = xarray.Dataset(data_vars={
                        DatasetVariableName.INPUT: (['index', 'input'], input_set),
                        DatasetVariableName.OUTPUT: (['index', 'phase', 'energy'], output_set),
                    })
                    if not output_path.exists():
                        partial_dataset.to_zarr(output_path, encoding=encoding)
                    else:
                        partial_dataset.to_zarr(output_path, append_dim='index')
                    total_rows_processed += len(input_set)
                    logger.info(f'File {file_index + 1}/{len(input_paths)}, '
                                f'row {index + 1}. Total rows: {total_rows_processed}')
                    input_set = []
                    output_set = []


if __name__ == '__main__':
    import time

    start = time.time()
    n_files = 1

    jaime_format_file_to_xarray_zarr(
        input_dir=Path('data/mcmc_samples'),
        output_path=Path('data/mcmc_samples/dataset.zarr'),
        n_files=n_files,
    )

    elapsed = time.time() - start
    logger.info(f'Total time: {elapsed:.2f}s ({elapsed/60:.2f} min)')
    logger.info(f'Time per file: {elapsed/n_files:.2f}s ({elapsed/(n_files*60):.2f} min)')

    dataset = xarray.open_zarr('data/mcmc_samples/dataset.zarr')
    print(dataset)