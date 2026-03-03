"""Read selected tags from the first DICOM file in each subdirectory and save as CSV and parquet.

Set chunk_size to save the results at set intervals.
Specify dicom_tags to retrieve using keywords, e.g. Series Description (0008,103E) is 'SeriesDescription'.
https://dicom.innolitics.com/ciods/mr-image/general-series/0008103e
Requires pyarrow to save parquet files.
"""
import argparse
from pathlib import Path
import csv
from math import floor, ceil
import gc

import pandas as pd
import pydicom
from pydicom.multival import MultiValue
from pydicom.errors import InvalidDicomError
import timeit


def list_dicom_dirs(dicom_dir: Path, output_dir: Path) -> list:
    """List all DICOM directories in a directory and save to a text file.

    Parameters:
        dicom_dir: Path to directory with subdirectories potentially containing DICOM files.
        output_dir: Path to directory to save the list of DICOM directories.

    Returns:
        List of DICOM directories.
    """
    dir_list_file = output_dir / 'dicom_dir_list.txt'
    if dir_list_file.exists():
        with open(dir_list_file, 'r') as f:
            dir_list = [Path(line.rstrip()) for line in f]
        print(f"Using existing list of {len(dir_list)} DICOM directories.")
    else:
        print("Listing DICOM directories (this may take some time).")
        tic = timeit.default_timer()
        dir_list = []
        with dir_list_file.open('w') as f:
            for d in dicom_dir.glob('**/'):
                f.write(str(d) + '\n')
                dir_list.append(d)
        toc = timeit.default_timer()
        print(f"Listed {len(dir_list)} DICOM directories in {(toc - tic):.2f} seconds.")
    return dir_list


def main(in_dir, out_dir, fields_file, chunk_size):
    """Read selected tags from the first DICOM file in each subdirectory and save as CSV and parquet.

    Parameters:
        in_dir:         Directory containing DICOM files.
        out_dir:        Directory to save CSV and Parquet output.
        fields_file:    Text file containing list of DICOM keywords to index.
                        Default: basic study and series info.
        chunk_size:     Save output in chunks of N DICOM files. Will attempt to combined chunks at the end.

    Returns:
        None. Saves results to CSV and Parquet files in out_dir.
    """
    dicom_dir = Path(in_dir)
    output_dir = Path(out_dir)
    if not output_dir.is_dir():
        output_dir.mkdir()

    if fields_file:
        with open(fields_file, 'r') as f:
            fields = [line.rstrip() for line in f]
    else:
        fields = ['StudyDate', 'StudyTime', 'AccessionNumber', 'StudyInstanceUID',
                  'SeriesDate', 'SeriesTime', 'SeriesNumber', 'SeriesDescription', 'SeriesInstanceUID']

    print(f"Constructing DICOM index from {dicom_dir}.")

    dicom_dirs = list_dicom_dirs(dicom_dir, output_dir)
    n_dicom = len(dicom_dirs)
    count_width = len(str(n_dicom))

    tic_total = timeit.default_timer()
    tic_chunk = tic_total
    count = 0
    count_dcm_chunk = 0
    chunks_saved = None
    chunk_width = None
    n_chunk = None
    if chunk_size:
        n_chunk = ceil(n_dicom / chunk_size)
        if n_chunk == 1:
            chunk_size = None
        else:
            chunk_width = len(str(n_chunk))
            # Look for existing saved chunks and try to resume
            chunks_saved = len([f for f in output_dir.glob('dicom_index_chunk*.parquet')])
            if chunks_saved > 0:
                if chunks_saved > n_chunk:
                    print(f"Found {chunks_saved} files in {output_dir} matching dicom_index_chunk*.parquet but "
                          f"expected a maximum of {n_chunk}. Stopping to avoid overwriting data."
                          f"Please clean up {output_dir} or use a different output_dir.")
                    exit()
                else:
                    print("Found existing chunk files. Attempting to resume.")
                    for check_chunk in range(1, chunks_saved + 1):
                        if not (output_dir / f'dicom_index_chunk{check_chunk:0{chunk_width}}.parquet').is_file():
                            print(f"Found {chunks_saved} files in {output_dir} matching dicom_index_chunk*.parquet but "
                                  f"could not find dicom_index_chunk{check_chunk:0{chunk_width}}.parquet.")
                            print(f"Stopping to avoid overwriting data. "
                                  f"Please clean up {output_dir} or use a different out_dir.")
                            exit()
                count = (chunks_saved - 1) * chunk_size

    metadata = []
    count_dcm_all = 0
    while count < n_dicom:
        dicom_dir = dicom_dirs[count]
        count += 1
        if chunk_size:
            print(f"Chunk {chunks_saved+1:0{chunk_width}}/{n_chunk}. Directory {count+1:0{count_width}}/{n_dicom}.\r",
                  end="", flush=True)
        else:
            print(f"Directory {count + 1:0{count_width}}/{n_dicom}.\r", end="", flush=True)

        dcm_files = list(dicom_dir.glob('*.dcm'))
        if len(dcm_files) == 0:
            continue
        dicom_file = dcm_files[0]
        count_dcm_all += 1
        if chunk_size:
            count_dcm_chunk += 1

        # Filename parts
        this_data = {'dicom_filepath': str(dicom_file),
                     'error': None}

        # Try reading DICOM file and record error if it fails
        try:
            d = pydicom.dcmread(dicom_file)
        except InvalidDicomError as e:
            this_data['error'] = str(e)
            metadata.append(this_data)
            continue

        # Read selected fields into dictionary
        for field in fields:
            if field in d:
                this_data[field] = d[field].value
                if field in ['ImageType', 'PatientOrientation', 'ImagePositionPatient']:
                    # These fields are expected to be lists.
                    this_data[field] = list(this_data[field])
                elif type(this_data[field]) == MultiValue:
                    if field in ['SeriesDescription', 'ScanningSequence', 'ScanOptions', 'SequenceVariant']:
                        # These fields are usually single strings, so convert lists to space-separated string.
                        this_data[field] = ' '.join(this_data[field])
                    else:
                        # If in doubt, convert to string as is.
                        this_data[field] = str(this_data[field])
            else:
                this_data[field] = None

        metadata.append(this_data)
        del d, this_data
        gc.collect()

        if chunk_size:
            chunk = floor(count / chunk_size)
            if chunk > chunks_saved:
                toc_chunk = timeit.default_timer()
                chunk_time = toc_chunk - tic_chunk
                total_time = toc_chunk - tic_total
                print('\r', end='')
                print(f"In chunk {chunk}, indexed {count_dcm_chunk} DICOM files from {chunk_size} directories in "
                      f"{round(chunk_time)} seconds.")
                print(f"In total, indexed {count_dcm_all} DICOM files from {count} directories in {round(total_time)} "
                      f"seconds.")
                print('Converting to dataframe')
                dicom_index = pd.DataFrame.from_records(metadata)
                print(f"Saving chunk {chunk} as CSV.")
                csv_file = output_dir / f'dicom_index_chunk{chunk:0{chunk_width}}.csv'
                dicom_index.to_csv(csv_file, index=False, quoting=csv.QUOTE_NONNUMERIC)
                print(f"Saving chunk {chunk} as parquet.")
                try:
                    dicom_index.to_parquet(csv_file.with_suffix('.parquet'))
                except Exception as e:
                    print(
                        "Could not save as parquet. It's probably a data type pyarrow doesn't like. Saving as pickle.")
                    dicom_index.to_pickle(str(csv_file.with_suffix('.pickle')))
                    print("Here's the error message:")
                    print(repr(e))
                chunks_saved += 1
                del metadata, dicom_index
                gc.collect()
                metadata = []
                count_dcm_chunk = 0
                tic_chunk = timeit.default_timer()

    print('\r', end='')
    toc_total = timeit.default_timer()
    total_time = toc_total - tic_total
    if chunk_size:
        # Save final chunk
        chunk_time = toc_total - tic_chunk
        chunk_count = count - chunk_size * chunks_saved
        chunk = chunks_saved + 1
        print(f"In final chunk, indexed {count_dcm_chunk} DICOM files from {chunk_count} directories in "
              f"{round(chunk_time)} seconds.")
        print(f"In total, indexed {count_dcm_all} DICOM files from {count} directories in {round(total_time)} seconds.")
        print("Converting to dataframe.")
        dicom_index = pd.DataFrame.from_records(metadata)
        print(f"Saving final chunk {chunk} as CSV.")
        csv_file = output_dir / f'dicom_index_chunk{chunk:0{chunk_width}}.csv'
        dicom_index.to_csv(csv_file, index=False, quoting=csv.QUOTE_NONNUMERIC)
        print(f"Saving final chunk {chunk} as parquet.")
        try:
            dicom_index.to_parquet(csv_file.with_suffix('.parquet'))
        except Exception as e:
            print("Could not save as parquet. It's probably a data type pyarrow doesn't like. Saving as pickle.")
            dicom_index.to_pickle(str(csv_file.with_suffix('.pickle')))
            print("Here's the error message:")
            print(repr(e))

        print("Attempting to concatenate chunks.")
        df_list = []
        missing_chunks = []
        for chunk in range(1, n_chunk + 1):
            if (output_dir / f'dicom_index_chunk{chunk:0{chunk_width}}.parquet').is_file():
                dicom_index = pd.read_parquet(output_dir / f'dicom_index_chunk{chunk:0{chunk_width}}.parquet')
                df_list.append(dicom_index)
                print(".", end="")
            elif (output_dir / f'dicom_index_chunk{chunk:0{chunk_width}}.pickle').is_file():
                dicom_index = pd.read_pickle(output_dir / f'dicom_index_chunk{chunk:0{chunk_width}}.pickle')
                df_list.append(dicom_index)
                print(".", end="")
            else:
                missing_chunks.append(chunk)
        dicom_index = pd.concat(df_list)
        print("\nDone. Saving concatenated index.")
        if len(missing_chunks) > 0:
            print("Could not find every chunk. Concatenated index will be saved without these chunks:")
            print(missing_chunks)
        output_file = output_dir / 'dicom_index'
        try:
            dicom_index.to_parquet(output_file.with_suffix('.parquet'))
        except Exception as e:
            print("Could not save as parquet. It's probably a data type pyarrow doesn't like. Saving as pickle.")
            dicom_index.to_pickle(str(output_file.with_suffix('.pickle')))
            print("Here's the error message:")
            print(repr(e))
    else:
        print(f"Indexed {count_dcm_all} DICOM files from {count} directories in {round(total_time)} seconds.")
        print("Converting to dataframe.")
        dicom_index = pd.DataFrame.from_records(metadata)
        print("Saving as CSV.")
        csv_file = output_dir / 'dicom_index.csv'
        dicom_index.to_csv(csv_file, index=False, quoting=csv.QUOTE_NONNUMERIC)
        print("Saving as parquet.")
        try:
            dicom_index.to_parquet(csv_file.with_suffix('.parquet'))
        except Exception as e:
            print("Could not save as parquet. It's probably a data type pyarrow doesn't like. Saving as pickle.")
            dicom_index.to_pickle(str(csv_file.with_suffix('.pickle')))
            print("Here's the error message:")
            print(repr(e))
    print('Done\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Index selected tags from the each .dcm file. Will attempt to resume if saved chunks exist in "
                    "`out_dir`.")
    parser.add_argument(
        '--in_dir',
        required=True,
        help='Root directory containing DICOM files.'
    )
    parser.add_argument(
        '--out_dir',
        required=True,
        help='Directory to save CSV and Parquet output.'
    )
    parser.add_argument(
        '--fields_file',
        required=False,
        help='Text file containing DICOM keywords to be indexed.'
    )
    parser.add_argument(
        '--chunk_size',
        required=False,
        type=int,
        help='Save output in chunks of N DICOM files. Will attempt to combined chunks at the end.'
    )
    main_args = parser.parse_args()
    main(in_dir=main_args.in_dir, out_dir=main_args.out_dir, fields_file=main_args.fields_file,
         chunk_size=main_args.chunk_size)
