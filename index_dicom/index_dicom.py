"""Extract metadata from DICOM images files to Pandas DataFrame and store as CSV / parquet / pickle.

Set chunk_size to save the results at set intervals.
Specify dicom_tags to retrieve using keywords, e.g. Series Description (0008,103E) is 'SeriesDescription'.
https://dicom.innolitics.com/ciods/mr-image/general-series/0008103e
Requires pyarrow to save parquet files.
"""
import argparse
import os
from pathlib import Path
import csv
from math import floor, ceil
import gc
from datetime import timedelta
import warnings

import pandas as pd
from pydicom import dcmread, Dataset
from pydicom.errors import InvalidDicomError
from pydicom.datadict import keyword_for_tag, dictionary_VR, dictionary_VM, dictionary_has_tag
from pydicom.multival import MultiValue
from pydicom.uid import UID
from pydicom.valuerep import DSfloat, IS, PersonName
import timeit
from tqdm import tqdm
import pyarrow as pa


_default_keywords_dir = [
    'PatientID', 'PatientSex', 'PatientBirthDate',
    'StudyInstanceUID', 'AccessionNumber', 'StudyDate', 'StudyTime', 'StudyDescription',
    'SeriesInstanceUID', 'SeriesNumber', 'SeriesDate', 'SeriesTime', 'SeriesDescription',
]

_default_keywords_file = _default_keywords_dir + [
    'SOPInstanceUID', 'InstanceNumber', 'AcquisitionDate', 'AcquisitionTime'
]

DEFAULT_KEYWORDS = {'dir': _default_keywords_dir, 'file': _default_keywords_file}

class DicomIndexer():
    # level: str
    # input_dir: str | Path
    # output_dir: str | Path
    # chunk_size: int | None = None
    # dicom_attributes: list | None = None

    # Initialise by checking the variables are valid
    def __init__(self, level: str, input_dir: str | Path, output_dir: str | Path, chunk_size: int | None = None, dicom_attributes: list | None = None, overwrite: bool = False):
        self.level = level.lower()
        self.input_dir = Path(input_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.chunk_size = chunk_size
        self.overwrite = overwrite
        if dicom_attributes:
            self.dicom_attributes = dicom_attributes
        else:
            # Make sure self.dicom_attributes is always a list, even if None or empty list is passed
            self.dicom_attributes = []

         # Check level is valid
        if self.level not in ["file", "dir"]:
            raise ValueError("Invalid level. Must be 'file' or 'dir'.")
        else:
            my_text = {'dir': "first file in each directory", 'file': "individual files"}
            print(f"Extracting metadata from {my_text[self.level]}.")

         # Check input directory exists
        if not self.input_dir.is_dir():
            raise ValueError(f"Input directory {self.input_dir} does not exist.")
        else:
            print(f"Input directory: {self.input_dir}")

         # Check output directory exists or create it
        if not self.output_dir.is_dir():
            print(f"Output directory: {self.output_dir} (creating new).")
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            print(f"Output directory: {self.output_dir} (already exists).")

         # Check chunk size is valid
        if self.chunk_size is not None:
            if not isinstance(self.chunk_size, int) or self.chunk_size <= 0:
                raise ValueError("Chunk size must be a positive integer.")
            else:
                print(f"Processing files in chunks of {self.chunk_size}.")
            self.chunked = True
        else:
            print("Processing all files at once (no chunking).")
            self.chunked = False

         # Check DICOM attributes are valid
        if not self.dicom_attributes or len(self.dicom_attributes) == 0:
            print("Using default DICOM attributes.")
            self.attribute_list = DEFAULT_KEYWORDS[self.level]
        elif len(self.dicom_attributes) == 1:
            if dictionary_has_tag(self.dicom_attributes[0]):
                self.attribute_list = self.dicom_attributes
            elif self.dicom_attributes[0] == "*":
                print("Extracting all DICOM attributes. Warning: this is intended for DICOM query results with a limited number of attributes, not image files, which will produce a very large output.")
                self.attribute_list = self.dicom_attributes
            else:
                # If the single attribute is not "*" and is not a valid DICOM keyword, check if it's a file containing keywords
                attribute_list_file = Path(self.dicom_attributes[0])
                if attribute_list_file.is_file():
                    with open(attribute_list_file, 'r') as f:
                        self.attribute_list = [line.strip() for line in f if line.strip()]
                else:
                    raise ValueError(f"Invalid DICOM attribute keyword or file: {self.dicom_attributes[0]}")
        else:
            # A list of multiple items must be keywords
            self.attribute_list = self.dicom_attributes
                
        # Now self.dicom_attributes must be a list keywords or "*". If it's not "*", check all keywords are valid.
        if self.attribute_list != ["*"]:
            for attr in self.attribute_list:
                if not dictionary_has_tag(attr):
                    raise ValueError(f"Invalid DICOM attribute keyword: {attr}")
            print(f"Extracting {len(self.attribute_list)} DICOM attributes.")

        if self.overwrite:
            print("Overwrite mode is ON: existing output files will be replaced.")

    def list_dcm_files(self, refresh: bool = False) -> list[Path]:
        """List DICOM ``.dcm`` files recursively and cache the result.

        Uses ``os.walk(..., followlinks=True)`` so symlinked directories are traversed.

        Assumption for ``level='dir'``:
            one directory corresponds to one DICOM series; in this mode, only the first
            ``.dcm`` file found in each directory is listed.

        Parameters:
            refresh: If True, ignore any existing cached list and rescan.

        Returns:
            List of DICOM file paths.
        """
        list_file = self.output_dir / (
            "dicom_dir_first_file_list.txt" if self.level == "dir" else "dicom_file_list.txt"
        )

        if list_file.exists() and not refresh:
            with list_file.open("r") as f:
                cached_files = [Path(line.rstrip()) for line in f if line.strip()]
            print(f"Using existing list of {len(cached_files)} DICOM files from {list_file}.")
            return cached_files

        print("Listing DICOM .dcm files (this may take some time).")
        tic = timeit.default_timer()
        dcm_files: list[Path] = []

        for root, dirs, files in os.walk(self.input_dir, followlinks=True):
            dirs.sort()
            files.sort()

            if self.level == "dir":
                first_dcm = next((f for f in files if f.lower().endswith(".dcm")), None)
                if first_dcm is not None:
                    dcm_files.append(Path(root) / first_dcm)
            else:
                for file_name in files:
                    if file_name.lower().endswith(".dcm"):
                        dcm_files.append(Path(root) / file_name)

        with list_file.open("w") as f:
            for path in dcm_files:
                f.write(str(path) + "\n")

        toc = timeit.default_timer()
        print(f"Listed {len(dcm_files)} DICOM files in {(toc - tic):.2f} seconds.")
        return dcm_files

    def concatenate_chunks(self) -> None:
        """Concatenate saved chunk files into final output.
        
        Reads chunk*.parquet or chunk*.pickle files and combines them into
        dicom_index.csv, dicom_index.parquet, and dicom_index.pickle.
        """
        print("Concatenating chunks.")
        tic = timeit.default_timer()
        df_list = []
        missing_chunks = []
        
        for chunk in tqdm(range(self.n_chunk), total=self.n_chunk, desc="Reading chunks"):
            chunk_stem = self.output_dir / f'dicom_index_chunk{chunk:0{self.chunk_width}}'
            if chunk_stem.with_suffix('.parquet').is_file():
                dicom_index = pd.read_parquet(chunk_stem.with_suffix('.parquet'))
                df_list.append(dicom_index)
            elif chunk_stem.with_suffix('.pickle').is_file():
                dicom_index = pd.read_pickle(chunk_stem.with_suffix('.pickle'))
                df_list.append(dicom_index)
            else:
                missing_chunks.append(chunk)
        
        dicom_index = pd.concat(df_list, ignore_index=True)
        # Will need save_tables method - placeholder for now
        # save_tables(dicom_index, self.output_dir / 'dicom_index')
        
        toc = timeit.default_timer()
        print(f"Concatenated {self.n_chunk - len(missing_chunks)} chunks in {toc - tic:.1f} seconds.")
        if len(missing_chunks) > 0:
            print("Warning: Could not find every chunk. Concatenated index saved without these chunks:")
            print(missing_chunks)

    def dcm_to_tags(self, dicom_file: Path) -> dict:
        """Read a DICOM file and return a dictionary of DICOM attributes.
        
        Adds meta-attributes for filepath. Handles errors.
        Will read only specified attributes if keywords provided, otherwise reads all.

        Parameters:
            dicom_file: Path to DICOM file.

        Returns:
            Dictionary of DICOM attributes.
        """
        this_data = {
            'dicom_filepath': str(dicom_file),
            'warnings': None,
            'error': None
        }
        
        try:
            read_warning = None
            with warnings.catch_warnings(record=True) as w:
                ds = dcmread(dicom_file)
                if len(w) > 0:
                    read_warning = 'dcmread: ' + str(w[0].message).split(':')[0]
        except InvalidDicomError as e:
            this_data['error'] = str(e)
            return this_data

        tag_warnings: list[str] = []
        try:
            if self.attribute_list == ["*"]:
                tag_dict, tag_warnings = self.get_all_elements(ds)
            else:
                tag_dict, tag_warnings = self.get_named_elements(ds, self.attribute_list)
            this_data.update(tag_dict)
        except Exception as e:
            this_data['error'] = f"Error processing DICOM metadata: {str(e)}"

        if read_warning:
            if tag_warnings:
                this_data['warnings'] = '\\n'.join([read_warning] + tag_warnings)
            else:
                this_data['warnings'] = read_warning
        elif tag_warnings:
            this_data['warnings'] = '\\n'.join(tag_warnings)

        return this_data

    def get_named_elements(self, ds, keywords: list[str]) -> tuple[dict, list[str]]:
        """Extract specified DICOM elements by keyword from a Dataset.
        
        TODO: Implement extraction logic based on query.py approach.
        
        Parameters:
            ds: pydicom Dataset
            keywords: List of DICOM keywords to extract
            
        Returns:
            Tuple of (attribute_dict, warnings_list)
        """
        # Placeholder - to be implemented with query.py logic
        return {}, []

    def get_all_elements(self, ds) -> tuple[dict, list[str]]:
        """Extract all DICOM elements from a Dataset.
        
        TODO: Implement extraction logic based on query.py approach.
        
        Parameters:
            ds: pydicom Dataset
            
        Returns:
            Tuple of (attribute_dict, warnings_list)
        """
        # Placeholder - to be implemented with query.py logic
        return {}, []

    def save_tables(self, dicom_index, file_stem: Path) -> None:
        """Save the DICOM index as CSV and parquet file.
        
        If there is an error saving to parquet, save as pickle and write an error file.

        Parameters:
            dicom_index: DataFrame containing the DICOM index.
            file_stem: Path to the output file stem.
        """
        dicom_index.to_csv(file_stem.with_suffix('.csv'), index=False, quoting=csv.QUOTE_NONNUMERIC)

        # Sanitize columns for parquet compatibility
        empty_lists = pd.Series([[]] * len(dicom_index), index=dicom_index.index)
        for col in dicom_index.columns:
            try:
                pa.Array.from_pandas(dicom_index[col])
            except Exception:
                if dictionary_has_tag(col):
                    vr = dictionary_VR(col)
                    vm = dictionary_VM(col)
                    if vm == '1':
                        if vr in ['DS', 'IS', 'FL', 'FD', 'OF', 'SL', 'SS', 'UL', 'US']:
                            dicom_index[col] = pd.to_numeric(dicom_index[col], errors='coerce')
                        else:
                            dicom_index[col] = dicom_index[col].fillna('')
                            dicom_index[col] = dicom_index[col].astype(str)
                            dicom_index[col] = dicom_index[col].replace('[]', '')
                    else:
                        dicom_index[col] = dicom_index[col].replace('', pd.NA)
                        dicom_index[col] = dicom_index[col].fillna(empty_lists)
                        dicom_index[col] = dicom_index[col].apply(lambda x: x if isinstance(x, list) else [x])
                else:
                    try_numeric = pd.to_numeric(dicom_index[col], errors='coerce')
                    if try_numeric.notna().sum() / len(dicom_index) > 0.5:
                        dicom_index[col] = try_numeric
                    elif dicom_index[col].apply(lambda x: isinstance(x, list)).sum() > 0.5:
                        dicom_index[col] = dicom_index[col].replace('', pd.NA)
                        dicom_index[col] = dicom_index[col].fillna(empty_lists)
                        dicom_index[col] = dicom_index[col].apply(lambda x: x if isinstance(x, list) else [x])
                    else:
                        dicom_index[col] = dicom_index[col].fillna('')
                        dicom_index[col] = dicom_index[col].astype(str)
                        dicom_index[col] = dicom_index[col].replace('[]', '')

        try:
            dicom_index.to_parquet(file_stem.with_suffix('.parquet'))
        except Exception as e:
            print(f"Error saving to parquet, saving as pickle instead:\\n{e}")
            dicom_index.to_pickle(file_stem.with_suffix('.pickle'))
            with file_stem.with_suffix('.parquet.error').open('w') as f:
                f.write(str(e))

    def run(self) -> None:
        """Execute the DICOM indexing workflow.
        
        Processes files in chunks if configured, otherwise processes all at once.
        """
        tic_total = timeit.default_timer()
        
        if not self.chunked:
            # Non-chunked: process all files at once
            tic = timeit.default_timer()
            metadata = []
            for dicom_file in tqdm(self.dcm_files, total=self.n_dcm, desc="Indexing files"):
                metadata.append(self.dcm_to_tags(dicom_file))
            toc = timeit.default_timer()
            print(f"Indexed {self.n_dcm} DICOM files in {(toc - tic):.0f} seconds.")
            
            dicom_index = pd.DataFrame(metadata)
            self.save_tables(dicom_index, self.output_dir / 'dicom_index')
        else:
            # Chunked: process and save in chunks
            assert self.chunk_size is not None
            chunk_times = {}
            for chunk in tqdm(range(self.chunks_saved, self.n_chunk), total=self.n_chunk, desc="Processing chunks"):
                tic_chunk = timeit.default_timer()
                
                start = chunk * self.chunk_size
                end = min((chunk + 1) * self.chunk_size, self.n_dcm)
                chunk_files = self.dcm_files[start:end]
                
                metadata = []
                for dicom_file in tqdm(chunk_files, total=len(chunk_files), desc=f"Chunk {chunk}", leave=False):
                    metadata.append(self.dcm_to_tags(dicom_file))
                
                dicom_index = pd.DataFrame(metadata)
                self.save_tables(dicom_index, self.output_dir / f'dicom_index_chunk{chunk:0{self.chunk_width}}')
                
                toc_chunk = timeit.default_timer()
                chunk_times[f'{chunk:0{self.chunk_width}}'] = toc_chunk - tic_chunk
            
            for n, t in chunk_times.items():
                print(f"Chunk {n}: {t:.0f} seconds.")
            
            self.concatenate_chunks()
        
        toc_total = timeit.default_timer()
        total_delta = timedelta(seconds=(toc_total - tic_total))
        print(f"Total time: {total_delta}")

    def prepare_run(self, refresh_file_list: bool | None = None) -> None:
        """Prepare file list, output safety checks, and chunk metadata."""
        if refresh_file_list is None:
            refresh_file_list = self.overwrite

        self.dcm_files = self.list_dcm_files(refresh=refresh_file_list)
        self.n_dcm = len(self.dcm_files)
        if self.n_dcm == 0:
            raise ValueError(f"No .dcm files found under {self.input_dir}.")

        self.count_width = len(str(self.n_dcm))

        final_stem = self.output_dir / "dicom_index"
        final_outputs = [
            final_stem.with_suffix(".csv"),
            final_stem.with_suffix(".parquet"),
            final_stem.with_suffix(".pickle"),
        ]

        def _handle_existing_files(existing_files: list[Path]) -> None:
            if not existing_files:
                return
            if not self.overwrite:
                existing_str = ", ".join(str(p.name) for p in existing_files)
                raise FileExistsError(
                    f"Found existing output files in {self.output_dir}: {existing_str}. "
                    "Use --overwrite to replace them."
                )
            for file_path in existing_files:
                file_path.unlink()
            print(f"Deleted {len(existing_files)} existing output file(s).")

        if not self.chunked:
            existing_final_outputs = [p for p in final_outputs if p.exists()]
            _handle_existing_files(existing_final_outputs)
            return

        assert self.chunk_size is not None
        self.n_chunk = ceil(self.n_dcm / self.chunk_size)
        if self.n_chunk <= 1:
            self.chunked = False
            self.chunk_size = None
            print("Chunk size larger than number of files; processing all files at once.")
            existing_final_outputs = [p for p in final_outputs if p.exists()]
            _handle_existing_files(existing_final_outputs)
            return

        self.chunk_width = len(str(self.n_chunk - 1))
        chunk_parquet = sorted(self.output_dir.glob("dicom_index_chunk*.parquet"))
        chunk_pickle = sorted(self.output_dir.glob("dicom_index_chunk*.pickle"))
        existing_chunks = chunk_parquet + chunk_pickle

        if self.overwrite and existing_chunks:
            for file_path in existing_chunks:
                file_path.unlink()
            print(f"Deleted {len(existing_chunks)} existing chunk file(s).")
            self.chunks_saved = 0
            return

        self.chunks_saved = len(existing_chunks)
        if self.chunks_saved == 0:
            return

        if self.chunks_saved > self.n_chunk:
            raise FileExistsError(
                f"Found {self.chunks_saved} chunk files in {self.output_dir} but expected at most {self.n_chunk}. "
                "Please clean up the output directory or use --overwrite."
            )

        for check_chunk in range(self.chunks_saved):
            chunk_stem = self.output_dir / f"dicom_index_chunk{check_chunk:0{self.chunk_width}}"
            if not chunk_stem.with_suffix('.parquet').is_file() and not chunk_stem.with_suffix('.pickle').is_file():
                raise FileNotFoundError(
                    f"Found {self.chunks_saved} chunk files, but missing expected chunk "
                    f"{chunk_stem.name}.parquet/.pickle. Please clean up or use --overwrite."
                )

        if self.chunks_saved == self.n_chunk:
            print(f"Found all {self.chunks_saved} chunk files.")
            existing_final_outputs = [p for p in final_outputs if p.exists()]
            if not existing_final_outputs:
                print("Final outputs not found. Will concatenate chunks.")
                self.concatenate_chunks()
                print("Concatenation complete. Exiting.")
                raise SystemExit(0)
            else:
                print(f"Final outputs already exist: {', '.join(p.name for p in existing_final_outputs)}. Nothing to do.")
                raise SystemExit(0)
        else:
            print(f"Found {self.chunks_saved} existing chunk files. Will resume from next chunk.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract metadata from DICOM files to CSV / parquet / pickle.",
        epilog="Warning: setting attributes to '*' will retrieve all attributes; "\
               "it is intended for DICOM query results with a limited number of attributes, "\
               "not image files, which will produce a very large output.")
    parser.add_argument(
        "--level",
        type=str,
        choices=["file", "dir"],
        required=True,
        help="Extraction level: 'file' for individual files, 'dir' for first file in each directory."
        )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Root of directory tree containing DICOM files."
        )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory to save output."
        )
    parser.add_argument(
        "--chunk_size",
        type=int,
        help="Number of files to process before saving."
        )
    parser.add_argument(
        "--attributes",
        nargs="*",
        help="DICOM attributes to extract (text file, one or more DICOM keywords, or '*'; default is a small set of basic attributes)."
        )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, overwrite existing output files/chunks in output_dir."
        )
    args = parser.parse_args()

    indexer = DicomIndexer(level=args.level, input_dir=args.input_dir, output_dir=args.output_dir, chunk_size=args.chunk_size, dicom_attributes=args.attributes, overwrite=args.overwrite)
    indexer.prepare_run()
    indexer.run()
