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
from math import ceil
from datetime import timedelta
import warnings
from typing import Any

import pandas as pd
from pandas.api.types import is_list_like
from pydicom import dcmread, Dataset, DataElement
from pydicom.errors import InvalidDicomError
from pydicom.datadict import dictionary_VR, dictionary_VM, dictionary_has_tag
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
STRING_VRS = {"AE", "AS", "CS", "DA", "DT", "LO", "LT", "PN", "SH", "ST", "TM", "UC", "UI", "UR", "UT"}
DEFAULT_MAX_COLUMNS = 256


class ColumnLimitExceededError(ValueError):
    """Raised when flattened metadata exceeds configured DataFrame column limit."""

class DicomIndexer():
    # level: str
    # input_dir: str | Path
    # output_dir: str | Path
    # chunk_size: int | None = None
    # dicom_attributes: list | None = None

    # Initialise by checking the variables are valid
    def __init__(self, level: str, input_dir: str | Path, output_dir: str | Path, chunk_size: int | None = None, dicom_attributes: list | None = None, overwrite: bool = False, max_columns: int = DEFAULT_MAX_COLUMNS):
        self.level = level.lower()
        self.input_dir = Path(input_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.chunk_size = chunk_size
        self.dicom_attributes = dicom_attributes
        self.overwrite = overwrite
        self.max_columns = max_columns
        self.seen_columns: set[str] = set()

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
            # A single item could be a keyword, "*", or a file containing keywords. Check these possibilities in order.
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
                
        # Now self.dicom_attributes must be a list of keywords or "*". If it's not "*", check all keywords are valid.
        if self.attribute_list != ["*"]:
            for attr in self.attribute_list:
                if not dictionary_has_tag(attr):
                    raise ValueError(f"Invalid DICOM attribute keyword: {attr}")
            print(f"Extracting {len(self.attribute_list)} DICOM attributes.")

        if self.overwrite:
            print("Overwrite mode is ON: existing output files will be replaced.")

        if not isinstance(self.max_columns, int) or self.max_columns <= 0:
            raise ValueError("max_columns must be a positive integer.")
        print(f"Maximum allowed columns in output table: {self.max_columns}.")

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
                self.update_seen_columns(dicom_index.columns, context=f"reading chunk {chunk}")
            elif chunk_stem.with_suffix('.pickle').is_file():
                dicom_index = pd.read_pickle(chunk_stem.with_suffix('.pickle'))
                df_list.append(dicom_index)
                self.update_seen_columns(dicom_index.columns, context=f"reading chunk {chunk}")
            else:
                missing_chunks.append(chunk)
        
        dicom_index = pd.concat(df_list, ignore_index=True)
        self.validate_column_limit(dicom_index, context="concatenating chunks")
        self.save_tables(dicom_index, self.output_dir / 'dicom_index')
        toc = timeit.default_timer()
        print(f"Concatenated {self.n_chunk - len(missing_chunks)} chunks in {toc - tic:.1f} seconds.")
        if len(missing_chunks) > 0:
            print("Warning: Could not find every chunk. Concatenated index saved without these chunks:")
            print(missing_chunks)

    def dcm_to_tags(self, dicom_file: Path) -> dict:
        """Read a DICOM file and return a dictionary of DICOM attributes.
        
        Adds meta-attributes for filepath, warnings and errors.
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

        attribute_warnings: list[str] = []
        try:
            attribute_dict, attribute_warnings = self.dataset_to_attributes(ds, keywords=self.attribute_list)
            this_data.update(attribute_dict)
            self.update_seen_columns(this_data.keys(), context=f"extracting {dicom_file}")
        except ColumnLimitExceededError:
            raise
        except Exception as e:
            this_data['error'] = f"Error processing DICOM metadata: {str(e)}"

        if read_warning:
            if attribute_warnings:
                this_data['warnings'] = '\\n'.join([read_warning] + attribute_warnings)
            else:
                this_data['warnings'] = read_warning
        elif attribute_warnings:
            this_data['warnings'] = '\\n'.join(attribute_warnings)

        return this_data

    def dataset_to_attributes(self, ds: Dataset, keywords: list[str], prefix: str = "") -> tuple[dict, list[str]]:
        """Extract DICOM elements from a Dataset, flattening sequences recursively.

        If ``keywords`` is ["*"], all top-level elements except PixelData are included.
        """
        dicom_dict: dict[str, Any] = {}
        warning_clips: list[str] = []
        if keywords != ["*"]:
            keys: list[Any] = list(keywords)
        else:
            keys = [k for k in ds.keys() if k != (0x7FE0, 0x0010)]

        for key in keys:
            field_name = str(key)

            if isinstance(key, str):
                if key not in ds:
                    dicom_dict[prefix + key] = None
                    continue

            with warnings.catch_warnings(record=True) as w:
                el = ds[key]
                if isinstance(key, str):
                    _key = prefix + key
                elif not el.is_private and el.keyword:
                    _key = prefix + el.keyword
                else:
                    # Keep unknown/private tags as readable and ds[...] compatible hex keys.
                    _key = prefix + f'0x{el.tag:08x}'
                field_name = _key
                if len(w) > 0:
                    clip = str(w[0].message).split(':')[0]
                    warning_clips.append(f"Loading {field_name}: {clip}")

            if el.VR == 'SQ':
                for i, sq_ds in enumerate(el.value):
                    sq_prefix = f'{field_name}.{i}.'
                    sq_dict, sq_warnings = self.dataset_to_attributes(sq_ds, keywords=["*"], prefix=sq_prefix)
                    dicom_dict.update(sq_dict)
                    warning_clips.extend(sq_warnings)
            else:
                with warnings.catch_warnings(record=True) as w:
                    norm = self._normalise_vr(el)
                    dicom_dict[field_name] = self._convert_value(norm)
                    if len(w) > 0:
                        clip = str(w[0].message).split(':')[0]
                        warning_clips.append(f"Converting {field_name}: {clip}")

        return dicom_dict, warning_clips

    def _normalise_vr(self, el: DataElement):
        """Normalise DataElement values by VR/VM to stabilise downstream table typing."""
        value = el.value
        vr = el.VR
        tag = el.tag

        my_null = "" if vr in STRING_VRS else None

        if dictionary_has_tag(tag):
            vm = dictionary_VM(tag)
        else:
            # Cannot continue with tests if dictionary VM is unknown
            return value

        if is_list_like(value) and not isinstance(value, (str, bytes, bytearray)):
            if vm == "1":
                # Value should not be list-like. Try to correct.
                if len(value) == 0:
                    return my_null
                elif len(value) == 1:
                    if value[0] == "" or value[0] is None or pd.isna(value[0]):
                        return my_null
                    return value[0]
                else:
                    # Value has multiple elements. Convert to str.
                    # This works if VR is a string type. If VR is numeric, there is no good fix.
                    # Keeping as str at least allows later introspection of the data irregularity.
                    warnings.warn(
                        f"Tag {el.keyword!r} (VR={vr}, VM={vm}) has {len(value)} elements "
                        f"but VM=1 was expected; coercing to str: {value!r}",
                        stacklevel=3,
                    )
                    return str(value)
            else:
                # Convert pydicom MultiArray etc. to list
                return list(value)
        else:
            # This value is scalar
            if vm != "1":
                # Value should be list-like. Try to correct.
                if value == "" or value is None or pd.isna(value):
                    return []
                elif isinstance(value, str):
                    if '\\' in value:
                        return value.split('\\')
                    elif '/' in value:
                        return value.split('/')
                    else:
                        return [value]
                else:
                    return [value]
            elif value == "" or value is None or pd.isna(value):
                return my_null
            else:
                return value

    def _convert_value(self, v: Any) -> Any:
        """Convert pydicom value to a serializable value."""
        if is_list_like(v) and not isinstance(v, (str, bytes, bytearray)):
            # All list-like values should already by converted to list, but use broad check to be defensive.
            return [self._convert_value(mv) for mv in v]
        if pd.isna(v):
            return None
        if type(v) is DSfloat:
            return float(v) if str(v) != "" else None
        if type(v) is IS:
            return int(v) if str(v) != "" else None
        if type(v) in (int, float):
            return v
        if type(v) in (str, PersonName, UID):
            cv = self._sanitise_unicode(str(v))
            if len(cv) > 256:
                cv = cv[:253] + '...'
            return cv
        if type(v) is bytes:
            s = v.decode('ascii', 'replace')
            return self._sanitise_unicode(s)
        if type(v) is bytearray:
            s = v.decode('ascii', 'replace')
            return self._sanitise_unicode(s)
        return repr(v)

    @staticmethod
    def _sanitise_unicode(s: str) -> str:
        return s.replace(u"\u0000", "").strip()

    def validate_column_limit(self, dicom_index: pd.DataFrame, context: str = "") -> None:
        """Raise if output table width exceeds configured max_columns."""
        n_cols = len(dicom_index.columns)
        if n_cols > self.max_columns:
            context_msg = f" while {context}" if context else ""
            raise ColumnLimitExceededError(
                f"Column limit exceeded{context_msg}: {n_cols} columns generated, "
                f"but max_columns is set to {self.max_columns}. "
                "This is usually caused by flattening nested DICOM sequences; reduce selected attributes "
                "or reduce sequence complexity."
            )

    def update_seen_columns(self, columns: Any, context: str = "") -> None:
        """Track global union of seen columns and enforce max_columns early."""
        self.seen_columns.update(str(c) for c in columns)
        n_cols = len(self.seen_columns)
        if n_cols > self.max_columns:
            context_msg = f" while {context}" if context else ""
            raise ColumnLimitExceededError(
                f"Column limit exceeded{context_msg}: {n_cols} unique columns seen so far, "
                f"but max_columns is set to {self.max_columns}."
            )

    def save_tables(self, dicom_index, file_stem: Path) -> None:
        """Save the DICOM index as CSV and parquet file.
        
        If there is an error saving to parquet, save as pickle and write an error file.

        Parameters:
            dicom_index: DataFrame containing the DICOM index.
            file_stem: Path to the output file stem.
        """
        self.validate_column_limit(dicom_index, context=f"saving {file_stem.name}")
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
            self.validate_column_limit(dicom_index, context="creating index")
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
                self.update_seen_columns(dicom_index.columns, context=f"creating index chunk {chunk}")
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

        # If chunked, calculate n_chunk and downgrade if chunk_size >= n_dcm.
        if self.chunked:
            assert self.chunk_size is not None
            self.n_chunk = ceil(self.n_dcm / self.chunk_size)
            if self.n_chunk == 1:
                self.chunked = False
                print("Chunk size larger than number of files; processing all files at once.")

        # Handle existing chunk files (only when still chunked).
        if self.chunked:
            self.chunk_width = len(str(self.n_chunk - 1))
            chunk_parquet = sorted(self.output_dir.glob("dicom_index_chunk*.parquet"))
            chunk_pickle = sorted(self.output_dir.glob("dicom_index_chunk*.pickle"))
            existing_chunks = chunk_parquet + chunk_pickle

            if self.overwrite and existing_chunks:
                for file_path in existing_chunks:
                    file_path.unlink()
                print(f"Deleted {len(existing_chunks)} existing chunk file(s).")
                existing_final_outputs = [p for p in final_outputs if p.exists()]
                if existing_final_outputs:
                    for file_path in existing_final_outputs:
                        file_path.unlink()
                    print(f"Deleted {len(existing_final_outputs)} existing output file(s).")
                self.chunks_saved = 0
                return

            self.chunks_saved = len(existing_chunks)

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

            if self.chunks_saved < self.n_chunk:
                if self.chunks_saved > 0:
                    print(f"Found {self.chunks_saved} existing chunk files. Will resume from next chunk.")
                return

            # All n_chunk chunks are present; proceed to checking final files.
            print(f"Found all {self.chunks_saved} chunk files.")

        # Either all chunks are present (chunked) or we are not chunked.
        # Check the final files.
        existing_final_outputs = [p for p in final_outputs if p.exists()]
        if existing_final_outputs:
            if self.overwrite:
                for file_path in existing_final_outputs:
                    file_path.unlink()
                print(f"Deleted {len(existing_final_outputs)} existing output file(s).")
            else:
                print(f"Final outputs already exist: {', '.join(p.name for p in existing_final_outputs)}. Nothing to do.")
                raise SystemExit(0)

        # Final files are absent (or were just deleted).
        if self.chunked:
            print("Final outputs not found. Will concatenate chunks.")
            self.concatenate_chunks()
            print("Concatenation complete. Exiting.")
            raise SystemExit(0)
        # else: begin unchunked run.


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
    parser.add_argument(
        "--max_columns",
        type=int,
        default=DEFAULT_MAX_COLUMNS,
        help="Maximum number of columns allowed in output tables after flattening DICOM metadata."
        )
    args = parser.parse_args()

    indexer = DicomIndexer(level=args.level, input_dir=args.input_dir, output_dir=args.output_dir, chunk_size=args.chunk_size, dicom_attributes=args.attributes, overwrite=args.overwrite, max_columns=args.max_columns)
    indexer.prepare_run()
    indexer.run()
