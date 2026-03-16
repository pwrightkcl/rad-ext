from pathlib import Path
import re
import csv
import glob
import argparse

import pandas as pd
from tqdm import tqdm


def _find_nifti_json_sidecars(nifti_dir: Path) -> list[Path]:
    """Find all JSON sidecar files in the sourcedata nifti directory.

    Check for cached list and load this if present. If not list files and save to cache file.

    Args:
        nifti_dir: Path to the sourcedata nifti directory.

    Returns:
        A list of Paths to the JSON sidecar files.
    """
    cache_file = nifti_dir / 'json_sidecar_list.txt'
    if cache_file.is_file():
        with cache_file.open('r') as f:
            json_files = [Path(line.strip()) for line in f]
    else:
        json_files = list(nifti_dir.glob('**/*.json'))
        with cache_file.open('w') as f:
            for json_file in json_files:
                f.write(str(json_file) + '\n')
    return json_files


def main(project_dir: Path):
    """Create a BIDS-compliant rawdata directory from converted NIfTI+JSON files and DICOM index.

    Saves a rawdata index to the metadata directory and generates a shell script to create the rawdata directory
    and link the NIfTI and JSON files (plus .bval and .bvec, if present).

    Requires a DICOM index from `import_dicom_imported.py` with columns:
    - `dicom_path`: the path to the indexed DICOM data
    - `subject`: the subject ID to use in the BIDS filename
    - `suffix`: the BIDS suffix to use for the NIfTI files

    We assume the source DICOM data are in 'sourcedata/dicom' and the converted NIfTI files are in 'sourcedata/nifti'.

    To handle cases where dcm2niix produces multiple NIfTI files per DICOM series directory, the description part of the 
    filename is converted to alphanumeric and used for the 'acq' entity. If there are multiple NIfTI files for one JSON 
    file, the part of the filename that varies is used for the 'rec' entity. If there is already a 'rec' entity from the
    DICOM index then the filename part is appended.

    Selected fields from the DICOM index are added as a convenience to save having to do a lookup later.

    Args:
        project_dir: Root directory for the BIDS project.

    Returns:
        None
    """
    print(f"Creating BIDS-compliant rawdata directory for {project_dir}.")
    
    print("Loading DICOM index.")
    di = pd.read_parquet(project_dir / 'metadata' / 'dicom_index_imported.parquet')

    print("Generating commands.")
    commands = project_dir / 'code' / 'sourcedata' / 'generated_scripts' / 'source2raw_commands.sh'
    if not commands.parent.exists():
        commands.parent.mkdir(parents=True)
    with commands.open('w') as f:
        f.write('#!/bin/bash\n')
        f.write('# Create a BIDS-compliant rawdata directory with links to sourcedata NIfTI files.\n')
    metadata_list = []
    made_directories = []
    json_files = _find_nifti_json_sidecars(project_dir / 'sourcedata' / 'nifti')
    for json_file in tqdm(json_files, desc="Processing JSON sidecar files", total=len(json_files)):
        this_meta = {'valid': True, 'reason': 'valid'}

        nifti_path = json_file.parent
        this_meta['source_dir'] = str(nifti_path)
        this_meta['source_json'] = json_file.name

        dicom_path = str(nifti_path).replace('/nifti/', '/dicom/')
        this_meta['dicom_path'] = dicom_path
        row_bool = di['dicom_path'].str.fullmatch(dicom_path)
        if sum(row_bool) == 0:
            this_meta['valid'] = False
            this_meta['reason'] = "Not in DICOM index."
            metadata_list.append(this_meta)
            continue
        if sum(row_bool) > 1:
            this_meta['valid'] = False
            this_meta['reason'] = "Multiple matches in DICOM index."
            metadata_list.append(this_meta)
            continue
        di_row: pd.Series = di.loc[row_bool, :].squeeze()  # type: ignore

        this_meta['subject'] = di_row['subject']

        columns_to_copy = [
            'AccessionNumber',
            'StudyDescription',
            'SeriesNumber',
            'SeriesDate',
            'SeriesTime',
            'SeriesDateTime',
            'SeriesDescription',
        ]
        for column in columns_to_copy:
            if column in di_row:
                this_meta[column] = di_row[column]

        nifti_path_parts = str(nifti_path).split('/')
        series_dir = nifti_path_parts[-1]
        study_dir = nifti_path_parts[-2]
        this_meta['session'] = re.sub(r'[^a-zA-Z0-9]', '', study_dir)
        sub = f'sub-{this_meta["subject"]}'
        ses = f'ses-{this_meta["session"]}'
        run = f'run-{re.sub(r"[^a-zA-Z0-9]", "", series_dir)}'
        raw_stem = '_'.join([sub, ses, run])

        json_stem = json_file.stem
        json_stem_start = f'ses-{study_dir}_run-{series_dir}'
        if json_stem == json_stem_start:
            acquisition = ''
        elif json_stem.startswith(json_stem_start + '_'):
            acquisition = json_stem[len(json_stem_start)+1:]
        else:
            this_meta['valid'] = False
            this_meta['reason'] = "JSON filename does not start with expected pattern based on session and series directories."
            metadata_list.append(this_meta)
            continue
        this_meta['acquisition'] = acquisition

        if acquisition:
            acq = 'acq-' + re.sub(r'[^a-zA-Z0-9]', '', acquisition)
            raw_stem = '_'.join([raw_stem, acq])

        if 'contrast_enhancement' in di_row and di_row['contrast_enhancement']:
            this_meta['contrast_enhancement'] = di_row['contrast_enhancement']
            ce = 'ce-' + di_row['contrast_enhancement']
            raw_stem = '_'.join([raw_stem, ce])

        if 'reconstruction' in di_row and di_row['reconstruction']:
            this_meta['reconstruction'] = di_row['reconstruction']
            rec = 'rec-' + di_row['reconstruction']
            raw_stem = '_'.join([raw_stem, rec])
        else:
            rec = ''

        suffix = di_row['suffix']
        if not suffix:
            this_meta['valid'] = False
            this_meta['reason'] = "No suffix."
            metadata_list.append(this_meta)
            continue
        if suffix == 'unknown':
            this_meta['valid'] = False
            this_meta['reason'] = "Suffix is 'unknown', so cannot assign raw filename."
            metadata_list.append(this_meta)
            continue
        raw_stem = '_'.join([raw_stem, suffix])

        rawdata_dir = project_dir / 'rawdata' / sub / ses / 'anat'
        if rawdata_dir not in made_directories:
            made_directories.append(rawdata_dir)
            if not rawdata_dir.is_dir():
                # Convert rawdata_dir into a string usable in a shell script with single quotes escaped.
                rawdata_dir_str = str(rawdata_dir).replace("'", "'\\''")
                with commands.open('a') as f:
                    f.write(f"mkdir -p '{rawdata_dir_str}'\n")
        raw_json = (rawdata_dir / raw_stem).with_suffix('.json')
        this_meta['json_file'] = str(raw_json)
        if not raw_json.is_file():
            json_file_str = str(json_file).replace("'", "'\\''")
            with commands.open('a') as f:
                f.write(f"ln -s '{json_file_str}' '{raw_json}'\n")

        nifti_count = 1
        for nifti_file in json_file.parent.glob(f'{glob.escape(json_stem)}*.nii.gz'):
            # Remove .nii.gz from the end of the filename
            nifti_stem = nifti_file.name.removesuffix('.nii.gz')
            if nifti_stem == json_stem:
                raw_nifti = (rawdata_dir / raw_stem).with_suffix('.nii.gz')
            else:
                extra_chars = nifti_stem.replace(json_stem, '')
                # There might be another JSON file with part of the suffix.
                # Add one character of extra_chars at a time to nifti_stem and check whether a JSON exists. If so, skip.
                # This painstaking approach is because there may be examples like:
                # file1.json
                # file1.nii.gz
                # file1_1.json
                # file1_1.nii.gz
                # file1_1_ROI1.nii.gz
                # We do not want file1.json to match file1_1_ROI1.nii.gz, but we can only rule it out by matching part
                # of the extra characters to file1_1.json.
                there_is_another_json = False
                for i in range(1, len(extra_chars) + 1):
                    if (json_file.parent / (json_stem + extra_chars[:i] + '.json')).is_file():
                        there_is_another_json = True
                        break
                if there_is_another_json:
                    continue

                if 'reconstruction' in this_meta:
                    # There is an existing 'rec' entity based on heuristics applied to DICOM tags.
                    new_rec = rec + re.sub(r'[^a-zA-Z0-9]', '', extra_chars)
                    this_raw_stem = raw_stem.replace(rec, new_rec)
                else:
                    # There is no 'rec' entity, so make a new one before the suffix in the filename.
                    new_rec = 'rec-' + re.sub(r'[^a-zA-Z0-9]', '', extra_chars)
                    # Use regex to match only suffix at the end of raw_stem.
                    this_raw_stem = re.sub(rf'{suffix}$', '_'.join([new_rec, suffix]), raw_stem)

                raw_nifti = (rawdata_dir / this_raw_stem).with_suffix('.nii.gz')

            this_meta[f'source_nifti_{nifti_count}'] = nifti_file.name
            this_meta[f'nifti_file_{nifti_count}'] = str(raw_nifti)
            if not raw_nifti.is_file():
                nifti_file_str = str(nifti_file).replace("'", "'\\''")
                with commands.open('a') as f:
                    f.write(f"ln -s '{nifti_file_str}' '{raw_nifti}'\n")

            # If there are bval or bvec files, link these too.
            bval_file = nifti_path / nifti_file.name.replace('.nii.gz', '.bval')
            if bval_file.is_file():
                raw_bval = rawdata_dir / raw_nifti.name.replace('.nii.gz', '.bval')
                this_meta[f'bval_file_{nifti_count}'] = str(raw_bval)
                if not raw_bval.is_file():
                    bval_file_str = str(bval_file).replace("'", "'\\''")
                    with commands.open('a') as f:
                        f.write(f"ln -s '{bval_file_str}' '{raw_bval}'\n")
            bvec_file = nifti_path / nifti_file.name.replace('.nii.gz', '.bvec')
            if bvec_file.is_file():
                raw_bvec = rawdata_dir / raw_nifti.name.replace('.nii.gz', '.bvec')
                this_meta[f'bvec_file_{nifti_count}'] = str(raw_bvec)
                if not raw_bvec.is_file():
                    bvec_file_str = str(bvec_file).replace("'", "'\\''")
                    with commands.open('a') as f:
                        f.write(f"ln -s '{bvec_file_str}' '{raw_bvec}'\n")

            nifti_count += 1

        metadata_list.append(this_meta)

    with commands.open('a') as f:
        f.write(f'echo "{commands.name} complete."')
    print()
    print("Complete. Saving rawdata index.")
    metadata_df = pd.DataFrame.from_records(metadata_list)
    rawdata_index_file = project_dir / 'metadata' / 'rawdata_index'
    metadata_df.to_csv(rawdata_index_file.with_suffix('.csv'), index=False, quoting=csv.QUOTE_NONNUMERIC)
    metadata_df.to_parquet(rawdata_index_file.with_suffix('.parquet'))
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rename sourcedata NIfTI files to BIDS specification using previously defined entities in DICOM index.")
    parser.add_argument("project_dir", type=Path, help="Root directory for the BIDS project.")
    args = parser.parse_args()
    main(args.project_dir)
