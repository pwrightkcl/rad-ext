"""Rename sourcedata NIfTI files to BIDS convention using previously defined entities in DICOM index.

To handle cases where dcm2niix produces multiple NIfTI files per DICOM directory, the description part of the filename
is converted to alphanumeric and used for the 'acq' entity. If there are multiple NIfTI files for one JSON file, the
part of the filename that varies is used for the 'rec' entity. If there is already a 'rec' entity from the DICOM index
then the filename part is appended.

Selected fields from the DICOM index are added as a convenience to save having to do a lookup later.
"""
from pathlib import Path
import re
import csv

import pandas as pd


root_dir = Path('/nfs/project/WellcomeHDN/kch-sequence-classifier')
sourcedata_dir = root_dir / 'sourcedata'
rawdata_dir = root_dir / 'rawdata'
metadata_dir = root_dir / 'metadata'
rawdata_index_file = metadata_dir / 'rawdata_index'
nifti_dir = sourcedata_dir / 'nifti'
commands = root_dir / 'code' / 'sourcedata' / 'source2raw_commands.sh'

di = pd.read_parquet(metadata_dir / 'dicom_index_imported.parquet')
if 'suffix' not in di.columns:
    print("DICOM index does not have a suffix column. Cannot continue")
    exit()

metadata_list = []
counter = 0
with commands.open('w') as f:
    f.write('#!/bin/bash\n')
for json_file in nifti_dir.glob('**/*.json'):
    counter += 1
    print(f'\r{counter:06}', end='', flush=True)
    this_meta = {'valid': True, 'reason': 'valid'}

    nifti_path = json_file.parent
    this_meta['source_dir'] = str(nifti_path)
    this_meta['source_json'] = json_file.name

    dicom_path = str(nifti_path).replace('/nifti/', '/dicom/')
    row_bool = di['output_path'].str.fullmatch(dicom_path)
    if sum(row_bool) == 0:
        print("\nWarning: could not match nifti path to a dicom path in DICOM index. Cannot process.")
        print(nifti_path)
        this_meta['valid'] = False
        this_meta['reason'] = "Not in DICOM index."
        metadata_list.append(this_meta)
        continue
    if sum(row_bool) > 1:
        print("\nWarning: nifti path matches more than one dicom path in DICOM index. Cannot process.")
        print(nifti_path)
        this_meta['valid'] = False
        this_meta['reason'] = "Multiple matches in DICOM index."
        metadata_list.append(this_meta)
        continue
    dicom_index_row = di.loc[row_bool, :].squeeze()

    this_meta['SeriesDateTime'] = dicom_index_row['SeriesDateTime']
    this_meta['SeriesDescription'] = dicom_index_row['SeriesDescription']
    this_meta['ImageType1'] = dicom_index_row['ImageType1']
    this_meta['ImageType2'] = dicom_index_row['ImageType2']
    this_meta['ImageType3'] = dicom_index_row['ImageType3']
    this_meta['subject_visit'] = dicom_index_row['subject_visit']

    nifti_path_parts = str(nifti_path).split('/')
    this_meta['series'] = nifti_path_parts[-1]
    this_meta['session'] = nifti_path_parts[-2]
    this_meta['subject'] = nifti_path_parts[-3]
    sub = 'sub-' + this_meta['subject']
    ses = 'ses-' + this_meta['session']
    run = 'run-' + re.sub(r'[^a-zA-Z0-9]', '', this_meta['series'])
    raw_stem = '_'.join([sub, ses, run])

    json_stem = json_file.stem
    bits = json_stem.split('_')
    this_meta['acquisition'] = '_'.join(bits[3:])
    if this_meta['acquisition']:
        acq = 'acq-' + re.sub(r'[^a-zA-Z0-9]', '', this_meta['acquisition'])
        raw_stem = '_'.join([raw_stem, acq])

    if 'contrast_enhancement' in dicom_index_row and dicom_index_row['contrast_enhancement']:
        this_meta['contrast_enhancement'] = dicom_index_row['contrast_enhancement']
        ce = 'ce-' + dicom_index_row['contrast_enhancement']
        raw_stem = '_'.join([raw_stem, ce])

    if 'reconstruction' in dicom_index_row and dicom_index_row['reconstruction']:
        this_meta['reconstruction'] = dicom_index_row['reconstruction']
        rec = 'rec-' + dicom_index_row['reconstruction']
        raw_stem = '_'.join([raw_stem, rec])
    else:
        rec = ''

    suffix = dicom_index_row['suffix']
    if not suffix:
        print("Warning: no suffix found. Cannot process.")
        print(nifti_path)
        this_meta['valid'] = False
        this_meta['reason'] = "No suffix."
        metadata_list.append(this_meta)
        continue
    raw_stem = '_'.join([raw_stem, suffix])
    raw_dir = rawdata_dir / sub / ses / 'anat'
    if not raw_dir.is_dir():
        with commands.open('a') as f:
            f.write(f'mkdir -p "{raw_dir}"\n')
    raw_json = (raw_dir / raw_stem).with_suffix('.json')
    this_meta['json_file'] = str(raw_json)
    if not raw_json.is_file():
        with commands.open('a') as f:
            f.write(f'cp "{json_file}" "{raw_json}"\n')

    nifti_count = 0
    for nifti_file in json_file.parent.glob(f'{json_stem}*.nii'):
        nifti_count += 1
        nifti_stem = nifti_file.stem
        if nifti_stem == json_stem:
            raw_nifti = (raw_dir / raw_stem).with_suffix('.nii')
            this_meta[f'nifti_file_{nifti_count}'] = str(raw_nifti)
            this_meta[f'source_nifti_{nifti_count}'] = nifti_file.name
            if not raw_nifti.is_file():
                with commands.open('a') as f:
                    f.write(f'cp "{nifti_file}" "{raw_nifti}"\n')
        elif not nifti_file.with_suffix('.json').exists():
            extra_rec = re.sub(r'[^a-zA-Z0-9]', '', nifti_stem.replace(json_stem, ''))
            if 'reconstruction' in this_meta:
                # There is an existing 'rec' entity based on heuristics applied to DICOM tags.
                new_rec = rec + extra_rec
                raw_nifti = (raw_dir / raw_stem.replace(rec, new_rec)).with_suffix('.nii')
            else:
                # There is no 'rec' entity, so make a new one before the suffix in the filename.
                new_rec = 'rec-' + extra_rec
                # Need to match on '_' + 'suffix' not 'suffix' because suffix may appear in the 'acq' entity.
                raw_nifti = (raw_dir / raw_stem.replace('_' + suffix,
                                                        '_' + '_'.join([new_rec, suffix])
                                                        )).with_suffix('.nii')
            this_meta[f'nifti_file_{nifti_count}'] = str(raw_nifti)
            this_meta[f'source_nifti_{nifti_count}'] = nifti_file.name
            if not raw_nifti.is_file():
                with commands.open('a') as f:
                    f.write(f'cp "{nifti_file}" "{raw_nifti}"\n')
    metadata_list.append(this_meta)
print()
metadata_df = pd.DataFrame.from_dict(metadata_list)
metadata_df.to_csv(rawdata_index_file.with_suffix('.csv'), index=False, quoting=csv.QUOTE_NONNUMERIC)
metadata_df.to_parquet(rawdata_index_file.with_suffix('.parquet'))
