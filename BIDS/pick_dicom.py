"""Apply heuristics to categorize DICOM images and pick images for a specific project.
"""
import argparse
from pathlib import Path
import csv

import pandas as pd

from dicom_heuristics import apply_heuristics


def _invalidate(di: pd.DataFrame, invalid: pd.Series, reason: str) -> None:
    """Mark DICOM index records as invalid based on a boolean mask and provide a reason for invalidation.
    
    Displays a count of how many records were invalidated and how many remain valid after the operation.

    Returns early if no records are to be invalidated.

    Args:
        di: The DICOM index DataFrame.
        invalid: A boolean Series indicating which records to invalidate.
        reason: A string describing the reason for invalidation.
        
    Returns:
        None. Modifies the input DataFrame in place.
    """
    if not invalid.any():
        return
    pre_valid = di['valid'].copy()
    pre = pre_valid.sum()
    di.loc[invalid, 'valid'] = False
    di.loc[pre_valid & invalid, 'reason'] = reason
    post = di['valid'].sum()
    if post < pre:
        print(f"Invalidated {pre - post} records: {reason}. {post} records remain.")


def main(input_index: Path, output_index: Path):
    """Apply heuristics to impute BIDS entities from DICOM attributes and pick images for a specific project.

    Args:
        input_index: Path to the input DICOM index file (parquet format).
        output_index: Path to save the output DICOM index file (parquet format).
    
    Returns:
        None. Saves the processed DICOM index to the specified output path.
    """
    print("Loading data")
    di = pd.read_parquet(input_index)
    print(f"Loaded {len(di)} records from {input_index}.")
    for column_to_drop in ['warnings', 'error']:
        if column_to_drop in di.columns:
            di = di.drop(columns=[column_to_drop])

    print("Preliminary validity checks ...")
    di['valid'] = True
    di['reason'] = "Valid"

    # Only include modality MR
    _invalidate(di, di['Modality'] != 'MR', 'Not MR.')

    # Exclude images with blank SeriesDescription
    _invalidate(di, di['SeriesDescription'].isnull(), "No SeriesDescription.")

    print(f"Found {di['valid'].sum()} valid records.")

    print('Applying heuristics to categorize images.')
    # Make a string column based on B value so it can be processed by a heuristic
    di['diffusion_weighted'] = 'False'
    di.loc[di['DiffusionBValue'] > 0, 'diffusion_weighted'] = 'True'
    heuristics = [
        ({'keyword': 'SeriesDescription', 'regex': r't1'},
        [{'entity': 'suffix', 'value': 'T1w'}]),

        ({'keyword': 'SeriesDescription', 'regex': r't1map'},
        [{'entity': 'suffix', 'value': 'T1map'}]),

        ({'keyword': 'SeriesDescription', 'regex': r'flair|dark'},
        [{'entity': 'suffix', 'value': 'FLAIR'}]),

        ({'keyword': 'SeriesDescription', 'regex': r't2.*tirm'},
        [{'entity': 'suffix', 'value': 'FLAIR'}]),

        ({'keyword': 'SeriesDescription', 'regex': r'dwi|dti|diff|resolve|trace'},
        [{'entity': 'suffix', 'value': 'dwi'}]),

        ({'keyword': 'diffusion_weighted', 'regex': r'True'},
        [{'entity': 'suffix', 'value': 'dwi'}]),

        ({'keyword': 'SeriesDescription', 'regex': r'tracew'},
        [{'entity': 'suffix', 'value': 'dwi'},
        {'entity': 'reconstruction', 'value': 'TRACEW'}]),

        ({'keyword': 'SeriesDescription', 'regex': r'adc|apparent|average.dc'},
        [{'entity': 'suffix', 'value': 'dwi'},
        {'entity': 'reconstruction', 'value': 'ADC'}]),

        ({'keyword': 'SeriesDescription', 'regex': r'fractional'},
        [{'entity': 'suffix', 'value': 'dwi'},
        {'entity': 'reconstruction', 'value': 'FA'}]),

        ({'keyword': 'ContrastBolusAgent', 'regex': r'^(?!\s*$|NONE$).+'},
        [{'entity': 'contrast_enhancement', 'value': 'yes'}]),

        ({'keyword': 'ContrastBolusRoute', 'regex': r'^(?!\s*$|NONE$).+'},
        [{'entity': 'contrast_enhancement', 'value': 'yes'}]),
    ]
    apply_heuristics(di, heuristics)

    # Apply project-specific criteria
    _invalidate(di, ~di['suffix'].isin(['T1w', 'FLAIR', 'dwi']), "Not T1, FLAIR, or DWI.")

    print("Final counts:")
    print(di['reason'].value_counts(dropna=False))

    print('Saving')
    di.to_csv(output_index.with_suffix('.csv'), index=False, quoting=csv.QUOTE_NONNUMERIC)
    di.to_parquet(output_index.with_suffix('.parquet'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply heuristics to impute BIDS entities from DICOM attributes and pick images for a specific project.")
    parser.add_argument("input_index", type=Path, help="Path to the input DICOM index file (parquet format).")
    parser.add_argument("output_index", type=Path, help="Path to save the output DICOM index file (parquet format).")
    args = parser.parse_args()
    main(args.input_index, args.output_index)
