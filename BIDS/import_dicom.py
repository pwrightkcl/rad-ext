"""Import selected DICOM series from a main dataset into a project-specific directory.
"""
import argparse
from pathlib import Path
import csv
import warnings

import pandas as pd
from tqdm import tqdm


def main (input_index: Path, project_dir: Path):
    """Import selected DICOM series from a main dataset into a BIDS project directory.

    Creates symlinks to the original DICOM files in a directory structure matching the study and series
    directories in the main dataset.

    Assumes that the indexed DICOM files are nested in at least two levels of directories, which will be
    treated as the study and series directories.
    
    For example:

    If the `dicom_filepath` column contains:
    /path/to/dataset/sourcedata/dicom/x123456/0001/image.dcm

    Then the `output_path` column will be set to:
    {project_dir}/sourcedata/dicom/x123456/0001

    Creates a new DICOM index `{project_dir}/metadata/dicom_index_imported` (CSV and parquet) containing only the records
    for the imported DICOM series, with new columns:
    
    - original_dicom_filepath: the .dcm file indexed in the source dataset, renamed from `dicom_filepath`
    - dicom_path: the path to the directory where the DICOM files will be imported (e.g. {output_root}/x123456/0001)
    - import_status: whether the directory was imported:
      - "exists": the output directory already exists, so no import needed
      - "to_ln": the output directory does not exist, but the parent directory does, so a symlink will be created
      - "to_mkdir_ln": the output directory and its parent directory do not exist, so parent and symlink will be created

    Generates a shell script `{project_dir}/code/sourcedata/import_commands.sh` containing the necessary `mkdir` and `ln` commands to perform the import.

    Args:
        input_index: Path to the input DICOM index file (parquet format); must have columns `dicom_filepath` and `valid`.
        project_dir: Root directory for the BIDS project.
    """
    di = pd.read_parquet(input_index)
    di = di.query('valid').copy()
    print(f"Processing {di.shape[0]} records from {input_index}.")

    di['dicom_filepath'] = di['dicom_filepath'].map(lambda x: Path(x).resolve())
    di['import_path'] = di['dicom_filepath'].map(lambda x: x.parent)
    di['series_dir'] = di['dicom_filepath'].map(lambda x: x.parent.name)
    di['study_dir'] = di['dicom_filepath'].map(lambda x: x.parent.parent.name)
    if di['study_dir'].eq('').any():
        raise ValueError("DICOM filepath must contain at least two parent directories.")
    di['output_path'] = di.apply(lambda row: project_dir / 'sourcedata' / 'dicom' / row['study_dir'] / row['series_dir'], axis=1)
    di['import_status'] = ''

    commands_mkdir = []
    commands_ln = []
    for index, row in tqdm(di.iterrows(), desc="Checking DICOM paths", total=di.shape[0]):
        import_path = row['import_path']
        output_path = row['output_path']
        output_parent = output_path.parent
        if output_parent.exists():
            if output_path.exists():
                di.loc[index, 'status'] = 'exists'
            else:
                di.loc[index, 'status'] = 'to_ln'
                commands_ln.append(f"ln -s {import_path} {output_path}\n")
        else:
            di.loc[index, 'status'] = 'to_mkdir_ln'
            commands_mkdir.append(f"mkdir -p {output_parent}\n")
            commands_ln.append(f"ln -s {import_path} {output_path}\n")

    # Remove duplicates from commands_mkdir.
    commands_mkdir = list(set(commands_mkdir))

    # Check there are no duplicates in commands_ln.
    if len(commands_ln) != len(set(commands_ln)):
        warnings.warn("Duplicate `ln` commands generated. Does you DICOM index contain multiple series per directory?")

    # Write commands to file.
    code_dir = project_dir / 'code' / 'sourcedata'
    if not code_dir.exists():
        code_dir.mkdir(parents=True)
    import_commands = code_dir / 'import_commands.sh'
    with import_commands.open('w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Import DICOMs\n")
        f.write("# mkdir\n")
        f.writelines(commands_mkdir)
        f.write("# ln\n")
        f.writelines(commands_ln)

    # Tidy columns and save
    di = di.rename(columns={'dicom_filepath': 'original_dicom_filepath', 'output_path': 'dicom_path'})
    di = di.drop(columns=['valid', 'reason', 'import_path', 'series_dir', 'study_dir'])
    di['original_dicom_filepath'] = di['original_dicom_filepath'].astype(str)
    di['dicom_path'] = di['dicom_path'].astype(str)
    output_index = project_dir / 'metadata' / 'dicom_index_imported'
    di.to_csv(output_index.with_suffix('.csv'), index=False, quoting=csv.QUOTE_NONNUMERIC)
    di.to_parquet(output_index.with_suffix('.parquet'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import selected DICOM series from a main dataset into a project-specific directory.")
    parser.add_argument("input_index", type=Path, help="Path to the input DICOM index file (parquet format).")
    parser.add_argument("project_dir", type=Path, help="Root directory for the BIDS project.")
    args = parser.parse_args()
    main(args.input_index, args.project_dir)