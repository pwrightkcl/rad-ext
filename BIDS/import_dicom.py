"""Import selected DICOM series from a main dataset into a project-specific directory.
"""
import argparse
from pathlib import Path
import csv
import warnings

import pandas as pd
from tqdm import tqdm


def main (input_index: Path,  output_index: Path, output_root: Path, import_commands: Path):
    """Import selected DICOM series from a main dataset into a project-specific directory.

    Creates symlinks to the original DICOM files in a directory structure matching the study and series
    directories in the main dataset.

    Assumes that the indexed DICOM files are nested in at least two levels of directories, which will be
    treated as the study and series directories. For example:

    If the 'dicom_filepath' column contains:
    /path/to/dataset/sourcedata/dicom/x123456/0001/image.dcm

    Then the 'output_path' column will be set to:
    {output_root}/x123456/0001

    Args:
        input_index: Path to the input DICOM index file (parquet format).
        output_index: Path to save the output DICOM index file (parquet format).
        output_root: Root directory for the output DICOM dataset.
        import_commands: Path to save the shell script with import commands.
    """
    if not (output_index).parent.exists():
        output_index.parent.mkdir(parents=True)
    if not import_commands.parent.exists():
        import_commands.parent.mkdir(parents=True)

    di = pd.read_parquet(input_index)
    di = di.query('valid').copy()
    print(f"Processing {di.shape[0]} records.")

    di['dicom_filepath'] = di['dicom_filepath'].map(Path)
    di['import_path'] = di['dicom_filepath'].map(lambda x: x.parent)
    di['series_dir'] = di['dicom_filepath'].map(lambda x: x.parent.name)
    di['study_dir'] = di['dicom_filepath'].map(lambda x: x.parent.parent.name)
    if di['study_dir'].eq('').any():
        raise ValueError("DICOM filepath must contain at least two parent directories.")
    di['output_path'] = di.apply(lambda row: output_root / row['study_dir'] / row['series_dir'], axis=1)
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
        warnings.warn(f"There will be duplicate `ln` commands in {import_commands}.")

    # Write commands to file.
    with import_commands.open('w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Import DICOMs\n")
        f.write("# mkdir\n")
        f.writelines(commands_mkdir)
        f.write("# ln\n")
        f.writelines(commands_ln)

    # Convert Path objects to str to save to parquet
    di['dicom_filepath'] = di['dicom_filepath'].astype(str)
    di['import_path'] = di['import_path'].astype(str)
    di['output_path'] = di['output_path'].astype(str)
    di.to_parquet(output_index.with_suffix('.parquet'))
    di.to_csv(output_index.with_suffix('.csv'), index=False, quoting=csv.QUOTE_NONNUMERIC)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import selected DICOM series from a main dataset into a project-specific directory.")
    parser.add_argument("input_index", type=Path, help="Path to the input DICOM index file (parquet format).")
    parser.add_argument("output_index", type=Path, help="Path to save the output DICOM index file (parquet format).")
    parser.add_argument("output_root", type=Path, help="Root directory for the output DICOM dataset.")
    parser.add_argument("import_commands", type=Path, help="Path to save the shell script with import commands.")
    args = parser.parse_args()
    main(args.input_index, args.output_index, args.output_root, args.import_commands)