#!/bin/bash
# Generate a shell script to convert all the DICOM files in a BIDS project directory to NIfTI using dcm2niix.
usage=$(cat << EOF
Usage: $0 <project_dir>

Generate a shell script to convert all the DICOM files in a BIDS project directory to NIfTI using dcm2niix.

The project directory should contain a 'sourcedata/dicom' subdirectory with DICOM files organized in subdirectories
for each study and series. The generated shell script will create corresponding subdirectories in 'sourcedata/nifti'
and run dcm2niix to convert the DICOM files to NIfTI format.

Arguments:
  project_dir: Root directory of the BIDS project, containing a 'sourcedata/dicom' subdirectory with DICOM files.
EOF
)

if [ "$#" -ne 1 ]; then
    echo "$usage"
    exit 1
fi
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "$usage"
    exit 0
fi
project_dir=$1
dicom_root="${project_dir}"/sourcedata/dicom
if [ ! -d "$dicom_root" ]; then
    echo "Error: DICOM root directory not found: $dicom_root"
    exit 1
fi
nifti_root="${project_dir}"/sourcedata/nifti
if [ ! -d "$nifti_root" ]; then
    echo "NIfTI root directory not found, creating: $nifti_root"
    mkdir -p "$nifti_root"
fi
commands_dir="${project_dir}"/code/sourcedata/generated_commands
if [ ! -d "$commands_dir" ]; then
    echo "Generated scripts directory not found, creating: $commands_dir"
    mkdir -p "$commands_dir"
fi
commands="${commands_dir}"/convert_commands.sh
echo "#dcm2niix commands" > "$commands"
for study_dir in "${dicom_root}"/*
do
  if [ ! -d "$study_dir" ]; then
    continue
  fi
  study=$(basename "$study_dir")
  for series_dir in "${study_dir}"/*
  do
    if [ ! -d "$series_dir" ]; then
      continue
    fi
    if [[ -z "$(find "$series_dir" -maxdepth 1 -type f -iname "*.dcm" -print -quit)" ]];
    then
      continue
    fi
    series=$(basename "${series_dir}")
    echo -n "$study $series "
    nifti_dir=${nifti_root}/${study}/${series}
    if [ -d "$nifti_dir" ]
    then
      echo "EXISTS"
    else
      echo -n "mkdir -p $nifti_dir && " >> "$commands"
      echo "dcm2niix -o $nifti_dir -f ses-${study}_run-${series}_%d -z y $series_dir" >> "$commands"
      echo "command written."
    fi
  done
done
