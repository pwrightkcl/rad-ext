#!/bin/bash
# TODO: write to accept input arguments instead of hardcoding paths
root_dir=/nfs/project/WellcomeHDN/kch-sequence-classifier
commands=${root_dir}/code/sourcedata/convert_commands.sh
echo "#dcm2niix commands" > $commands
for subject_dir in ${root_dir}/sourcedata/dicom/???????
do
  subject=$(basename $subject_dir)
  for session_dir in ${subject_dir}/????????
  do
    session=$(basename $session_dir)
    for series_dir in ${session_dir}/*
    do
      series=$(basename ${series_dir})
      echo -n "$subject $session $series "
      nifti_dir=${root_dir}/sourcedata/nifti/${subject}/${session}/${series}
      if [ -d $nifti_dir ]
      then
        echo "EXISTS"
      else
        echo -n "mkdir -p $nifti_dir && " >> $commands
        echo "dcm2niix -o $nifti_dir -f sub-${subject}_ses-${session}_run-${series}_ -z n $series_dir" >> $commands
        echo "command written."
      fi
    done
  done
done
