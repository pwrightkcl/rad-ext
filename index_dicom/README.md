# Radiology metadata

## Index DICOM metadata

This script will read metadata from a directory full of DICOM (`.dcm`) files and collect specified attributes into a Pandas DataFrame, saved as CSV and parquet (or pickle if saving as parquet raises an exception). It has two modes:

* Index by file
  * Walks the input directory and indexes every file ending `.dcm`.
  * This is mostly applicable to DICOM responses returned by C-FIND queries, which typically query a limited number of attributes
* Index by directory
  * Walks the input directory and indexes one `.dcm` file per subdirectory
  * This is mostly applicable to DICOM image files, which we assume are stored in subdirectories containing images from the same series, and we are only interested in attributes at the series level or above.

Arguments:

* `--level`: "file" or "dir" for file or directory mode
* `--in_dir`: roote of the directory tree containing DICOM files
* `--out_dir`: directory where the CSV and parquet files will be saved
* `--chunk_size` [optional]: save the DataFrame after processing the specified number of files
  * This makes it easier to resume if indexing is interrupted.
  * It also avoids the script slowing down as the DataFrame grows.
* `--attributes` [optional]: which DICOM attributes to index, specified as one of:
  * one or more DICOM keywords
  * path to a text file containing DICOM keywords
  * `"*"` to index all available attributes
  * if unspecified, index a small default set of attributes
* `--overwrite`: if set, overwrite existing output files / chunks in `output_dir`; in unset, `chunk_size` is set, and an incomplete set of chunks is in `output_dir`, attempt to resume.
* `--max_columns`: Maximum number of columns allowed in output tables after flattening DICOM metadata.
