# Radiology metadata

## Index DICOM metadata

This script will read metadata from a directory tree containing DICOM (`.dcm`) files and collect specified attributes into a Pandas DataFrame, saved as CSV and parquet (or pickle if saving as parquet raises an exception).

It has two modes:

* Index by file
  * Walks the input directory and indexes every file ending `.dcm`.
  * This is mostly applicable to DICOM responses returned by C-FIND queries, where a single `.dcm` file can represent a constrained set of attributes at the study and series level.
* Index by directory
  * Walks the input directory and indexes one `.dcm` file per subdirectory
  * This is mostly applicable to DICOM image files, which we assume are stored in subdirectories containing images from the same series, and we are only interested in attributes at the series level or above.

Arguments:

* `--level`: "file" or "dir" for file or directory mode
* `--input_dir`: root of the directory tree containing DICOM files
* `--output_dir`: directory where the CSV and parquet files will be saved
* `--chunk_size` [optional]: save a DataFrame each time the specified number of files are processed, then combine DataFrames at the end
* `--attributes` [optional]: DICOM attributes to index: one or more DICOM keywords, path to text file containing DICOM keywords, or `'*'`; default is a small set of basic attributes.
* `--overwrite`: if set, overwrite existing output files, otherwise try to resume.
* `--max_columns`: Maximum number of columns allowed in output tables after flattening DICOM metadata.

Outputs:

* `dicom_index.csv`
* `dicom_index.parquet` (or `.pkl` if parquet fails)

### Chunking and resuming

Setting a chunk size will make the script save the indexed data to parquet each time the chunk size is reached. This prevents the dataset from growing too large in memory and slowing the indexing process and it allows resuming if the script is interrupted.

When all chunks have been indexed, the script will reload the saved chunks and concatenate them into the final output files.

If the chunk size is not smaller than the number of files found, the script will run in a single pass.

If `--overwrite` is not set, the script will check for existing files and attempt to resume:

* If indexing in a single pass and the output files exist, exit with a "nothing to do" message.
* If indexing by chunk:
  * If all chunks are present:
    * If final outputs are present, exit with a "nothing to do message".
    * Otherwise, concatenate the chunks, save the final outputs, and finish.
  * If a partial set of chunks are present:
    * Resume from the next chunk
    * Raise an error if:
      * The number of existing chunks is larger than expected
      * The chunks are not in order (e.g. 5 chunk files are found, but are not numbered 0-4)

### Data normalisation

The script verifies that incoming data conform to the expected DICOM Value Representation (VR) and Value Multiplicity (VM). If the data types vary within a column, this can raise errors from pyarrow when attempting to save to parquet. If saving to parquet fails, the script will save to pickle.

#### Scalar attributes (VM=1)

Scalar attributes are forced to be scalars:

* Empty or missing values are coerced to `""` for string VR or `None` for numeric VR
* For list-like values:
  * Empty lists are treated like empty or missing values
  * For lists of length 1, the first value is used
  * For longer lists, the value is coerced to string. For numeric columns, this may cause a pyarrow error, but will avoid data loss and allows introspection of the problematic data in the pickle file.

#### Multi-valued attributes (VM > 1)

Multi-valued attributes are coerced to `list`:

* List-like values are cast to `list` because pyarrow does not recognise the pydicom `MultiValue` type.
* Empty scalars are coerced to the empty list `[]`.
* Scalar strings will be split on the delimiters `\` or `/`.
* Other scalars will be stored as a singleton list.

#### Type conversion

Values are converted to Python serialisable types:

* `PersonName` and `UID` -> `str`
* `DSfloat` -> `float`
* `IS` -> `int`
* `bytes` and `bytearray` -> `str`
* String values have Unicode null characters and surrounding whitespace removed

### Sequence flattening and column limits

DICOM Sequence attributes are flattened for storage in DataFrame format using dot indexing.

For example:

```text
OtherPatientIDsSequence
├── 0
│   ├── PatientID
│   ├── IssuerOfPatientID
│   └── TypeOfPatientID
└── 1
    ├── PatientID
    ├── IssuerOfPatientID
    └── TypeOfPatientID  
```

Becomes:

```text
OtherPatientIDsSequence.0.PatientID
...
OtherPatientIDsSequence.1.TypeOfPatientID
```

If indexing all attributes using `--attributes "*"`, the number of columns can grow very large. The maximum number of columns is limited to 256 by default. The `"*"` option is only recommended when indexing DICOM files representing C-FIND query responses, since these are constrained by the attributes specified in the initial query. For image files, flattening all attributes could result in thousands of columns. Use `--max_columns` to adjust the limit if needed.
