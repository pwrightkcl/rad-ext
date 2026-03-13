import re
import warnings
from typing import Annotated

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, RootModel


class FindRule(BaseModel):
    """A find rule specifying a DICOM keyword and a regex to match against it."""

    model_config = ConfigDict(extra='forbid')
    keyword: str
    regex: str


class WriteRule(BaseModel):
    """A write rule specifying a BIDS entity and value to assign on match."""

    model_config = ConfigDict(extra='forbid')
    entity: str
    value: str


_WriteRuleList = Annotated[list[WriteRule], Field(min_length=1)]
_Heuristic = tuple[FindRule, _WriteRuleList]
_HeuristicList = RootModel[Annotated[list[_Heuristic], Field(min_length=1)]]


def apply_heuristics(di: pd.DataFrame, heuristics: list[tuple[dict, list[dict]]] | None = None) -> None:
    """Apply heuristics to impute BIDS entities from DICOM attributes.

    The heuristics argument must be a list of 2-tuples. Each 2-tuple comprises a find rule and a write list,
    validated against the FindRule and WriteRule schemas respectively.

    The find rule must have keys:
        *   'keyword': a DICOM keyword column in di; if not present, the heuristic is skipped with a warning.
        *   'regex': regular expression to match against the column

    The write list must be a non-empty list of dicts with keys:
        *   'entity': a BIDS entity column, to be added to the dataframe and populated for matching records.
        *   'value': a value to assign to that column for records matched by the find rule

    **Examples.** First, if SeriesDescription contains 'flair' or 'dark', set 'suffix' to 'FLAIR' . Second, if
    SeriesDescription contains adc, apparent, or average.dc (where dot is any character) then set suffix to 'dwi' and
    'reconstruction' to 'ADC'.

    [
     ({'keyword': 'SeriesDescription', 'regex': r'flair|dark'},
      [{'entity': 'suffix', 'value': 'FLAIR'}]),
     ({'keyword': 'SeriesDescription', 'regex': r'adc|apparent|average.dc'},
      [{'entity': 'suffix', 'value': 'dwi'},
       {'entity': 'reconstruction', 'value': 'ADC'}]),
    ]

    Args:
        di : Pandas DataFrame containing all the columns searched for by the heuristics. Normally these should be DICOM
             keywords like 'SeriesDescription'.
        heuristics : A list of heuristics to search on DICOM keywords and impute BIDS entities to matches.

    Returns:
        None. Modifies the input DataFrame in place.
    """
    if heuristics is None:
        heuristics = [
            ({'keyword': 'SeriesDescription', 'regex': r't1'},
             [{'entity': 'suffix', 'value': 'T1w'}]),

            ({'keyword': 'SeriesDescription', 'regex': r'mp-?rage'},
             [{'entity': 'suffix', 'value': 'T1w'},
              {'entity': 'acquisition', 'value': 'MPRAGE'}]),

            ({'keyword': 'SeriesDescription', 'regex': r't1map'},
             [{'entity': 'suffix', 'value': 'T1map'}]),

            ({'keyword': 'SeriesDescription', 'regex': r't2'},
             [{'entity': 'suffix', 'value': 'T2w'}]),

            ({'keyword': 'SeriesDescription', 'regex': r't2\*'},
             [{'entity': 'suffix', 'value': 'T2star'}]),

            ({'keyword': 'SeriesDescription', 'regex': r'flair|dark'},
             [{'entity': 'suffix', 'value': 'FLAIR'}]),

            ({'keyword': 'SeriesDescription', 'regex': r't2.*tirm'},
             [{'entity': 'suffix', 'value': 'FLAIR'},
              {'entity': 'acquisition', 'value': 'TIRM'}]),

            ({'keyword': 'SeriesDescription', 'regex': r'pd'},
             [{'entity': 'suffix', 'value': 'PD'}]),

            ({'keyword': 'SeriesDescription', 'regex': r'pd.*t2'},
             [{'entity': 'suffix', 'value': 'PDT2'}]),

            ({'keyword': 'SeriesDescription', 'regex': r'angio'},
             [{'entity': 'suffix', 'value': 'angio'}]),

            ({'keyword': 'AngioFlag', 'regex': r'Y'},
             [{'entity': 'suffix', 'value': 'angio'}]),

            ({'keyword': 'ContrastBolusAgent', 'regex': r'^(?!\s*$|NONE$).+'},
             [{'entity': 'contrast_enhancement', 'value': 'yes'}]),

            ({'keyword': 'SeriesDescription', 'regex': r'dwi|dti|diff|resolve|trace'},
             [{'entity': 'suffix', 'value': 'dwi'}]),

            ({'keyword': 'SeriesDescription', 'regex': r'tracew'},
             [{'entity': 'suffix', 'value': 'dwi'},
              {'entity': 'reconstruction', 'value': 'TRACEW'}]),

            ({'keyword': 'DiffusionWeighted', 'regex': r'True'},
             [{'entity': 'suffix', 'value': 'dwi'}]),

            ({'keyword': 'SeriesDescription', 'regex': r'adc|apparent|average.dc'},
             [{'entity': 'suffix', 'value': 'dwi'},
              {'entity': 'reconstruction', 'value': 'ADC'}]),

            ({'keyword': 'SeriesDescription', 'regex': r'fractional'},
             [{'entity': 'suffix', 'value': 'dwi'},
              {'entity': 'reconstruction', 'value': 'FA'}]),
        ]

    # Validate heuristics structure using Pydantic models.
    validated = _HeuristicList.model_validate(heuristics)

    # Collect new entity columns needed across all heuristics, then add them to the DataFrame.
    new_columns = []
    for _, write_rules in validated.root:
        for write_rule in write_rules:
            if write_rule.entity not in di.columns and write_rule.entity not in new_columns:
                new_columns.append(write_rule.entity)
    if new_columns:
        di[new_columns] = ""

    # Apply heuristics. Skip rules whose keyword is not a column in di.
    di['parsed'] = False
    for find_rule, write_rules in validated.root:
        if find_rule.keyword not in di.columns:
            warnings.warn(f"Heuristic keyword not found in DataFrame, skipping: {find_rule.keyword}.")
            continue
        match_records = di[find_rule.keyword].str.contains(find_rule.regex, regex=True, flags=re.IGNORECASE, na=False)
        if match_records.any():
            for write_rule in write_rules:
                di.loc[match_records, write_rule.entity] = write_rule.value
                di.loc[match_records, 'parsed'] = True
    print(f"Matched heuristics to {di['parsed'].sum()} records.")
    di.drop(columns='parsed', inplace=True)

    # Set unmatched records' suffix to 'unknown'.
    if 'suffix' not in di.columns:
        warnings.warn("No heuristic set the 'suffix' entity; all records assigned suffix 'unknown'.")
        di['suffix'] = 'unknown'
    else:
        no_suffix = di['suffix'].fillna("").eq("")
        if no_suffix.any():
            if no_suffix.all():
                warnings.warn("No heuristic set the 'suffix' entity; all records assigned suffix 'unknown'.")
            n_no_suffix = no_suffix.sum()
            di.loc[no_suffix, 'suffix'] = 'unknown'
            print("Assigned suffix 'unknown' to {} records.".format(n_no_suffix))
