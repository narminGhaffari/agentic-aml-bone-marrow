# Input Schema

The full AML pathology pipeline expects two metadata tables.

## Clinical Table

Accepted formats: `.csv`, `.xlsx`

Required columns:

- `PATIENT`: patient or case identifier.
- `WHO_CLASSE_SIMPLE`: class label for AML-versus-healthy classification. Expected values are `AML` and `HEALTHY`.

Optional columns:

- `NPM1`: mutation label for NPM1 prediction. Expected values are `1` for mutated and `0` for wild-type. Missing values are excluded from NPM1 evaluation.

## Slide Table

Accepted formats: `.csv`

Required columns:

- `PATIENT`: patient or case identifier matching the clinical table.
- `FILENAME`: WSI or feature filename.

## Private Data Mounts

For full reproduction, protected files should be mounted as:

```text
/data/private/metadata/clinical.csv
/data/private/metadata/slides.csv
/data/private/wsi/<FILENAME>
```

The public repository does not include these files.

## Expected Outputs

The validation script writes:

```text
/results/input_validation_summary.json
```

Full manuscript reproduction additionally requires the private WSI data, trained STAMP checkpoints,
feature extractor dependencies, and VLM/API configuration.
