# Python 3.9 Compatibility Fixes

## Summary
Updated all code to be compatible with Python 3.9 (instead of Python 3.10+) and fixed file paths for the current project structure.

## Files Modified

### 1. **shelter_flow.py**
**Issue**: Used Python 3.10+ union type syntax (`|`) in type hints
- Line 120: `local_dir: str | Path` → `local_dir: Union[str, Path]`
- Line 148: `local_path: str | Path | None` → `local_path: Optional[Union[str, Path]]`
- Line 124: `files: list[Path]` → `files: list`

**Fix**: Added `from typing import Union, Optional` and replaced all union operators with compatible types.

### 2. **sna_pipeline_sasm.py**
**Issue**: File paths were relative without directory prefix
- `LOCAL_FILES` dict had paths like `"2013-street-needs-assessment-results.xlsx"`
- Should be: `"source_data/2013-street-needs-assessment-results.xlsx"`

**Fix**: Updated all three entries in `LOCAL_FILES` (lines 63-65) to include `source_data/` prefix.

### 3. **compare_pipelines.py**
**Issues**:
1. Used Python 3.10+ union type syntax (`|`) in function signature
2. Old pipeline file paths were relative (should use `synthetic_data/` subdirectory)
3. Function tried to access `gender` and `race` columns that don't exist in old pipeline

**Fixes**:
- Added `from typing import Optional` import (line 42)
- Updated file paths (lines 52-61):
  - `OLD_INDIVIDUALS = "synthetic_data/synthetic_individuals.csv"`
  - `OLD_REGION_YEAR = "synthetic_data/region_year_features.csv"`
  - `OLD_FORECAST = "synthetic_data/forecast_results.csv"`
- Changed type hint (line 100): `new_quality: pd.DataFrame | None` → `new_quality: Optional[pd.DataFrame]`
- Updated `compare_distributions()` function to gracefully handle missing columns:
  - Added conditional checks for `gender` and `race` columns
  - Prints message if columns not available in old pipeline
  - Avoids KeyError exceptions

### 4. **sasm_generator.py**
✅ No changes needed - already compatible with Python 3.9

## Python 3.9 vs 3.10+ Changes

| Feature | Python 3.9 | Python 3.10+ |
|---------|-----------|------------|
| Union types | `Union[A, B]` | `A \| B` |
| Optional types | `Optional[A]` | `A \| None` |
| List generics | `list` | `list[A]` |
| Import | `from typing import Union, Optional` | Built-in |

## File Path Structure
```
homeless-synthetic-data/
├── source_data/
│   ├── 2013-street-needs-assessment-results.xlsx
│   ├── 2018-street-needs-assessment-results.xlsx
│   ├── 2021-street-needs-assessment-results.xlsx
│   └── toronto-shelter-system-flow.csv
├── synthetic_data/
│   ├── synthetic_individuals.csv      (old pipeline output)
│   ├── region_year_features.csv       (old pipeline output)
│   └── forecast_results.csv           (old pipeline output)
├── sasm_synthetic_individuals.csv     (new SASM pipeline output)
├── sasm_region_year_features.csv      (new SASM pipeline output)
├── sasm_forecast_results.csv          (new SASM pipeline output)
└── sasm_quality_log.csv               (new SASM pipeline output)
```

## Testing
All files now pass Python compilation check:
```bash
python -m py_compile sna_pipeline_sasm.py sasm_generator.py shelter_flow.py compare_pipelines.py
```

## Usage
```bash
# Run old pipeline
python sna_pipeline.py --local --local-flow source_data/toronto-shelter-system-flow.csv

# Run new SASM pipeline
python sna_pipeline_sasm.py --local --local-flow source_data/toronto-shelter-system-flow.csv

# Compare both pipelines
python compare_pipelines.py
```
