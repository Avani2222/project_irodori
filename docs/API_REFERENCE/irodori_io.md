# API Reference: CSV <-> HyperTable Module

This module provides functions for loading hyperspectral data from a CSV file into a `HyperTable` object and saving a `HyperTable` back to CSV format.

---

## 1. `load_csv`

**Signature:**
```python
load_csv(
    filepath: str,
    wavelengths: Optional[Union[List[float], np.ndarray]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    header: bool = True
) -> HyperTable
```
**Description:**
Loads hyperspectral data from a CSV file into a HyperTable object. Assumes the first column contains labels and the remaining columns are spectral band values.
Parameters:

**filepath (str):** Path to the CSV file. Rows = samples, Columns = [label | spectral bands].
wavelengths (list or np.ndarray, optional): Explicit list of wavelengths. If None, column names (excluding the first column) are used.
metadata (dict, optional): Additional metadata (e.g., file source, sensor type).
header (bool, default=True): Whether the CSV file has a header row. If False, columns are auto-numbered.
**Returns:**
HyperTable — A HyperTable object containing the loaded data.
**Use Case:**
Easily import CSV spectral data into a structured HyperTable for further analysis.

---

## 2. save_csv
**Signature:**
```python
save_csv(
    hyper_table: HyperTable,
    filepath: str,
    include_header: bool = True
)
```
**Description:**
Saves a HyperTable object to a CSV file. The first column contains labels, followed by spectral bands.
**Parameters:**

hyper_table (HyperTable): The HyperTable object to be saved.
filepath (str): Path to save the CSV file.
include_header (bool, default=True): Whether to include column names (wavelengths) in the CSV.
**Returns:**
None
**Use Case:**
Export HyperTable data to CSV for sharing, storage, or use in other tools.
