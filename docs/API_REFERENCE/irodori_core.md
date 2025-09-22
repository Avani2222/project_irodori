# API Reference – HyperTable Class

This document describes the available functions and properties of the `HyperTable` class for managing hyperspectral data in tabular format.

---

## 1. Initialization

### `HyperTable(data: pd.DataFrame, wavelengths: Optional[Union[List[float], np.ndarray]] = None, metadata: Optional[Dict[str, Any]] = None)`

- **Description:**  
  Initialize a `HyperTable` object. Each row represents a sample/pixel, and each column represents a spectral band. Optionally includes labels, wavelengths, and metadata.

- **Parameters:**  
  - `data (pd.DataFrame)`: Hyperspectral data (rows = samples, columns = bands). First column can be labels if wavelengths are provided for remaining columns.  
  - `wavelengths (list or np.ndarray, optional)`: Wavelengths corresponding to columns (excluding labels).  
  - `metadata (dict, optional)`: Extra metadata about the dataset.  

- **Raises:**  
  - `ValueError` if `data` is empty or not a DataFrame, or if wavelengths length does not match number of columns.  

- **Returns:**  
  - `HyperTable` instance  

- **Use case:**  
  Store and manage hyperspectral data with optional labels, wavelengths, and metadata.

---

## 2. Properties

### `shape`

- **Description:** Returns the shape of the table (samples, bands).  
- **Returns:** `tuple` `(n_samples, n_bands)`  

### `samples`

- **Description:** Returns the number of samples (rows).  
- **Returns:** `int`  

### `bands`

- **Description:** Returns the number of spectral bands (columns).  
- **Returns:** `int`  

---

## 3. Data Access

### `get_pixel(index: int)`

- **Description:** Get the spectral signature of a single sample/pixel.  
- **Parameters:**  
  - `index (int)`: Row index of the sample.  
- **Returns:**  
  - `np.ndarray`: Array of spectral values for that pixel.  
- **Use case:**  
  Extract spectral data for a specific sample for analysis or visualization.

### `get_band(band_index: int)`

- **Description:** Get all sample values for a specific spectral band.  
- **Parameters:**  
  - `band_index (int)`: Column index of the spectral band.  
- **Returns:**  
  - `np.ndarray`: Array of values for the selected band.  
- **Use case:**  
  Analyze a single band across all samples (e.g., vegetation index computation).

---

## 4. Wavelength Management

### `set_wavelengths(start: float, end: float)`

- **Description:** Generate and assign evenly spaced wavelengths between start and end values.  
- **Parameters:**  
  - `start (float)`: Starting wavelength (e.g., 400 nm).  
  - `end (float)`: Ending wavelength (e.g., 1000 nm).  
- **Returns:** `None`  
- **Use case:** Assign wavelengths when not provided, ensuring proper band referencing.

---

## 5. Summary

### `summary()`

- **Description:** Generate a statistical summary of the dataset.  
- **Returns:**  
  - `pd.DataFrame`: Contains count, mean, std, min, quartiles, max for each band.  
- **Use case:** Quickly inspect data distribution and detect anomalies.

---

## 6. Representation

### `__repr__()`

- **Description:** Provides a developer-friendly string representation of the object.  
- **Returns:** `str`  
- **Example Output:**  
- **Use case:** Quickly inspect metadata, number of samples, bands, and wavelength range.

---

# Notes

- The `spectra` attribute stores the underlying NumPy array of spectral values for fast computation: `self.spectra = self.data.values`.  
- If no labels are provided, `self.labels` is `None`.  
- Wavelengths may be `None` if column headers are non-numeric.
