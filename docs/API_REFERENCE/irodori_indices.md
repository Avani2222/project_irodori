# API Reference – Vegetation and Water Indices Module

This module contains functions to compute common vegetation, water, and soil indices from hyperspectral data. Each function can optionally visualize the index as a heatmap.

---

### 1. `compute_ndvi`
**Formula:** $NDVI = (NIR - RED) / (NIR + RED)$
**Description:** This index is used to measure **vegetation health and density**. It compares the near-infrared (**NIR**) light that plants strongly reflect to the red light they absorb. Healthy vegetation shows high NDVI values. 🌳
**Use Case:** Detects vegetation health and density. NDVI is widely used in agriculture, forestry, and environmental monitoring.
**Returns:** `np.ndarray` of NDVI values per pixel.

***

### 2. `compute_ndwi`
**Formula:** $NDWI = (GREEN - NIR) / (GREEN + NIR)$
**Description:** NDWI is designed to **detect and monitor water bodies**. It uses the difference between green and near-infrared light to highlight water features, as water strongly absorbs NIR. 💧
**Use Case:** Highlights water features and monitors surface water bodies.
**Returns:** `np.ndarray` of NDWI values per pixel.

***

### 3. `compute_savi`
**Formula:** $SAVI = ((NIR - RED) / (NIR + RED + L)) * (1 + L)$
**Description:** SAVI is an improvement on NDVI for areas with **sparse vegetation or exposed soil**. The 'L' factor helps to reduce the influence of soil brightness, providing a more accurate measure of vegetation in these conditions. 🌾
**Use Case:** Reduces soil brightness influence in areas with sparse vegetation.
**Returns:** `np.ndarray` of SAVI values per pixel.

***

### 4. `compute_custom_index`
**Formula:** (User-defined `formula` string)
**Description:** This is a versatile function that allows you to calculate **any custom spectral index** by providing a formula and a mapping of band names to their respective wavelengths. It's ideal for research and experimental purposes. 🔬
**Use Case:** Flexible computation for research-specific or experimental indices.
**Returns:** `np.ndarray` of computed index values.

***

### 5. `compute_evi`
**Formula:** $EVI = G * (NIR - RED) / (NIR + C1*RED - C2*BLUE + L)$
**Description:** EVI is an enhanced vegetation index that is more sensitive than NDVI in areas with a **high density of vegetation**. It also works to reduce the influence of atmospheric particles and soil background noise. 🌿
**Use Case:** Provides improved sensitivity in high biomass areas and reduces atmospheric influences.
**Returns:** `np.ndarray` of EVI values per pixel.

***

### 6. `compute_gndvi`
**Formula:** $GNDVI = (NIR - GREEN) / (NIR + GREEN)$
**Description:** Similar to NDVI, GNDVI is used to assess vegetation, but it is specifically **sensitive to chlorophyll content**. It uses the green wavelength instead of red, making it effective for monitoring plant stress and nutrient levels. 🧪
**Use Case:** Sensitive to chlorophyll content and plant health.
**Returns:** `np.ndarray` of GNDVI values per pixel.

***

### 7. `compute_arvi`
**Formula:** $ARVI = (NIR - (RED - gamma*(BLUE - RED))) / (NIR + (RED - gamma*(BLUE - RED)))$
**Description:** ARVI is a specialized index designed to be **resistant to atmospheric effects** like haze and aerosols. It uses a correction factor involving the blue wavelength to improve accuracy in hazy conditions. 🌬️
**Use Case:** Minimizes atmospheric effects, useful in areas with high aerosol or haze.
**Returns:** `np.ndarray` of ARVI values per pixel.

***

### 8. `compute_mndwi`
**Formula:** $MNDWI = (GREEN - SWIR) / (GREEN + SWIR)$
**Description:** This is a modification of NDWI that uses the short-wave infrared (**SWIR**) band instead of NIR. The SWIR band is particularly good at **suppressing signals from built-up areas**, making MNDWI more effective at highlighting open water. 🏞️
**Use Case:** Enhances water features while suppressing built-up land signals.
**Returns:** `np.ndarray` of MNDWI values per pixel.

***

### 9. `compute_ndsi`
**Formula:** $NDSI = (SWIR1 - SWIR2) / (SWIR1 + SWIR2)$
**Description:** NDSI is a specialized index that primarily **highlights soil features**. In certain contexts, it can also be used to detect snow and ice due to their unique reflectance properties in the SWIR wavelengths. 🏔️
**Use Case:** Highlights soil features and can detect snow or bare ground in certain wavelengths.
**Returns:** `np.ndarray` of NDSI values per pixel.
