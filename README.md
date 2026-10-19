# Data Analysis: Comprehensive Photovoltaic Device Characterization

## Overview

This repository contains comprehensive data analysis workflows for photovoltaic devices, combining four key analysis approaches:

1. **X-Ray Diffraction (XRD) Analysis**: Comparison of experimental diffraction patterns with reference data and texture-coefficient analysis
2. **UV-Visible Spectroscopy (UV-Vis)**: Absorption-coefficient and optical band-gap analysis, including preparation of absorption data for SCAPS
3. **Device Performance Modeling**: Comparison of experimental J-V (current density-voltage) measurements with SCAPS 1D numerical simulations
4. **Solar Cell Performance**: EQE and J-V curve analysis to evaluate and compare photovoltaic device performance

## 1. XRD Data Analysis

### Purpose
X-Ray Diffraction (XRD) analysis compares measured diffraction patterns with crystallographic reference data to examine the samples' crystal structure and preferred orientations. Texture coefficients quantify how strongly selected crystallographic planes are represented relative to the reference.

### Key Files
- **`XRD/Texture_Coefficient_Analysis.ipynb`**: Matches experimental diffraction peaks to the Sb₂Se₃ reference (JCPDS 15-0861), calculates texture coefficients, and generates plots and summary statistics.
- **`XRD/TC_comparison.ipynb`**: Loads XRD spectra and sample data from the second-round dataset, displays spectra against Sb₂Se₃ and MoSe₂ reference patterns, and compares texture-coefficient results across samples.

- **Data Files**:
  - `data/XRD/2 Round/`: Experimental `.ASC` spectra and corresponding Excel files used in sample comparisons.
  - `data/XRD/Training_BraggBrentano_Fast.xrdml`: Additional XRD measurement file.
  - `JCPDS-ICDD/`: Reference cards for Sb₂Se₃ (15-0861) and MoSe₂ (29-0914).

### Texture Coefficient Theory
The texture coefficient $T_c(hkl)$ quantifies the preferred orientation of crystallographic planes relative to random orientation:

$$T_c(hkl) = \frac{I(hkl)/I_0(hkl)}{(1/n)\sum_i I_i(hkl)/I_{0i}(hkl)}$$

where:
- $I(hkl)$ = measured X-ray intensity for plane (hkl)
- $I_0(hkl)$ = standard intensity for randomly oriented sample
- $n$ = number of major peaks

**Interpretation**:
- $T_c > 1$: Preferred orientation (textured)
- $T_c ≈ 1$: Random orientation
- $T_c < 1$: Suppressed orientation

### Output
- Texture coefficient values for major crystallographic planes
- Comparisons of preferred orientations between samples
- XRD spectrum, texture-coefficient, and summary plots/tables saved in `XRD/Plots/`

---

## 2. UV-Visible Spectroscopy (UV-Vis) Analysis

### Purpose
UV-Visible spectroscopy analysis uses transmittance and reflectance measurements to calculate the absorption coefficient and estimate the optical band gap ($E_g$) of semiconductor materials. The current notebook uses Tauc plots to estimate band gaps from the absorption data.

### Key Files
- **`UV_Vis_analysis.ipynb`**: Main analysis notebook for band gap determination
  - Reads wavelength, transmittance, and reflectance data from the Excel workbook
  - Calculates and plots the absorption coefficient
  - Generates combined and individual Tauc plots and extracts band gaps
  - Exports Tauc-plot data and a band-gap summary
  - Writes SCAPS-compatible absorption files and plots for the processed samples

- **Data Files**:
  - `data/UV-Vis/4_Sb2Se3_Corrected.xlsx`: Current workbook used by the notebook; other measurement workbooks are also stored in `data/UV-Vis/`.
  - Generated output files in `UV-Vis/Plots/`:
    - `Combined_Direct_Indirect_BandGap.png`: Overview plot comparing all samples
    - `Absorption_Coefficient.png`: Absorption spectra of all samples
    - `Tauc_plot_extrapolation_*.png`: Individual Tauc plots with band-gap extrapolation
    - `Tauc_plot_curves_and_extrapolation.csv` and `results_tauc_plot.xlsx`: Tauc-plot data exports
    - `Summary_Bandgaps.xlsx`: Summary table of calculated band gaps
    - `SCAPS_Absorption/`: SCAPS-compatible `.abs` files and corresponding plots

### Absorption Coefficient Theory

The absorption coefficient is calculated from transmittance and reflectance using:

$$\alpha = \frac{1}{t} \cdot \ln \left( \frac{(1-R)^2}{T} + \sqrt{\frac{(1-R)^4}{4T^2} + R^2} \right) \quad \text{(1)}$$

where:
- $\alpha$ = absorption coefficient (cm⁻¹)
- $t$ = thickness of the film (cm)
- $T$ = transmittance (fraction)
- $R$ = reflectance (fraction)

### Photon Energy Calculation

The photon energy (in eV) is calculated from wavelength using:

$$h\nu = \frac{hc}{\lambda} \cdot \frac{1}{e} \quad \text{(2)}$$

where:
- $h$ = Planck's constant (6.626 × 10⁻³⁴ J·s)
- $c$ = speed of light (3 × 10⁸ m/s)
- $\lambda$ = wavelength (nm)
- $e$ = elemental charge (1 eV)

### Tauc Plot Method

The Tauc plot uses the relation:

$$(h\nu \cdot \alpha)^{1/\gamma} = B(h\nu - E_g) \quad \text{(3)}$$

where:
- $\gamma$ = transition type (2 for indirect, 1/2 for direct)
- $B$ = proportionality constant
- $E_g$ = band gap energy (eV)

**Procedure**:
1. **Data Smoothing**: Savitzky-Golay filter preserves slopes while removing noise
2. **Linear Region Identification**: Find region with maximum derivative
3. **Extrapolation**: Fit line through linear region to x-axis intercept
4. **Band Gap Extraction**: Intercept at y=0 gives $E_g$

### Output
- Optical band-gap estimates from Tauc plots
- Absorption coefficient vs. energy plots
- SCAPS-compatible absorption files for device simulation
- Tauc-plot data exports and a summary table of calculated values

---

## 3. SCAPS 1D Simulation vs Experimental Data Comparison

### Purpose
This analysis compares experimental photovoltaic device performance (measured J-V curves) with SCAPS 1D (Solar Cell Capacitance Simulator) results. The notebooks read measurements for the CdS3_2 device and compare them with selected simulations or batches of SCAPS `.iv` simulations. This helps assess how well the simulations reproduce the measured curve and performance metrics.

### Key Files
- **`SCAPS_Simulations_for_1_plot_scaps.ipynb`**: Plots the experimental J-V curve alongside a selected SCAPS simulation and produces comparison plots and a table.
- **`SCAPS_Simulations_iv_files.ipynb`**: Reads multiple SCAPS `.iv` files, compares their J-V curves with experiment using performance-parameter scores and mean squared error (MSE), and summarizes the best matches.

- **Data Files**:
  - `data/CdS3_2/CdS3_2_IV Graph.xlsx`: Experimental J-V measurements.
  - `data/CdS3_2/CdS3_2_Results Table.xlsx`: Experimental device results.
  - `data/CdS3_2/2/Bad Best/2_5Rs_voc.xlsx`: Selected simulation workbook used by the single-simulation notebook.
  - `data/CdS3_2/3/*.iv`: SCAPS simulation files used by the batch-comparison notebook.

### J-V Curve Analysis Theory

#### Single-Diode Model
Solar cell behavior is modeled using the Shockley ideal diode equation:

$$J = J_L - J_0 \left[\exp\left(\frac{q(V + J \cdot R_s)}{nk_BT}\right) - 1\right] - \frac{V + J \cdot R_s}{R_{sh}}$$

where:
- $J$ = current density (mA/cm²)
- $J_L$ = photocurrent density
- $J_0$ = dark saturation current density
- $q$ = elementary charge (1.602 × 10⁻¹⁹ C)
- $V$ = applied voltage (V)
- $n$ = ideality factor (1-2)
- $k_B$ = Boltzmann constant
- $T$ = absolute temperature (K)
- $R_s$ = series resistance (Ω·cm²)
- $R_{sh}$ = shunt resistance (Ω·cm²)

#### Key Performance Metrics

1. **Open Circuit Voltage ($V_{oc}$)**:
   - Maximum voltage at zero current
   - Determined by: $V_{oc} = \frac{nk_BT}{q}\ln\left(\frac{J_L}{J_0} + 1\right)$

2. **Short Circuit Current Density ($J_{sc}$)**:
   - Maximum current at zero voltage
   - Proportional to incident photon flux and quantum efficiency

3. **Fill Factor ($FF$)**:
   - Ratio of maximum power to theoretical rectangular area
   - $FF = \frac{V_{mp} \cdot J_{mp}}{V_{oc} \cdot J_{sc}}$

4. **Power Conversion Efficiency ($\eta$)**:
   - Overall efficiency of solar-to-electrical conversion
   - $\eta = \frac{P_{max}}{P_{in}} = \frac{V_{oc} \cdot J_{sc} \cdot FF}{P_{in}}$

### SCAPS 1D Methodology
SCAPS models thin-film solar-cell electrical behaviour by solving coupled semiconductor transport equations:

- **Poisson Equation**: Describes electrostatic potential distribution
- **Continuity Equations**: Govern electron and hole transport with drift-diffusion

SCAPS outputs include:
- J-V curves under illumination (current density vs. voltage)
- Device performance parameters such as $V_{oc}$, $J_{sc}$, fill factor, and efficiency

### Comparison Methods

#### Method 1: Parameter-Based Matching
Calculates normalized distance between simulation parameters and experimental target:

$$\text{Score} = \sum_i \left(\frac{X_i^{sim} - X_i^{target}}{X_i^{target}}\right)^2$$

**Advantage**: Identifies simulations matching specific design targets

#### Method 2: Shape-Based Matching (MSE)
Compares entire J-V curve morphology using Mean Squared Error:

$$\text{MSE} = \frac{1}{N}\sum_{i=1}^{N}\left(J_i^{exp} - J_i^{sim}(V_i)\right)^2$$

**Advantage**: Captures realistic device physics and loss mechanisms

### Output
- Experimental and simulated J-V comparison plots
- Parameter-score and MSE comparisons, including top-matching simulations in the batch workflow
- Comparison tables and exported plotted J-V data in `SCAPS Simulations/Plots/`

---

## 4. Solar Cell Performance: EQE and J-V Curve Analysis

### Purpose
This analysis evaluates measured solar-cell performance using external quantum efficiency (EQE) spectra and current density-voltage (J-V) curves. It calculates the short-circuit current density from EQE data, applies an EQE-based correction to measured results when needed, and compares cell performance across samples.

### Key Files
- **`EQE_Analysis.ipynb`**: Processes EQE spectra and the AM1.5G reference solar spectrum to calculate $J_{sc}$. It also calculates a correction factor by comparing EQE-derived and solar-simulator $J_{sc}$ values, generates corrected performance results, plots EQE spectra, and estimates the band gap from EQE data.
- **`JV_Analysis_Correction.ipynb`**: Loads measured J-V data, converts current to current density using the cell area, corrects the current sign where needed, and plots the J-V curves. It also supports extracting series and shunt resistance values.
- **`Solar_Cell_Analysis.ipynb`**: Summarizes device result tables, calculates additional cell parameters, compares results across samples with boxplots, and plots the J-V curves of cells selected for high power-conversion efficiency or open-circuit voltage.

### Key Outputs
- EQE spectra and EQE-derived short-circuit current density
- Correction factors and corrected device-performance results
- EQE-based band-gap estimates
- Corrected J-V curves and series/shunt resistance estimates
- Performance comparisons, boxplots, and J-V curves for selected high-performing cells

---

## File Structure

```
Data_Analysis/
├── README.md                           # This file
├── project_paths.py                    # Shared project/data path helper
├── XRD/                                # XRD analysis notebooks and results
│   ├── Texture_Coefficient_Analysis.ipynb
│   ├── TC_comparison.ipynb
│   └── Plots/
├── UV-Vis/                             # UV-Visible spectroscopy analysis
│   ├── UV_Vis_analysis.ipynb
│   └── Plots/                          # Tauc plots, summaries, and SCAPS absorption files
├── EQE/                                # Solar cell performance analysis
│   ├── EQE/                            # EQE data and plots
│   ├── EQE_Analysis.ipynb
│   ├── JV_Analysis_Correction.ipynb
│   └── Solar_Cell_Analysis.ipynb
├── SCAPS Simulations/                  # SCAPS simulation notebooks
│   ├── SCAPS_Simulations_for_1_plot_scaps.ipynb
│   ├── SCAPS_Simulations_iv_files.ipynb
│   └── Plots/                          # Comparison plots, tables, and exported curves
├── data/                               # XRD, UV-Vis, EQE, and device/simulation inputs
│   ├── 1st Round/
│   ├── 2nd Round/
│   ├── CdS1/
│   ├── CdS2/
│   ├── CdS3_1/
│   ├── CdS3_2/
│   ├── UV-Vis/
│   ├── XRD/
│   └── ...
└── JCPDS-ICDD/                         # XRD reference cards
```

---

## Requirements

### Python Packages (for Jupyter notebooks)
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computing
- `matplotlib`: Basic plotting
- `plotly`: Interactive visualizations
- `scipy`: Scientific computing (interpolation, signal processing, curve fitting)
- `scikit-learn`: Machine learning utilities
- `openpyxl`: Excel file reading/writing
- `jupyter`: Running the analysis notebooks

### Installation
```bash
pip install pandas numpy matplotlib plotly scipy scikit-learn openpyxl jupyter
```

---

## How to Use

### 1. XRD Analysis
```bash
jupyter notebook XRD/Texture_Coefficient_Analysis.ipynb
```
- Use the texture-coefficient notebook to compare experimental peaks with the Sb₂Se₃ reference.
- Use `XRD/TC_comparison.ipynb` to compare the XRD spectra and texture coefficients for samples in `data/XRD/2 Round/`.
- View generated plots and tables in `XRD/Plots/`.

### 2. UV-Vis Band Gap Analysis
```bash
jupyter notebook UV-Vis/UV_Vis_analysis.ipynb
```
- Load UV-Vis measurement data from Excel files
- The notebook will automatically:
  1. Calculate absorption coefficient from transmittance and reflectance
  2. Generate plots of the absorption coefficient and combined Tauc analysis
  3. Smooth the Tauc data with a Savitzky-Golay filter and estimate direct and indirect band gaps
  4. Export individual Tauc plots, data, and a band-gap summary
  5. Generate SCAPS-compatible absorption files and plots

### 3. SCAPS vs Experimental Comparison
- Open `SCAPS Simulations/SCAPS_Simulations_for_1_plot_scaps.ipynb` to compare experiment with a selected SCAPS simulation.
- Open `SCAPS Simulations/SCAPS_Simulations_iv_files.ipynb` to compare multiple `.iv` simulations using parameter scores and J-V curve MSE.
- Set the target performance parameters in the notebook configuration, then review plots and tables in `SCAPS Simulations/Plots/`.

### 4. Solar Cell Performance (EQE and J-V Analysis)
- Run `EQE/EQE_Analysis.ipynb` to calculate EQE-derived $J_{sc}$, correction factors, corrected results, and EQE-based band-gap estimates.
- Run `EQE/JV_Analysis_Correction.ipynb` to process and plot measured J-V curves.
- Run `EQE/Solar_Cell_Analysis.ipynb` to compare device results and visualize selected cells.

### 5. Customization
Edit the configuration cell in the relevant notebook under `SCAPS Simulations/`. Set `TARGET_VOC`, `TARGET_JSC`, `TARGET_FF`, and `TARGET_ETA` to the performance values for the device being compared. The notebooks have separate defaults; also check `ACTIVE_AREA_CM2` and `INVERT_EXP_CURRENT` against the experimental measurement.

---

## Key Outputs and Interpretations

### From XRD Analysis
- **Preferred orientations**: Crystallographic planes with highest texture coefficients
- **Reference-pattern comparison**: Experimental peaks compared with Sb₂Se₃ and MoSe₂ reference patterns
- **Sample comparison**: Texture coefficients and plots for the loaded samples

### From UV-Vis Analysis
- **Optical band gap ($E_g$)**: Both direct and indirect values from Tauc method
- **Band gap uncertainties**: Quantified via error propagation from linear fit
- **Absorption coefficient**: Full spectrum showing material's light absorption properties
- **SCAPS compatibility**: Generated .abs files ready for device simulations

### From SCAPS Comparison
- **Best shape match**: Simulation with J-V curve most similar to experimental data (lowest MSE)
- **Best parameter match**: Simulation whose performance metrics most closely match the configured target values
- **Comparison plots and tables**: Experimental and simulated curves, scores, and summary comparisons

### From Solar Cell Performance Analysis
- **EQE-derived $J_{sc}$**: Current density calculated using the AM1.5G spectrum
- **Corrected results**: Device performance corrected using the EQE-to-solar-simulator current ratio
- **EQE band-gap estimate**: Optical band gap estimated from the EQE spectrum
- **J-V characteristics**: Corrected curves and, where available, series and shunt resistance estimates
- **Device comparisons**: Distributions of measured parameters and J-V curves for selected high-PCE or high-$V_{oc}$ cells

---

## References

### UV-Vis Analysis & Band Gap Determination
1. **Savitzky-Golay Filter**:
   - Savitzky, A. & Golay, M. J. E. (1964). Smoothing and differentiation of data by simplified least squares procedures. Analytical Chemistry, 36(8), 1627-1639.
   - Use case: Digital signal processing for noise reduction while preserving slopes and features in spectroscopy data

### SCAPS Software & Device Simulation
1. **SCAPS 1D Software**: 
   - Website: https://scaps.elis.ugent.be/
   - Developer: Ghent University
   - Purpose: Numerical simulation of thin-film photovoltaic devices

### XRD & Crystallography
1. **Texture Coefficient Analysis**:
   - Cullity, B. D. (1978). Elements of X-ray Diffraction (3rd ed.). Addison-Wesley.
   - Application: Quantifying preferred crystallographic orientations in thin films

### Photovoltaic Device Physics
1. **Solar Cell Characterization**:
   - Green, M. A. (1982). Solar Cells: Operating Principles, Technology and Systems Applications. University of New South Wales Press.
   - Shockley, W. & Queisser, H. J. (1961). Detailed balance limit of efficiency of p-n junction solar cells. Journal of Applied Physics, 32(3), 510-519.
   - Topics: Device physics, J-V characteristics, performance metrics (Voc, Jsc, FF, efficiency)

---
