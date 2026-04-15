# Data Analysis: Comprehensive Photovoltaic Device Characterization

## Overview

This repository contains comprehensive data analysis workflows for photovoltaic devices, combining three key analysis approaches:

1. **X-Ray Diffraction (XRD) Analysis**: Texture coefficient determination and crystallographic characterization of thin-film samples
2. **UV-Visible Spectroscopy (UV-Vis)**: Optical band gap determination and absorption coefficient analysis
3. **Device Performance Modeling**: Comparison of experimental J-V (current density-voltage) measurements with SCAPS 1D numerical simulations

## 1. XRD Data Analysis

### Purpose
X-Ray Diffraction (XRD) analysis characterizes the crystallographic properties of the absorber layer in photovoltaic devices. The analysis focuses on determining the **texture coefficient** of preferred crystallographic orientations, which directly impacts device performance and light absorption characteristics.

### Key Files
- **`Texture_Coefficient_Analysis.ipynb`**: Main analysis notebook for XRD texture coefficient determination
  - Reads XRD diffraction patterns from experimental data
  - Calculates texture coefficients for major crystallographic planes
  - Generates visualizations of orientation distribution
  - Identifies preferred growth directions

- **Data Files**:
  - `data/Training_BraggBrentano_1hr_1.xrdml`: XRD measurement in XML format
  - `data/Training_BraggBrentano_1hr_1.ASC`: XRD measurement in ASCII format
  - `Training_BraggBrentano_Fast.csv`: Processed XRD data

- **JCPDS-ICDD/**: Reference database folder containing standard crystal structure data for phase identification

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
- Preferred orientation identification
- Visualization plots of XRD patterns and texture analysis

---

## 2. UV-Visible Spectroscopy (UV-Vis) Analysis

### Purpose
UV-Visible spectroscopy analysis determines the optical band gap ($E_g$) of semiconductor materials from transmission and reflection measurements. The analysis employs two complementary methods:
- **Tauc Plot Method**: Linear extrapolation of absorption data to determine band gap energy
- **Sigmoid-Boltzmann Fitting**: Smooth fitting approach using sigmoid function for improved accuracy

### Key Files
- **`UV_Vis_analysis.ipynb`**: Main analysis notebook for band gap determination
  - Reads UV-Vis data (wavelength, transmittance, reflectance) from Excel files
  - Calculates absorption coefficient from optical properties
  - Applies Savitzky-Golay smoothing to reduce noise
  - Determines both direct and indirect band gaps using Tauc plots
  - Performs Sigmoid-Boltzmann fitting for alternative band gap estimation
  - Generates SCAPS absorption files (.abs) for device simulations

- **Data Files**:
  - `data/UV-Vis/CdS_JOliveira_7abr26.xlsx`: Raw UV-Vis measurements containing wavelength, transmittance (T), and reflectance (R) data
  - Generated output files in `UV-Vis/Plots/`:
    - `Combined_Direct_Indirect_BandGap.png`: Overview plot comparing all samples
    - `Absorption_Coefficient.png`: Absorption spectra of all samples
    - `Tauc_plot_extrapolation_*.png`: Individual Tauc plots with linear extrapolation
    - `SB_*.png`: Sigmoid-Boltzmann fitting results
    - `results_tauc_plot.xlsx`: Tauc plot data export
    - `Summary_Bandgaps.xlsx`: Summary table of calculated band gaps

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

### Sigmoid-Boltzmann Method

An alternative approach fitting absorption data using:

$$\alpha(E) = \alpha_{max} + \frac{\alpha_{min} - \alpha_{max}}{1 + \exp\left(\frac{E - E_0^{Boltz}}{δE}\right)} \quad \text{(4)}$$

Band gaps calculated from fitted parameters:

$$E_g^{dir} = E_0^{Boltz} - 0.3 \cdot δE \quad \text{(5)}$$

$$E_g^{indir} = E_0^{Boltz} - 4.3 \cdot δE \quad \text{(6)}$$

where 0.3 and 4.3 are empirical constants determined by Zanatta.

### Output
- Direct and indirect optical band gap values (eV) with uncertainties
- Absorption coefficient vs. energy plots
- Comparison of band gaps from both Tauc and Sigmoid-Boltzmann methods
- SCAPS-compatible absorption files for device simulation
- Summary tables with all calculated values

---

## 3. SCAPS 1D Simulation vs Experimental Data Comparison

### Purpose
This analysis compares experimental photovoltaic device performance (measured J-V curves) against theoretical predictions from SCAPS 1D (Solar Cell Capacitance Simulator) numerical simulations. The goal is to:
- Validate device models against experimental measurements
- Identify which simulation parameters best match measured behavior
- Understand dominant loss mechanisms
- Optimize device architecture

### Key Files
- **`SCAPS_Simulations.ipynb`**: Comprehensive analysis notebook featuring:
  1. Configuration of target performance metrics
  2. Reading and processing experimental J-V data
  3. SCAPS simulation data parsing and extraction
  4. Parameter-based comparison (comparing Voc, Jsc, FF, η to target values)
  5. Shape-based comparison (Mean Squared Error analysis)
  6. Summary comparison showing best matching simulations

- **Data Files**:
  - `data/CdS3_IV Graph_1.xlsx`: Experimental J-V curve measurements (raw data)
  - `data/CdS3_IV_scaps.xlsx`: SCAPS 1D simulation results containing:
    - Voltage (V) and current density (mA/cm²) points
    - Extracted parameters: Voc, Jsc, FF (%), η (%)
    - Multiple simulation scenarios with varied device parameters

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
SCAPS solves coupled differential equations describing semiconductor transport:

- **Poisson Equation**: Describes electrostatic potential distribution
- **Continuity Equations**: Govern electron and hole transport with drift-diffusion

SCAPS outputs include:
- J-V curves under illumination (current density vs. voltage)
- Derived parameters: Voc, Jsc, FF, efficiency
- Analysis of loss mechanisms and device architecture impact

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
- Comparison plots showing experimental vs. top 5 simulations (by two methods)
- Parameter matching tables showing deviations from experimental values
- Summary comparison with best-matching simulations
- Identification of optimal device configurations

---

## File Structure

```
Data_Analysis/
├── README.md                           # This file
├── SCAPS_Simulations.ipynb             # SCAPS simulation analysis notebook
├── Texture_Coefficient_Analysis.ipynb  # XRD texture coefficient notebook
├── UV-Vis/                             # UV-Visible spectroscopy analysis
│   ├── UV_Vis_analysis.ipynb           # Band gap determination notebook
│   └── Plots/                          # Generated plots and results
│       ├── Combined_Direct_Indirect_BandGap.png
│       ├── Absorption_Coefficient.png
│       ├── Tauc_plot_extrapolation_*.png
│       ├── SB_*.png
│       ├── SCAPS_Absorption/           # Absorption files for SCAPS
│       │   ├── *_SCAPS_absorption.abs
│       │   └── *_SCAPS_absorption_plot.png
│       ├── results_tauc_plot.xlsx
│       └── Summary_Bandgaps.xlsx
├── XRD/                                # X-Ray Diffraction analysis
│   ├── Texture_Coefficient_Analysis.ipynb
│   ├── TC_results_reach_TC.csv
│   ├── TC_summary_statistics.csv
│   └── Plots/                          # Generated visualization files
├── SCAPS Simulations/                  # SCAPS simulation notebooks
│   ├── SCAPS_Simulations_*.ipynb
│   └── Plots/                          # Comparison plots and figures
├── data/                               # Data folder
│   ├── XRD/
│   │   ├── Training_BraggBrentano_1hr_1.xrdml     # XRD raw data (XML)
│   │   ├── Training_BraggBrentano_1hr_1.ASC       # XRD raw data (ASCII)
│   │   ├── Training_BraggBrentano_Fast.xrdml      # Fast XRD scan
│   │   └── Training_BraggBrentano_Fast.csv
│   ├── UV-Vis/
│   │   └── CdS_JOliveira_7abr26.xlsx              # UV-Vis measurements
│   ├── CdS1/                           # Sample data
│   ├── CdS2/
│   ├── CdS3_1/
│   ├── CdS3_2/
│   └── [Other sample folders]
├── JCPDS-ICDD/                         # Reference crystal structure database
└── [Additional data and results]
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
- `nbformat`: Jupyter notebook support

### Installation
```bash
pip install pandas numpy matplotlib plotly scipy scikit-learn openpyxl nbformat
```

---

## How to Use

### 1. XRD Analysis
```bash
jupyter notebook XRD/Texture_Coefficient_Analysis.ipynb
```
- Load XRD data from `data/XRD/` folder
- Calculate texture coefficients
- View crystallographic analysis results

### 2. UV-Vis Band Gap Analysis
```bash
jupyter notebook UV-Vis/UV_Vis_analysis.ipynb
```
- Load UV-Vis measurement data from Excel files
- The notebook will automatically:
  1. Calculate absorption coefficient from transmittance and reflectance
  2. Generate plots of optical properties
  3. Apply Savitzky-Golay smoothing filter
  4. Extract band gaps using Tauc plot method (direct and indirect)
  5. Perform Sigmoid-Boltzmann fitting for alternative band gap determination
  6. Generate SCAPS-compatible absorption files
  7. Export summary tables with calculated band gap values

### 3. SCAPS vs Experimental Comparison
```bash
jupyter notebook SCAPS\ Simulations/SCAPS_Simulations_*.ipynb
```
- Set **target parameters** (experimental Voc, Jsc, FF, η) in Section 1
- The notebook will automatically:
  1. Load experimental J-V data
  2. Parse SCAPS simulation results
  3. Perform parameter-based matching
  4. Perform shape-based matching (MSE analysis)
  5. Display comparison plots and tables
  6. Identify best-matching simulations

### 4. Customization
Edit the configuration cell in `SCAPS Simulations/SCAPS_Simulations_*.ipynb`:
```python
TARGET_VOC = 0.38        # Volts
TARGET_JSC = 20.31       # mA/cm²
TARGET_FF = 49.56        # %
TARGET_ETA = 4.57        # %
ACTIVE_AREA_CM2 = 0.1    # Cell area in cm²
INVERT_EXP_CURRENT = True  # Sign convention for current
```

---

## Key Outputs and Interpretations

### From XRD Analysis
- **Preferred orientations**: Crystallographic planes with highest texture coefficients
- **Crystal quality**: Sharp peaks indicate good crystallinity
- **Phase purity**: Identification of secondary phases

### From UV-Vis Analysis
- **Optical band gap ($E_g$)**: Both direct and indirect values from Tauc method
- **Band gap uncertainties**: Quantified via error propagation from linear fit
- **Absorption coefficient**: Full spectrum showing material's light absorption properties
- **Comparison of methods**: Tauc plot vs. Sigmoid-Boltzmann approaches
- **SCAPS compatibility**: Generated .abs files ready for device simulations
- **Material quality assessment**: Sharpness of absorption edge and absence of sub-bandgap absorption

### From SCAPS Comparison
- **Best shape match**: Simulation with J-V curve most similar to experimental data (lowest MSE)
- **Best parameter match**: Simulation with Voc, Jsc, FF, η closest to target values
- **Parameter deviations**: Quantified differences in each performance metric
- **Loss mechanism identification**: J-V shape reveals dominant losses (series resistance, shunt resistance, ideality)

---

## References

### UV-Vis Analysis & Band Gap Determination
1. **Savitzky-Golay Filter**:
   - Savitzky, A. & Golay, M. J. E. (1964). Smoothing and differentiation of data by simplified least squares procedures. Analytical Chemistry, 36(8), 1627-1639.
   - Use case: Digital signal processing for noise reduction while preserving slopes and features in spectroscopy data

2. **Sigmoid-Boltzmann Band Gap Fitting**:
   - Zanatta, A. R. (2019). Revisiting the optical band gap of semiconductors and the proposal of a unified diagram. Scientific Reports, 9, 11225.
   - Reference: https://www.nature.com/articles/s41598-019-47670-y
   - Method for determining direct and indirect optical band gaps from absorption coefficient data using empirical constants

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
