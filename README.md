# Data Analysis: XRD Characterization and SCAPS 1D Device Simulation Comparison

## Overview

This repository contains comprehensive data analysis workflows for photovoltaic devices, combining two key analysis approaches:

1. **X-Ray Diffraction (XRD) Analysis**: Texture coefficient determination and crystallographic characterization of thin-film samples
2. **Device Performance Modeling**: Comparison of experimental J-V (current density-voltage) measurements with SCAPS 1D numerical simulations

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

## 2. SCAPS 1D Simulation vs Experimental Data Comparison

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
├── data/                               # Data folder
│   ├── CdS3_IV Graph_1.xlsx            # Experimental J-V measurements
│   ├── CdS3_IV_scaps.xlsx              # SCAPS simulation results
│   ├── Training_BraggBrentano_1hr_1.xrdml     # XRD raw data (XML)
│   ├── Training_BraggBrentano_1hr_1.ASC       # XRD raw data (ASCII)
│   ├── Training_BraggBrentano_Fast.xrdml      # Fast XRD scan
│   └── reach_TC.xlsx                   # Processed XRD results
├── JCPDS-ICDD/                         # Reference crystal structure database
├── TC_results_reach_TC.csv             # Texture coefficient results
├── TC_summary_statistics.csv           # Statistical summary of TC analysis
└── [Output plots and figures]          # Generated visualization files
```

---

## Requirements

### Python Packages (for Jupyter notebooks)
- `pandas`: Data manipulation and analysis
- `numpy`: Numerical computing
- `matplotlib`: Basic plotting
- `plotly`: Interactive visualizations
- `scipy`: Scientific computing (interpolation, MSE calculation)
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
jupyter notebook Texture_Coefficient_Analysis.ipynb
```
- Load XRD data from `data/` folder
- Calculate texture coefficients
- View crystallographic analysis results

### 2. SCAPS vs Experimental Comparison
```bash
jupyter notebook SCAPS_Simulations.ipynb
```
- Set **target parameters** (experimental Voc, Jsc, FF, η) in Section 1
- The notebook will automatically:
  1. Load experimental J-V data
  2. Parse SCAPS simulation results
  3. Perform parameter-based matching
  4. Perform shape-based matching (MSE analysis)
  5. Display comparison plots and tables
  6. Identify best-matching simulations

### 3. Customization
Edit the configuration cell in `SCAPS_Simulations.ipynb`:
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

### From SCAPS Comparison
- **Best shape match**: Simulation with J-V curve most similar to experimental data (lowest MSE)
- **Best parameter match**: Simulation with Voc, Jsc, FF, η closest to target values
- **Parameter deviations**: Quantified differences in each performance metric
- **Loss mechanism identification**: J-V shape reveals dominant losses (series resistance, shunt resistance, ideality)

---

## References

1. **SCAPS Software**: 
   - Website: https://scaps.elis.ugent.be/
   - Developer: Ghent University

2. **Texture Coefficient Analysis**:
   - Cullity, B. D. Elements of X-ray Diffraction (3rd ed.)

3. **Photovoltaic Device Physics**:
   - Green, M. A. Solar Cells: Operating Principles, Technology and Systems Applications
   - Shockley, W. & Queisser, H. J. Detailed Balance Limit of Efficiency of p-n Junction Solar Cells

---

## Contact & Support

For questions or issues related to this analysis, please refer to the individual notebook documentation and comments within the code.

**Last Updated**: March 2026
