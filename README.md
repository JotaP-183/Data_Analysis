# Data Analysis: Photovoltaic Device Characterization

## Overview

This repository contains comprehensive data analysis workflows for photovoltaic devices, combining four key analysis approaches:

1. **X-Ray Diffraction (XRD) Analysis**: Texture coefficient determination and crystallographic characterization of thin-film samples
2. **UV-Visible Spectroscopy (UV-Vis)**: Optical band gap determination and absorption coefficient analysis
3. **Device Performance Modeling**: Comparison of experimental J-V (current density-voltage) measurements with SCAPS 1D numerical simulations
4. **Solar Cell Performance**: EQE and J-V curve analysis to evaluate and compare photovoltaic device performance

## 1. XRD Data Analysis

### Purpose
X-Ray Diffraction (XRD) analysis characterizes the crystallographic properties of the absorber layer in photovoltaic devices. The analysis focuses on determining the **texture coefficient** of preferred crystallographic orientations, which directly impacts device performance and light absorption characteristics.

### Key Files
- **`TC_comparison.ipynb`**: Main analysis notebook for XRD texture coefficient determination
  - Reads XRD diffraction patterns from experimental data
  - Calculates texture coefficients for major crystallographic planes
  - Generates visualizations of orientation distribution
  - Identifies preferred growth directions

### Texture Coefficient
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

## 2. UV-Visible Spectrophotometry (UV-Vis) Analysis

### Purpose
UV-Visible spectrophotometry analysis determines absorption and the optical band gap ($E_g$) of semiconductor materials from transmission and reflection measurements. The analysis employs the **Tauc Plot Method** for the linear extrapolation of absorption data to determine band gap energy.


### Key Files
- **`UV_Vis_analysis.ipynb`**: Main analysis notebook for band gap determination
  - Reads UV-Vis data (wavelength, transmittance, reflectance) from Excel files
  - Calculates absorption coefficient from optical properties
  - Applies Savitzky-Golay smoothing to reduce noise
  - Determines both direct and indirect band gaps using Tauc plots
  - Generates SCAPS absorption files (.abs) for device simulations

### Absorption Coefficient

The absorption coefficient is calculated from transmittance and reflectance using:

- **If αd >> 1** (high absorption regime): $R_{\text{meas}} \approx R$, which means the simple approximation:

$$
\alpha = -\frac{1}{d}\ln\left[
\frac{T_{\text{meas}}}{(1-R_{\text{meas}})^2}
\right]
$$

- **If αd ≤ 1** (low absorption regime):
$$\begin{cases}
T_{\text{meas}} = \frac{(1-R)^2 e^{-\alpha d}}{1 - R^2 e^{-2\alpha d}} \\
e^{-2\alpha d} = \frac{R_{\text{meas}} - R}{R(1 - 2R + R R_{\text{meas}} )}
\end{cases}$$

where α and R can be determined simultaneously and:
- **$T_{\text{meas}}$** is the measured transmittance
- **R** is the "reflection coefficient"
- **α** is the absorption coefficient (cm⁻¹)
- **d** is the thickness of the film (cm)

### Photon Energy Calculation

The photon energy (in eV) is calculated from wavelength using:

$$h\nu = \frac{hc}{\lambda} \cdot \frac{1}{e} \quad \text{(2)}$$

where:
- $h$ = Planck's constant (6.626 × 10⁻³⁴ J·s)
- $c$ = speed of light (3 × 10⁸ m/s)
- $\lambda$ = wavelength (nm)
- $e$ = elemental charge (1 eV)

### Tauc Plot Method for Bandgap determination

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
- Direct and indirect optical band gap values (eV) with uncertainties
- Absorption coefficient vs. energy plots
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
- **`SCAPS_Simulations_iv_files.ipynb`**: Comprehensive analysis notebook featuring:
  1. Configuration of target performance metrics
  2. Reading and processing experimental J-V data
  3. SCAPS simulation data parsing and extraction
  4. Parameter-based comparison (comparing Voc, Jsc, FF, η to target values)
  5. Shape-based comparison (Mean Squared Error analysis)
  6. Summary comparison showing best matching simulations

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

## References

### UV-Vis Analysis & Band Gap Determination
1. **Absorption deduction**:
   - David C. Look, Jacob H. Leach; On the accurate determination of absorption coefficient from reflectanceand transmittance measurements: Application to Fe-doped GaN. J. Vac. Sci. Technol. B 1 July 2016; 34 (4): 04J105.
   - Purpose: absorption coefficient and reflecctance coefficient deduction.
2. **Savitzky-Golay Filter**:
   - Savitzky, A. & Golay, M. J. E. (1964). Smoothing and differentiation of data by simplified least squares procedures. Analytical Chemistry, 36(8), 1627-1639.
   - Purpose: Digital signal processing for noise reduction while preserving slopes and features in spectroscopy data.

### SCAPS Software & Device Simulation
1. **SCAPS 1D Software**: 
   - Website: https://scaps.elis.ugent.be/
   - Purpose: Numerical simulation of thin-film photovoltaic devices.

### XRD & Crystallography
1. **Texture Coefficient Analysis**:
   - Cullity, B. D. (1978). Elements of X-ray Diffraction (3rd ed.). Addison-Wesley.
   - Purposee: Quantifying preferred crystallographic orientations in thin films.

### Photovoltaic Device Physics
1. **Solar Cell Characterization**:
   - Green, M. A. (1982). Solar Cells: Operating Principles, Technology and Systems Applications. University of New South Wales Press.
   - Shockley, W. & Queisser, H. J. (1961). Detailed balance limit of efficiency of p-n junction solar cells. Journal of Applied Physics, 32(3), 510-519.
   - Purpose: Device physics, J-V characteristics, performance metrics (Voc, Jsc, FF, efficiency) determination.

---
