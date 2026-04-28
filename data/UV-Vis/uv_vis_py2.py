"""
Band-gap extraction for thin films with alpha*t < 1
=====================================================
Uses the general thin-film optical equations from Slides 21-22:

    T_meas = (1-R)^2 * exp(-alpha*t) / (1 - R^2 * exp(-2*alpha*t))
    R_meas = R + R*(1-R)^2 * exp(-2*alpha*t) / (1 - R^2 * exp(-2*alpha*t))

alpha is extracted numerically by solving a quadratic in x = exp(-alpha*t).
The Tauc plot (alpha*hv)^2 vs hv is then used for direct band-gap determination.

Usage:
    python bandgap_thinfilm_solver.py
    
    Edit the CONFIGURATION section below for your data.

References:
    Swanepoel, J. Phys. E 16, 1214 (1983)   [envelope method]
    Slide 21 of FisMicroSist-Cap5_Optoeletrónica.pdf
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# ─────────────────────────────────────────────
# CONFIGURATION  (edit these)
# ─────────────────────────────────────────────
DATA_FILE   = "C:/Users/jp_ol/OneDrive/Ambiente de Trabalho/TESE/Data_Analysis/data/UV-Vis/CdS_JOliveira_21abr26.xlsx"          # Excel file with columns: wavelength, T (%), R (%)
SHEETS      = ["CdS1", "CdS2", "CdS3"]
THICKNESS_M = 50e-9                   # Film thickness in metres (50 nm)
GAP_TYPE    = "direct"                # "direct" → n=2 → (αhν)²; "indirect" → (αhν)^0.5
LAMBDA_MIN  = 250                     # Wavelength range to load (nm)
LAMBDA_MAX  = 700
EG_RANGE    = (2.5, 4.5)             # Expected band-gap range (eV)
COLORS      = ["#1a73e8","#e8711a","#2da02d","#c62828"]
# ─────────────────────────────────────────────

H_EV = 4.135667696e-15   # Planck constant (eV·s)
C    = 2.998e8            # Speed of light (m/s)


def alpha_from_TR(T_pct: float, R_pct: float, t: float) -> float:
    """
    Numerically solve for the absorption coefficient alpha [m^-1] given
    measured transmittance T (%) and reflectance R (%) and film thickness t (m).

    Method
    ------
    Approximation for Fresnel reflectance at interface:
        R_interface ≈ R_meas / (2 - R_meas)           (low-absorption limit)

    Then solve for x = exp(-alpha*t) from the quadratic:
        T_meas * R_i^2 * x^2  +  (1 - R_i)^2 * x  -  T_meas  =  0

    Return NaN if no physical solution exists.
    """
    T = max(T_pct, 1e-6) / 100.0
    R = max(R_pct, 1e-6) / 100.0
    if T <= 0 or R <= 0 or R >= 1:
        return np.nan

    # Interface reflectance (Fresnel, long-λ limit)
    Ri = np.clip(R / (2.0 - R), 1e-4, 0.9999)

    # Quadratic: a*x^2 + b*x + c = 0,  x = exp(-alpha*t)
    a =  T * Ri**2
    b = (1.0 - Ri)**2
    c = -T

    disc = b**2 - 4.0 * a * c
    if disc < 0:
        return np.nan

    x = (-b + np.sqrt(disc)) / (2.0 * a)
    if not (0 < x <= 1.0):
        x = (-b - np.sqrt(disc)) / (2.0 * a)
    if not (0 < x <= 1.0):
        return np.nan

    return -np.log(x) / t


def process_sheet(df_raw: pd.DataFrame, t: float) -> pd.DataFrame:
    """Compute alpha, hv, and Tauc quantity for one sample."""
    df = df_raw[["wavelength", "T", "R"]].dropna().copy()
    df = df[(df["wavelength"] >= LAMBDA_MIN) & (df["wavelength"] <= LAMBDA_MAX)]
    df["alpha_m"]  = [alpha_from_TR(r.T, r.R, t) for r in df.itertuples(index=False)]
    df["hv_eV"]    = H_EV * C / (df["wavelength"] * 1e-9)
    df["alpha_t"]  = df["alpha_m"] * t

    if GAP_TYPE == "direct":
        df["tauc"] = (df["alpha_m"] * df["hv_eV"]) ** 2
        df["tauc_label"] = r"$(\alpha h\nu)^2$"
    else:
        df["tauc"] = np.sqrt(df["alpha_m"] * df["hv_eV"])
        df["tauc_label"] = r"$(\alpha h\nu)^{1/2}$"

    return df.dropna(subset=["alpha_m", "tauc"]).reset_index(drop=True)


def best_linear_fit(df: pd.DataFrame,
                    hv_col: str = "hv_eV",
                    y_col:  str = "tauc",
                    min_pts: int = 10) -> tuple:
    """
    Slide a window over the Tauc data and return the segment with
    the highest R² whose x-intercept (= Eg) falls in EG_RANGE.
    Returns (Eg, R², slope, intercept, hv_lo, hv_hi).
    """
    data = df[(df[hv_col] > EG_RANGE[0] - 0.5) & (df[hv_col] < EG_RANGE[1] + 0.5)
              & (df[y_col] > 0)].sort_values(hv_col).reset_index(drop=True)
    hvs, ys = data[hv_col].values, data[y_col].values

    best = dict(r2=0, Eg=None, slope=None, intercept=None, lo=None, hi=None)

    for i in range(len(hvs)):
        for j in range(i + min_pts, len(hvs)):
            span = hvs[j] - hvs[i]
            if span < 0.15 or span > 0.60:
                continue
            slope, intercept, r, *_ = linregress(hvs[i:j+1], ys[i:j+1])
            if slope <= 0:
                continue
            Eg = -intercept / slope
            if not (EG_RANGE[0] < Eg < EG_RANGE[1]):
                continue
            if r**2 > best["r2"]:
                best.update(r2=r**2, Eg=Eg, slope=slope, intercept=intercept,
                            lo=hvs[i], hi=hvs[j])

    return (best["Eg"], best["r2"], best["slope"],
            best["intercept"], best["lo"], best["hi"])


def main():
    xl = pd.read_excel(DATA_FILE, sheet_name=None, header=0)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    results = []

    for idx, sheet in enumerate(SHEETS):
        df = process_sheet(xl[sheet], THICKNESS_M)
        Eg, r2, slope, intercept, lo, hi = best_linear_fit(df)

        ax = axes[idx]
        plot = df[(df.hv_eV >= 2.9) & (df.hv_eV <= 4.8) & (df.tauc > 0)]
        ax.plot(plot.hv_eV, plot.tauc, color=COLORS[idx % len(COLORS)],
                lw=1.8, alpha=0.9, label="Data")

        if Eg:
            hv_fit = np.linspace(Eg * 0.99, hi + 0.1, 300)
            ax.plot(hv_fit, np.clip(slope * hv_fit + intercept, 0, None),
                    "k--", lw=1.8, label=f"Linear fit  R²={r2:.4f}")
            ax.axvline(Eg, color="red", ls=":", lw=2.0)
            ax.scatter([Eg], [0], color="red", zorder=6, s=80)
            ymax = plot.tauc.quantile(0.97)
            ax.annotate(f"Eg = {Eg:.3f} eV", xy=(Eg, ymax*0.06),
                        xytext=(Eg + 0.07, ymax * 0.28), fontsize=12,
                        color="red", fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color="red", lw=1.2))
            results.append(dict(sheet=sheet, Eg_eV=round(Eg, 3), R2=round(r2, 4),
                                fit_lo=round(lo, 3), fit_hi=round(hi, 3)))

        tauc_lbl = r"$(\alpha h\nu)^2$" if GAP_TYPE == "direct" else r"$(\alpha h\nu)^{1/2}$"
        ax.set_xlim([2.9, 4.6])
        ax.set_ylim([0, plot.tauc.quantile(0.97) * 1.15])
        ax.set_xlabel("Photon energy  hν (eV)", fontsize=11)
        ax.set_ylabel(f"{tauc_lbl}  [eV² m⁻²]", fontsize=11)
        ax.set_title(sheet, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9.5)
        ax.grid(True, alpha=0.25)
        ax.yaxis.get_major_formatter().set_powerlimits((0, 0))

    fig.suptitle(f"Tauc Plots — {'Direct' if GAP_TYPE=='direct' else 'Indirect'} Band Gap\n"
                 r"$\alpha$ from general thin-film: $T = \frac{(1-R)^2 e^{-\alpha t}}{1-R^2 e^{-2\alpha t}}$",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("tauc_output.png", dpi=160, bbox_inches="tight")
    plt.show()
    print("\n=== Band-gap results ===")
    for r in results:
        print(f"  {r['sheet']}: Eg = {r['Eg_eV']} eV  (R²={r['R2']},  fit {r['fit_lo']}–{r['fit_hi']} eV)")


if __name__ == "__main__":
    main()
