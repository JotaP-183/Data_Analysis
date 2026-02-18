import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# ---------------------------------------------------------
# 1. REFERENCE DATABASE
# ---------------------------------------------------------
reference_card = {
    '110': {'2theta': 10.676, 'I0': 8.0},
    '020': {'2theta': 15.029, 'I0': 25.0},
    '200': {'2theta': 15.211, 'I0': 6.0},
    '120': {'2theta': 16.874, 'I0': 55.0},
    '220': {'2theta': 21.446, 'I0': 10.0},
    '101': {'2theta': 23.643, 'I0': 12.0},
    '130': {'2theta': 23.901, 'I0': 30.0},
    '310': {'2theta': 24.152, 'I0': 16.0},
    '021': {'2theta': 27.023, 'I0': 12.0},
    '230': {'2theta': 27.395, 'I0': 70.0},
    '211': {'2theta': 28.2, 'I0': 75.0},
    '040': {'2theta': 30.326, 'I0': 2.0},
    '400': {'2theta': 30.699, 'I0': 4.0},
    '221': {'2theta': 31.16, 'I0': 100.0},
    '301': {'2theta': 32.22, 'I0': 60.0},
    '330': {'2theta': 32.424, 'I0': 10.0},
    '311': {'2theta': 33.115, 'I0': 20.0},
    '240': {'2theta': 34.075, 'I0': 60.0},
    '420': {'2theta': 34.358, 'I0': 20.0},
    '321': {'2theta': 35.7, 'I0': 30.0},
    '041': {'2theta': 37.984, 'I0': 25.0},
    '340': {'2theta': 38.354, 'I0': 14.0},
    '430': {'2theta': 38.49, 'I0': 14.0},
    '141': {'2theta': 38.801, 'I0': 35.0},
    '411': {'2theta': 39.081, 'I0': 12.0},
    '510': {'2theta': 39.456, 'I0': 8.0},
    '331': {'2theta': 39.71, 'I0': 8.0},
    '250': {'2theta': 41.305, 'I0': 35.0},
    '520': {'2theta': 41.705, 'I0': 20.0},
    '440': {'2theta': 43.694, 'I0': 10.0},
    '431': {'2theta': 44.95, 'I0': 35.0},
    '501': {'2theta': 45.068, 'I0': 30.0},
    '530': {'2theta': 45.258, 'I0': 30.0},
    '151': {'2theta': 45.354, 'I0': 25.0},
    '002': {'2theta': 45.571, 'I0': 25.0},
    '060': {'2theta': 46.209, 'I0': 8.0},
    '600': {'2theta': 46.84, 'I0': 8.0},
    '160': {'2theta': 46.892, 'I0': 8.0},
    '610': {'2theta': 47.49, 'I0': 8.0},
    '212': {'2theta': 48.902, 'I0': 8.0},
    '620': {'2theta': 49.469, 'I0': 4.0},
    '441': {'2theta': 49.584, 'I0': 4.0},
    '540': {'2theta': 49.903, 'I0': 4.0},
    '351': {'2theta': 50.765, 'I0': 8.0},
    '531': {'2theta': 51.039, 'I0': 16.0},
    '061': {'2theta': 51.879, 'I0': 45.0},
    '360': {'2theta': 52.166, 'I0': 20.0},
    '322': {'2theta': 53.956, 'I0': 14.0},
    '621': {'2theta': 54.865, 'I0': 6.0},
    '710': {'2theta': 55.807, 'I0': 6.0},
    '142': {'2theta': 56.328, 'I0': 1.0},
    '412': {'2theta': 56.479, 'I0': 1.0},
    '640': {'2theta': 56.821, 'I0': 6.0},
    '270': {'2theta': 56.898, 'I0': 8.0},
    '720': {'2theta': 57.559, 'I0': 12.0},
    '242': {'2theta': 58.115, 'I0': 18.0},
    '422': {'2theta': 58.276, 'I0': 10.0},
    '370': {'2theta': 59.854, 'I0': 8.0},
    '171': {'2theta': 60.155, 'I0': 6.0},
}

# ---------------------------------------------------------
# HELPER FUNCTION: PARSE VALUE AND ERROR
# ---------------------------------------------------------
def parse_value_with_error(value_str):
    """
    Parses a string like "99.1234(8)" and returns (99.1234, 0.0008).
    """
    value_str = str(value_str).strip().replace(',', '.')
    
    if '(' not in value_str:
        try:
            return float(value_str), 0.0
        except:
            return np.nan, np.nan

    try:
        parts = value_str.split('(')
        base_str = parts[0]
        error_str = parts[1].replace(')', '')
        
        base_val = float(base_str)
        error_int = int(error_str)
        
        if '.' in base_str:
            decimal_places = len(base_str.split('.')[1])
            factor = 10 ** (-decimal_places)
            error_val = error_int * factor
        else:
            error_val = float(error_int)
            
        return base_val, error_val
        
    except Exception as e:
        return np.nan, np.nan

# ---------------------------------------------------------
# 2. CALCULATION FUNCTION
# ---------------------------------------------------------
def calculate_texture_coefficient(file_path, tolerance_degrees=0.1):
    print(f"--> Attempting to read: {file_path}")

    try:
        df = pd.read_excel(file_path)
    except:
        print("Excel failed, attempting to read as CSV...")
        try:
            df = pd.read_csv(file_path, sep=';') 
            if df.shape[1] < 2:
                df = pd.read_csv(file_path, sep=',') 
        except Exception as e:
            print(f"CRITICAL ERROR: Could not read the file. Details: {e}")
            return

    try:
        col_pos = [c for c in df.columns if "Pos" in c and "2Th" in c][0]
        col_height = [c for c in df.columns if "Height" in c][0] 
        col_match = [c for c in df.columns if "Matched by" in c][0]
        print(f"Identified columns: Pos='{col_pos}', Height='{col_height}', Match='{col_match}'")
    except IndexError:
        print("ERROR: Column names not found.")
        return

    # =========================================================================
    # DATA SEPARATION
    # =========================================================================
    
    pos_vals, pos_errs = [], []
    height_vals, height_errs = [], []

    for x in df[col_pos]:
        v, e = parse_value_with_error(x)
        pos_vals.append(v)
        pos_errs.append(e)
        
    for x in df[col_height]:
        v, e = parse_value_with_error(x)
        height_vals.append(v)
        height_errs.append(e)

    df[col_pos] = pos_vals
    df['Pos_Error'] = pos_errs
    df[col_height] = height_vals
    df['Height_Error'] = height_errs 

    df = df.dropna(subset=[col_pos, col_height])
    # =========================================================================

    df_filtered = df[df[col_match].astype(str).str.contains("00-015-0861", na=False)].copy()

    if df_filtered.empty:
        print("WARNING: No peak corresponding to '00-015-0861' found.")
        return

    print(f"--> Peaks found: {len(df_filtered)}")

    hkl_labels = []
    I_exp_raw = [] 
    I_exp_error_raw = [] 
    I_std_raw = [] 

    # MATCHING
    for hkl, ref_data in reference_card.items():
        ref_2theta = ref_data['2theta']
        ref_int = ref_data['I0']

        mask = (df_filtered[col_pos] >= ref_2theta - tolerance_degrees) & (df_filtered[col_pos] <= ref_2theta + tolerance_degrees)
        matches = df_filtered[mask]

        if not matches.empty:
            idx_max = matches[col_height].idxmax()
            exp_int = matches.loc[idx_max, col_height]
            exp_err = matches.loc[idx_max, 'Height_Error']

            hkl_labels.append(hkl)
            I_exp_raw.append(exp_int)
            I_exp_error_raw.append(exp_err)
            I_std_raw.append(ref_int)

    I_exp = np.array(I_exp_raw)
    I_exp_err = np.array(I_exp_error_raw)
    I_std = np.array(I_std_raw)
    N = len(I_exp)
    
    if N == 0:
        print("ERROR: No matching peaks found!")
        return

    # Normalization
    max_val = np.max(I_exp)
    if max_val != 0:
        I_exp_norm = (I_exp / max_val) * 100
        I_exp_norm_err = (I_exp_err / max_val) * 100
    else:
        I_exp_norm = I_exp
        I_exp_norm_err = I_exp_err

    # TC CALCULATION
    ratios = I_exp_norm / I_std
    average_denominator = np.average(ratios)
    
    if average_denominator == 0:
        print("Error: Denominator is zero.")
        return

    TC_values = ratios / average_denominator
    TC_errors = (I_exp_norm_err / I_std) / average_denominator

    # =========================================================================
    # WINDOW 1: BAR CHART
    # =========================================================================
    fig1 = plt.figure(figsize=(10, 6))
    plt.title('Sb2Se3 - XRD DATA', fontsize=14)
    
    bars = plt.bar(hkl_labels, TC_values, 
                   yerr=TC_errors, capsize=4, ecolor='gray',
                   color='royalblue', edgecolor='black')
    
    plt.xlabel('Plane (hkl)', fontsize=12)
    plt.ylabel('Texture Coefficient (TC)', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend()
    
    for bar in bars:
        yval = bar.get_height()
        if yval > 0.01:
             plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, round(yval, 2), ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()

    # =========================================================================
    # WINDOW 2: DATA TABLE
    # =========================================================================
    table_data = []
    
    for i, (hkl, tc, err) in enumerate(zip(hkl_labels, TC_values, TC_errors), 1):
        # Combine TC and Error
        combined_val = f"{tc:.4f} \u00B1 {err:.4f}"
        table_data.append([i, hkl, combined_val])

    # Create table figure
    fig2, ax_table = plt.subplots(figsize=(6, len(table_data)*0.5 + 1))
    ax_table.axis('tight')
    ax_table.axis('off')
    ax_table.set_title("Detailed Results", fontweight="bold")

    # Draw Table
    table = ax_table.table(cellText=table_data,
                           colLabels=["No.", "Plane (hkl)", "TC Value"],
                           cellLoc='center',
                           loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.6) 
    plt.show()

# ---------------------------------------------------------
# 3. EXECUTION
# ---------------------------------------------------------
file_path = "C:/Users/jp_ol/OneDrive/Ambiente de Trabalho/XRD/reach_TC.xlsx"

calculate_texture_coefficient(file_path)