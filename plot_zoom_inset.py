import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import os
import sys

# Aggiungiamo la directory corrente al path per importare correttamente
sys.path.append(os.getcwd())

from inference_comparison import SCENARI, run_prediction_for_model, OUTPUT_DIR
import main

def plot_with_zooms(df_results, alpha, title, save_path):
    """
    Plots the main series and two zoomed insets with refined scaling.
    """
    fig, ax = plt.subplots(figsize=(16, 9))
    
    dates = df_results.index
    
    # --- MAIN PLOT ---
    ax.plot(dates, df_results['Actual'], color='black', alpha=0.2, label='Actual Returns', linewidth=0.8)
    
    model_cols = [c for c in df_results.columns if c != 'Actual']
    colors = [ '#d62728','#1f77b4', '#2ca02c', '#ff7f0e']
    
    for i, col in enumerate(model_cols):
        color = colors[i % len(colors)]
        valid_series = df_results[col].dropna()
        ax.plot(valid_series.index, valid_series, color=color, label=col, linewidth=1.2, alpha=0.7)

    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    ax.legend(loc='upper left', frameon=True, shadow=True)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.set_ylabel("Returns / VaR", fontsize=12)
    
    # --- ZOOM 1: Dec 2020 ---
    zoom1_start = pd.Timestamp("2020-09-23")
    zoom1_end = pd.Timestamp("2020-10-13")
    
    # Inset più piccolo (25% della larghezza)
    axins1 = inset_axes(ax, width="25%", height="25%", loc='upper center', bbox_to_anchor=(0.05, 0.0, 1, 1), bbox_transform=ax.transAxes)
    
    axins1.plot(dates, df_results['Actual'], color='black', alpha=0.3, linewidth=1)
    for i, col in enumerate(model_cols):
        color = colors[i % len(colors)]
        valid_series = df_results[col].dropna()
        axins1.plot(valid_series.index, valid_series, color=color, linewidth=2)
        
        subset = df_results.loc[zoom1_start:zoom1_end]
        if not subset.empty and col in subset.columns:
            violations = subset['Actual'] < subset[col]
            if violations.any():
                axins1.scatter(subset.index[violations], subset.loc[violations, 'Actual'], color=color, marker='o', s=30, zorder=5)

    axins1.set_xlim(zoom1_start, zoom1_end)
    # Scaling Y con margine maggiore (25%)
    sub_df1 = df_results.loc[zoom1_start:zoom1_end]
    if not sub_df1.empty:
        # Consideriamo solo le colonne presenti nello slice
        y_min = sub_df1.min().min()
        y_max = sub_df1.max().max()
        margin = (y_max - y_min) * 0.25
        axins1.set_ylim(y_min - margin, y_max + margin)
    
    axins1.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    axins1.tick_params(axis='both', which='major', labelsize=7)
    axins1.set_title("Dec 2020 Detail", fontsize=10, fontweight='bold')
    axins1.grid(True, linestyle=':', alpha=0.6)
    
    mark_inset(ax, axins1, loc1=2, loc2=4, fc="none", ec="0.4", linestyle='--', alpha=0.6)
    
    # --- ZOOM 2: May 2021 ---
    zoom2_start = pd.Timestamp("2021-03-12")
    zoom2_end = pd.Timestamp("2021-03-25")
    
    axins2 = inset_axes(ax, width="25%", height="25%", loc='center right', bbox_to_anchor=(-0.05, 0.1, 1, 1), bbox_transform=ax.transAxes)
    
    axins2.plot(dates, df_results['Actual'], color='black', alpha=0.3, linewidth=1)
    for i, col in enumerate(model_cols):
        color = colors[i % len(colors)]
        valid_series = df_results[col].dropna()
        axins2.plot(valid_series.index, valid_series, color=color, linewidth=2)
        
        subset = df_results.loc[zoom2_start:zoom2_end]
        if not subset.empty and col in subset.columns:
            violations = subset['Actual'] < subset[col]
            if violations.any():
                axins2.scatter(subset.index[violations], subset.loc[violations, 'Actual'], color=color, marker='o', s=30, zorder=5)
            
    axins2.set_xlim(zoom2_start, zoom2_end)
    sub_df2 = df_results.loc[zoom2_start:zoom2_end]
    if not sub_df2.empty:
        y_min = sub_df2.min().min()
        y_max = sub_df2.max().max()
        margin = (y_max - y_min) * 0.25
        axins2.set_ylim(y_min - margin, y_max + margin)

    axins2.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    axins2.tick_params(axis='both', which='major', labelsize=7)
    axins2.set_title("May 2021 Detail", fontsize=10, fontweight='bold')
    axins2.grid(True, linestyle=':', alpha=0.6)
    
    mark_inset(ax, axins2, loc1=1, loc2=3, fc="none", ec="0.4", linestyle='--', alpha=0.6)

    # Output finale
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot salvato in: {save_path}")

def main():
    scenario_name = "BTC, SPY, GLD e VIX 2018-21"
    if scenario_name not in SCENARI:
        print(f"Errore: Scenario '{scenario_name}' non trovato.")
        return

    config = SCENARI[scenario_name]
    analisi.CONFIG['train_split'] = config['train_split']
    
    df_combined = pd.DataFrame()
    for model_cfg in config['models']:
        s_var, s_actual = run_prediction_for_model(
            model_cfg, 
            config['data_path'], 
            config['train_split'], 
            config['alpha']
        )
        if s_var is None: continue
        if df_combined.empty:
            df_combined = pd.DataFrame(s_actual).rename(columns={'Actual': 'Actual'})
        df_combined[model_cfg['label']] = s_var
        df_combined['Actual'] = df_combined['Actual'].combine_first(s_actual)

    if df_combined.empty:
        return
        
    df_combined.sort_index(inplace=True)
    save_path = os.path.join(OUTPUT_DIR, f"{scenario_name}_zoomed.png")
    plot_with_zooms(df_combined, config['alpha'], f"VaR Comparison with Refined Zooms - {scenario_name}", save_path)

if __name__ == "__main__":
    main()