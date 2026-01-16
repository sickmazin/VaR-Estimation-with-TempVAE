import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import sys

# Importiamo le classi necessarie
from TempVae import TempVAE
from analisi import FinancialDataset, run_inference, run_generation

# ==========================================
# CONFIGURAZIONE DATASET-CENTRICA
# ==========================================
SCENARI = {
    # -------------------------------------------------------------------------
    # SCENARIO 1: Primo Dataset
    # -------------------------------------------------------------------------
    "BTC, SPY, GLD e VIX 2018-21": {
        "data_path": "dataset/log_returns.csv",
        "train_split": 0.66,
        "alpha": 0.05,
        "plots_to_gen": ["series"],

        "models": [
            {
                "label": "Model paper (M=21)",
                "path": "paper_plots/ORIGINALE_2/best_model.pth",
                "color": "blue",
                "seq_len": 21,
                "latent_dim": 10,
                "hidden_dim": 16,
            },
            {
                "label": "Model tuned (M=28)",
                "path": "paper_plots/MIO_18-21/best_model.pth",
                "color": "red",
                "seq_len": 28,
                "latent_dim": 10,
                "hidden_dim": 26,
            }
        ]
    },

    # -------------------------------------------------------------------------
    # SCENARIO 2: Secondo Dataset (Esempio con Regimi)
    # -------------------------------------------------------------------------
     "Asset Class Universe 2004-25": {
        "data_path": "dataset/log_returns_completo.csv",
        "train_split": 0.66,
        "alpha": 0.05,
        "plots_to_gen": ["series", "regimes"],

         "models": [
             {
                 "label": "Model paper(M=21)",
                 "path": "paper_plots/ORIGINALE_00-25/best_model.pth",
                 "color": "blue",
                 "seq_len": 21,
                 "latent_dim": 10,
                 "hidden_dim": 16,
             },
             {
                 "label": "Model tuned (M=28)",
                 "path": "paper_plots/MIO_00-25/best_model.pth",
                 "color": "red",
                 "seq_len": 28,
                 "latent_dim": 10,
                 "hidden_dim": 26,
             }
         ]
    }
}
OUTPUT_DIR = "comparison_results"


DEVICE = 'cpu'#'cuda' if torch.cuda.is_available() else 'cpu'

# ==========================================
# FUNZIONI DI PLOTTING (BASATE SU PANDAS)
# ==========================================

def plot_var_series_multi(df_results, alpha, title, save_path):
    """
    df_results: DataFrame con indice Datetime e colonne: ['Actual', 'VaR Model A', 'VaR Model B', ...]
    """
    plt.figure(figsize=(14, 7))

    dates = df_results.index

    # 1. Plot Actual (Sfondo)
    # Usiamo la colonna 'Actual' (che è comune, ma prendiamo la prima non-nulla se ci sono buchi dovuti all\'allineamento)
    plt.plot(dates, df_results['Actual'], color='black', alpha=0.4, label='Actual Returns', linewidth=1)

    # 2. Plot Modelli
    # Identifica le colonne dei modelli (tutte tranne Actual)
    model_cols = [c for c in df_results.columns if c != 'Actual']

    # Recupera i colori salvati negli attributi del dataframe se possibile, altrimenti ciclo standard
    # Per semplicità qui usiamo una mappa colori o passiamo i colori nel df (non ideale).
    # Facciamo che cerchiamo di recuperare i colori dalla config globale passata come argomento?
    # Meglio: assumiamo che df_results abbia colonne con i nomi label dei modelli.

    colors = ['red', 'blue', 'green', 'orange', 'purple']

    for i, col in enumerate(model_cols):
        # Cerca il colore nella config (metodo lento ma sicuro) o usa default
        color = colors[i % len(colors)]

        var_series = df_results[col]
        # Pulisci NaN (perché modelli diversi iniziano in date diverse)
        valid_idx = var_series.dropna().index

        plt.plot(valid_idx, var_series[valid_idx], color=color, label=col, linewidth=1.5, alpha=0.9)

        # Violazioni
        # Confronta con Actual sulle date valide
        subset = df_results.loc[valid_idx]
        violations = subset['Actual'] < subset[col]

        if violations.any():
            v_dates = subset[violations].index
            v_vals = subset.loc[violations, 'Actual']
            plt.scatter(v_dates, v_vals, color=color, marker='o', s=15, zorder=5, alpha=0.6)

    plt.title(f"{title} (VaR {1-alpha:.0%})", fontsize=14)
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"   -> Salvato: {save_path}")

def plot_regimes_multi(df_results, alpha, title_prefix, save_path):
    # Ampliamo la lista dei regimi per coprire anche dati storici
    potential_periods = [
        ("GFC (2008)", "2008-08-01", "2009-03-31"),
        ("Euro Debt Crisis (2011)", "2011-05-01", "2011-12-31"),
        ("Covid Crash (2020)", "2020-01-01", "2020-06-30"),
        ("Bear Market (2022)", "2022-01-01", "2022-12-31"),
        ("Recent Period", "2024-01-01", "2025-12-31")
    ]
    
    valid_periods = []
    
    # Assicuriamoci che l'indice sia datetime ordinato
    df_results.sort_index(inplace=True)
    d_start, d_end = df_results.index[0], df_results.index[-1]
    
    print(f"    Check Regimi su intervallo dati: {d_start.date()} -> {d_end.date()}")
    
    for name, s, e in potential_periods:
        # Controlla se c'è sovrapposizione tra [s, e] e [d_start, d_end]
        # Overlap logic: max(start1, start2) < min(end1, end2)
        overlap_start = max(pd.Timestamp(s), d_start)
        overlap_end = min(pd.Timestamp(e), d_end)
        
        if overlap_start < overlap_end:
            valid_periods.append((name, s, e))
            
    if not valid_periods:
        print(f"    [WARNING] Nessun periodo di stress predefinito cade nell'intervallo dati del Test Set.")
        return

    fig, axes = plt.subplots(len(valid_periods), 1, figsize=(14, 6 * len(valid_periods)))
    if len(valid_periods) == 1: axes = [axes]
    
    plt.subplots_adjust(hspace=0.3)
    model_cols = [c for c in df_results.columns if c != 'Actual']
    # Palette colori fissa per consistenza
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    
    for ax, (name, start, end) in zip(axes, valid_periods):
        # Slice DataFrame per date
        df_sub = df_results.loc[start:end]
        
        if df_sub.empty: 
            continue
        
        # Actual
        ax.plot(df_sub.index, df_sub['Actual'], color='black', alpha=0.3, linewidth=1, label='Actual' if ax == axes[0] else "")
        
        # Models
        for i, col in enumerate(model_cols):
            color = colors[i % len(colors)]
            series = df_sub[col].dropna()
            
            if series.empty: continue
            
            # Re-allinea indici per evitare errori di shape
            common_idx = series.index.intersection(df_sub.index)
            if common_idx.empty: continue
            
            series_aligned = series.loc[common_idx]
            actual_aligned = df_sub.loc[common_idx, 'Actual']

            ax.plot(common_idx, series_aligned, color=color, linewidth=2, label=col)
            
            # Violazioni
            viol = actual_aligned < series_aligned
            if viol.any():
                v_dates = common_idx[viol]
                v_vals = actual_aligned[viol]
                ax.scatter(v_dates, v_vals, color=color, marker='x', s=40, zorder=5)

        ax.set_title(f"{title_prefix} - {name}", fontsize=12, fontweight='bold')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.grid(True, alpha=0.3)
        if ax == axes[0]: ax.legend(loc='upper right')

    plt.savefig(save_path)
    plt.close()
    print(f"   -> Salvato Regimi: {save_path}")

# ==========================================
# ENGINE DI CALCOLO
# ==========================================

def run_prediction_for_model(model_cfg, data_path, train_split, alpha):
    """
    Istanzia un Dataset specifico per la seq_len del modello.
    Restituisce una Pandas Series con indice=Date, value=VaR.
    Restituisce anche la serie Actual Returns allineata a queste date.
    """
    seq_len = model_cfg['seq_len']
    
    # 1. Init Dataset specifico
    ds = FinancialDataset(data_path, seq_len)
    # Sovrascriviamo split se diverso nel config generale (ma qui ds lo calcola nel init)
    # FinancialDataset usa CONFIG globale in analisi.py? 
    # In analisi.py: split_idx = int(len(self.data) * CONFIG['train_split'])
    # Se cambiamo train_split in CONFIG globale prima di init, funziona.
    
    input_dim = ds.X_train.shape[-1]
    
    # 2. Init Model
    model = TempVAE(
        input_dim=input_dim,
        latent_dim=model_cfg['latent_dim'],
        hidden_dim=model_cfg['hidden_dim'],
        dropout=0.0
    ).to(DEVICE)
    
    try:
        model.load_state_dict(torch.load(model_cfg['path'], map_location=DEVICE))
        model.eval()
    except Exception as e:
        print(f"    [ERROR] Load failed for {model_cfg['label']}: {e}")
        return None, None

    # 3. Monte Carlo VaR
    X_test = ds.X_test.to(DEVICE)
    MC = 1000
    weights_pf = np.ones(input_dim) / input_dim
    var_forecasts = []
    actual_returns = []
    
    print(f"    -> Calcolo VaR per {model_cfg['label']} (SeqLen={seq_len})...")
    
    with torch.no_grad():
        mu_z_seq = run_inference(model, X_test)
        z_last = mu_z_seq[:, -1, :] 
        B = z_last.shape[0]
        
        # Expand
        z_expanded = z_last.unsqueeze(1).expand(B, MC, -1).reshape(B*MC, -1).unsqueeze(1)
        
        # Generate
        mu_r, diag_r = run_generation(model, z_expanded)
        mu_r = mu_r.squeeze(1)
        diag_r = diag_r.squeeze(1)
        
        # Sample
        eps = torch.randn_like(mu_r)
        r_sim = (mu_r + torch.sqrt(diag_r) * eps).reshape(B, MC, input_dim).cpu().numpy()
        
        # Calc VaR
        for i in range(B):
            sim_destand = r_sim[i] * ds.std + ds.mean
            port_sim = np.dot(sim_destand, weights_pf)
            var_forecasts.append(np.percentile(port_sim, alpha * 100))
            
            # Actual Return (per verifica allineamento)
            real_scaled = ds.X_test[i, -1, :].numpy()
            real_val = real_scaled * ds.std + ds.mean
            actual_returns.append(np.dot(real_val, weights_pf))
            
    # Crea Pandas Series
    dates = ds.test_dates
    
    # Check lengths
    if len(dates) != len(var_forecasts):
        print(f"WARNING: Dates length {len(dates)} != Forecasts {len(var_forecasts)}")
        min_len = min(len(dates), len(var_forecasts))
        dates = dates[:min_len]
        var_forecasts = var_forecasts[:min_len]
        actual_returns = actual_returns[:min_len]

    s_var = pd.Series(data=var_forecasts, index=dates, name=f"VaR {model_cfg['label']}")
    s_actual = pd.Series(data=actual_returns, index=dates, name="Actual")
    
    return s_var, s_actual

def process_scenario(scenario_name, config):
    print(f"\n>>> ANALISI SCENARIO: {scenario_name}")
    
    # Impostiamo CONFIG globale in analisi (hack necessario perché FinancialDataset la usa)
    import analisi
    analisi.CONFIG['train_split'] = config['train_split']
    
    # DataFrame contenitore: Indicizzato per Data, contiene Actual e Colonne Modelli
    # Inizializziamo con Actual vuoto, lo riempiremo col primo modello
    df_combined = pd.DataFrame()
    
    for model_cfg in config['models']:
        s_var, s_actual = run_prediction_for_model(
            model_cfg, 
            config['data_path'], 
            config['train_split'], 
            config['alpha']
        )
        
        if s_var is None: continue
        
        # Merge intelligente
        if df_combined.empty:
            df_combined = pd.DataFrame(s_actual).rename(columns={'Actual': 'Actual'})
            
        # Aggiungi colonna VaR del modello (allinea automaticamente le date)
        df_combined[model_cfg['label']] = s_var
        
        # Aggiorna Actual se abbiamo più dati (es. se questo modello ha seq_len più corta
        # e quindi inizia prima, vogliamo che 'Actual' copra anche quel periodo)
        # combine_first riempie i buchi di Actual esistente con i valori nuovi
        df_combined['Actual'] = df_combined['Actual'].combine_first(s_actual)

    if df_combined.empty or len(df_combined.columns) <= 1:
        print("    [WARNING] Nessun risultato da plottare.")
        return

    # Ordina per data
    df_combined.sort_index(inplace=True)

    # Plotting
    if "series" in config['plots_to_gen']:
        plot_var_series_multi(
            df_combined, config['alpha'],
            title=f"VaR Comparison - {scenario_name}",
            save_path=os.path.join(OUTPUT_DIR, f"{scenario_name}_series.png")
        )
        
    if "regimes" in config['plots_to_gen']:
        plot_regimes_multi(
            df_combined, config['alpha'],
            title_prefix=scenario_name,
            save_path=os.path.join(OUTPUT_DIR, f"{scenario_name}_regimes.png")
        )

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    for name, cfg in SCENARI.items():
        process_scenario(name, cfg)
        
    print(f"\n>>> DONE. Output in: {OUTPUT_DIR}")
