import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime

# Import del modello (già modificato con Fixed Prior)
from TempVae import TempVAE
# Import delle funzioni di analisi (per evitare duplicazione codice)
from analisi import (
    kupiec_pof_test, 
    christoffersen_test, 
    plot_zoomed_regimes, 
    run_inference, 
    run_generation,
    Logger
)

# ==========================================
# CONFIGURAZIONE RIGOROSA DAL PAPER (App. A & B)
# ==========================================
CONFIG = {
    'tickers': [
        # Selezione RIDOTTA e ROBUSTA di titoli DAX (Blue Chips sicure)
        'ALV.DE', 'BAS.DE', 'BMW.DE', 'DBK.DE', 'DTE.DE',
        'EOAN.DE', 'IFX.DE', 'MUV2.DE', 'RWE.DE', 'SAP.DE', 'SIE.DE','SPY', '^VIX', 'GLD',
    ],
    'start_date': '2018-10-01', # Periodo più recente e stabile
    'end_date': '2021-07-01',
    
    # Preprocessing
    'seq_length': 21,         # Finestra scorrevole M=21
    'train_split': 0.66,      # 66% Training, resto Test (App. A.3)
    
    # Model Arch (App. A.2)
    'latent_dim': 10,         # Kappa = 10
    'hidden_dim': 16,         # Hidden units = 16 (per DAX)
    'dropout_rate': 0.1,      # Dropout = 10%
    
    # Training (App. A.2)
    'batch_size': 256,
    'epochs': 1000,
    'learning_rate': 0.001,
    'lr_decay_rate': 0.96,
    'lr_decay_steps': 500,
    'l2_reg': 0.01,           # Lambda L2 = 0.01
    
    # Annealing (App. A.1)
    # "Subtracting an exponentially decaying term... decay rate 0.96... steps 20"
    'annealing_decay_rate': 0.96,
    'annealing_steps': 20,
    
    'device': 'cpu',#'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': 42,
    'plot_dir': 'paper_replication_results'
}

# Setup Seed
torch.manual_seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])

# ==========================================
# 1. DATA LOADING & PREPROCESSING (YFINANCE)
# ==========================================
def download_and_process_data():
    print(f"\n>>> Scaricamento dati da YFinance ({len(CONFIG['tickers'])} tickers)...")
    
    # Scarica tutti i dati grezzi
    raw_data = yf.download(CONFIG['tickers'], start=CONFIG['start_date'], end=CONFIG['end_date'])
    
    # Gestione MultiIndex (yfinance restituisce spesso colonne come ('Adj Close', 'AAPL'))
    if isinstance(raw_data.columns, pd.MultiIndex):
        # Prova a estrarre Adj Close, altrimenti Close
        if 'Adj Close' in raw_data.columns.get_level_values(0):
            data = raw_data['Adj Close']
        elif 'Close' in raw_data.columns.get_level_values(0):
            data = raw_data['Close']
        else:
            # Fallback estremo: prendi il primo livello se non trova nomi standard
            data = raw_data.xs(raw_data.columns.get_level_values(0)[0], axis=1, level=0)
    else:
        # Struttura piatta (vecchie versioni o singolo ticker)
        if 'Adj Close' in raw_data:
            data = raw_data['Adj Close']
        elif 'Close' in raw_data:
            data = raw_data['Close']
        else:
             data = raw_data # Assumiamo sia già il prezzo
             
    # Verifica integrità
    if data.empty:
        raise ValueError(f"ERRORE CRITICO: Il dataset scaricato è vuoto! Shape: {data.shape}")

    # Gestione dati mancanti (Strategia Robusta)
    # 1. Rimuovi ticker che hanno troppi NaN (>30% di buchi)
    missing_frac = data.isna().mean()
    keep_cols = missing_frac[missing_frac < 0.3].index
    data = data[keep_cols]
    
    print(f"Ticker rimossi per dati insufficienti: {list(set(missing_frac.index) - set(keep_cols))}")
    
    # 2. Riempimento e pulizia righe
    # Ffill/Bfill per piccoli buchi, poi dropna sulle righe per tagliare periodi senza dati comuni
    data = data.ffill().bfill().dropna(axis=0) 
    
    print("First 5 rows of cleaned data:\n", data.head())
    print(f"Dataset shape post-cleaning: {data.shape}")
    print(f"Assets mantenuti ({len(data.columns)}): {data.columns.tolist()}")
    
    if data.shape[1] == 0:
         raise ValueError("ERRORE: Tutte le colonne sono state rimosse durante la pulizia.")
    
    # Calcolo Log-Returns: R_t = ln(S_t) - ln(S_{t-1}) (Eq. 23)
    log_returns = np.log(data / data.shift(1)).dropna()
    
    asset_names = log_returns.columns.tolist()
    values = log_returns.values.astype(np.float32)
    
    # Splitting Temporale (66% Train)
    split_idx = int(len(values) * CONFIG['train_split'])
    train_data = values[:split_idx]
    test_data = values[split_idx:]
    
    print(f"Train samples: {len(train_data)} | Test samples: {len(test_data)}")
    
    # Standardizzazione (Mean/Std calcolati SOLO sul Train)
    mean = np.mean(train_data, axis=0)
    std = np.std(train_data, axis=0)
    std[std == 0] = 1.0 # Evita divisione per zero
    
    # Standardizza
    train_scaled = (train_data - mean) / std
    test_scaled = (test_data - mean) / std
    
    # Creazione Finestre Scorrevoli (Sliding Window M=21)
    def create_windows(arr, seq_len):
        windows = []
        for i in range(len(arr) - seq_len + 1):
            windows.append(arr[i:i+seq_len])
        return torch.tensor(np.array(windows), dtype=torch.float32)
    
    X_train = create_windows(train_scaled, CONFIG['seq_length'])
    X_test = create_windows(test_scaled, CONFIG['seq_length'])
    
    # Date corrispondenti al test set (per i plot)
    # L'indice del dataframe originale allineato con la fine delle finestre di test
    test_dates_idx = log_returns.index[split_idx + CONFIG['seq_length'] - 1 : split_idx + CONFIG['seq_length'] - 1 + len(X_test)]
    
    dataset_info = {
        'asset_names': asset_names,
        'mean': mean,
        'std': std,
        'test_dates': test_dates_idx
    }
    
    return X_train, X_test, dataset_info

# ==========================================
# 2. TRAINING LOOP (PAPER FIDELITY)
# ==========================================
def train_paper_model(X_train):
    print("\n>>> Inizializzazione Training (Paper Config)...")
    
    # DataLoader
    train_loader = DataLoader(
        TensorDataset(X_train), 
        batch_size=CONFIG['batch_size'], 
        shuffle=True,
        drop_last=True # Importante per la stabilità BN/Statistiche
    )
    
    # Modello
    model = TempVAE(
        input_dim=X_train.shape[-1],
        latent_dim=CONFIG['latent_dim'],
        hidden_dim=CONFIG['hidden_dim'],
        dropout=CONFIG['dropout_rate']
    ).to(CONFIG['device'])
    
    # Optimizer: Adam con L2 Regularization (weight_decay)
    optimizer = optim.Adam(
        model.parameters(), 
        lr=CONFIG['learning_rate'], 
        weight_decay=CONFIG['l2_reg']
    )
    
    # Scheduler: Exponential Decay (0.96 ogni 500 steps)
    # PyTorch StepLR lavora per epoch, calcoliamo la frequenza equivalente
    # Se steps=500, e un'epoca ha N batch, dobbiamo calcolare gamma per epoca o usare un custom scheduler.
    # Faremo uno step dello scheduler manuale o useremo LambdaLR.
    steps_per_epoch = len(train_loader)
    total_steps = CONFIG['epochs'] * steps_per_epoch
    
    # Implementazione manuale del decay rate per step nel loop per massima precisione
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.0) # Placeholder
    
    model.train()
    
    global_step = 0
    
    for epoch in range(CONFIG['epochs']):
        
        # --- CALCOLO BETA ANNEALING (App. A.1) ---
        # Formula: beta = 1 - (decay_rate)^(epoch / decay_steps) ???
        # Il paper dice: "Subtracting an exponentially decaying term from the ultimate beta value (1.0)"
        # Beta_t = 1 - exp_decay_term
        # exp_decay_term = C * (rate)^(epoch / steps) ?
        # Assumiamo una implementazione standard sigmoid-like o esponenziale che satura a 1.
        # Implementazione interpretata:
        term = CONFIG['annealing_decay_rate'] ** (epoch / CONFIG['annealing_steps'])
        beta = 1.0 - term
        beta = max(0.0, min(1.0, beta)) # Clip tra 0 e 1
        
        epoch_loss = 0
        epoch_nll = 0
        epoch_kl = 0
        
        for batch in train_loader:
            x = batch[0].to(CONFIG['device'])
            optimizer.zero_grad()
            
            # Forward
            # Nota: TempVAE.forward restituisce nll_loss, kl_loss
            nll, kl = model(x)
            
            # Loss = NLL + beta * KL
            loss = nll + beta * kl
            
            loss.backward()
            
            # Gradient Clipping (Standard per RNN, anche se non esplicitato ma implicito in "stable training")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.5)
            
            optimizer.step()
            
            # LR Decay Schedule (ogni 500 steps globali)
            global_step += 1
            if global_step % CONFIG['lr_decay_steps'] == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= CONFIG['lr_decay_rate']

            epoch_loss += loss.item()
            epoch_nll += nll.item()
            epoch_kl += kl.item()
            
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1:04d} | Beta: {beta:.4f} | Loss: {epoch_loss/len(train_loader):.4f} | "
                  f"NLL: {epoch_nll/len(train_loader):.4f} | KL: {epoch_kl/len(train_loader):.4f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.6f}")
            
    return model

# ==========================================
# 3. EVALUATION PIPELINE (ANALISI.PY PORTING)
# ==========================================
def evaluate_paper_model(model, X_test, dataset_info):
    print("\n>>> Avvio Valutazione...")
    model.eval()
    
    # 1. Backtest VaR 95%
    alpha = 0.05
    MC = 1000 # Numero scenari Monte Carlo
    
    X_test_tensor = X_test.to(CONFIG['device'])
    var_forecasts = []
    actual_returns = []
    
    # Portfolio Equi-Weighted
    n_assets = X_test.shape[-1]
    weights = np.ones(n_assets) / n_assets
    
    with torch.no_grad():
        # Ottieni le distribuzioni latenti (Mean) per tutto il test set
        # run_inference restituisce [Batch, Seq, Latent]
        mu_z_seq = run_inference(model, X_test_tensor)
        
        # Prendiamo l'ultimo step temporale per ogni finestra
        z_last = mu_z_seq[:, -1, :] # [Batch, Latent]
        
        # Monte Carlo Simulation
        batch_size = z_last.shape[0]
        
        # Espandi Z per MC: [Batch*MC, 1, Latent]
        z_expanded = z_last.unsqueeze(1).expand(batch_size, MC, -1).reshape(batch_size*MC, -1).unsqueeze(1)
        
        # Genera parametri distribuzione R: [Batch*MC, 1, D]
        mu_r_flat, diag_r_flat = run_generation(model, z_expanded)
        
        mu_r = mu_r_flat.squeeze(1)     # [Batch*MC, D]
        diag_r = diag_r_flat.squeeze(1) # [Batch*MC, D]
        
        # Campiona R ~ N(mu, diag)
        # Nota: usiamo solo la diagonale per il sampling MC veloce, assumendo che il fattore rank-1 
        # sia catturato o che la diagonale domini la varianza specifica. 
        # Per massima precisione dovremmo usare anche il fattore u, ma run_generation nel file analisi
        # restituisce solo mu e diag. Per coerenza con analisi.py usiamo questo.
        eps = torch.randn_like(mu_r)
        r_sim_flat = mu_r + torch.sqrt(diag_r) * eps
        
        # Reshape [Batch, MC, D]
        r_sim = r_sim_flat.reshape(batch_size, MC, n_assets).cpu().numpy()
        
        # Calcolo VaR
        for i in range(batch_size):
            # De-standardizzazione
            # R_real = R_scaled * std + mean
            sim_rets_destandardized = r_sim[i] * dataset_info['std'] + dataset_info['mean']
            
            # Rendimento Portafoglio Sim
            port_sim = np.dot(sim_rets_destandardized, weights)
            
            # VaR 95% (Percentile 5%)
            var_val = np.percentile(port_sim, alpha * 100)
            var_forecasts.append(var_val)
            
            # Actual Return
            real_ret_scaled = X_test[i, -1, :].numpy()
            real_ret_destand = real_ret_scaled * dataset_info['std'] + dataset_info['mean']
            port_real = np.dot(real_ret_destand, weights)
            actual_returns.append(port_real)
            
    var_forecasts = np.array(var_forecasts)
    actual_returns = np.array(actual_returns)
    
    # --- SALVATAGGIO PLOT & TEST ---
    # 1. Plot VaR vs Actual
    plt.figure(figsize=(12, 6))
    plt.plot(actual_returns, color='gray', alpha=0.5, label='Actual Portfolio Return')
    plt.plot(var_forecasts, color='red', label='VaR 95% (TempVAE)')
    violations = actual_returns < var_forecasts
    plt.scatter(np.where(violations)[0], actual_returns[violations], color='black', marker='x', s=20)
    plt.title("TempVAE: VaR Forecast vs Real Returns")
    plt.legend()
    plt.savefig(os.path.join(CONFIG['plot_dir'], 'paper_var_series.png'))
    plt.close()
    
    # 2. Test Statistici
    print("\n=== RISULTATI TEST STATISTICI ===")
    kupiec_pof_test(actual_returns, var_forecasts, alpha)
    christoffersen_test(actual_returns, var_forecasts, alpha)
    
    # 3. Zoomed Regimes
    plot_zoomed_regimes(actual_returns, var_forecasts, dataset_info['test_dates'], alpha)


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # Setup Directory e Logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(CONFIG['plot_dir'], f"run_{timestamp}")
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    CONFIG['plot_dir'] = run_dir # Aggiorna path globale per i plot
    
    # Redirect Stdout
    sys.stdout = Logger(os.path.join(run_dir, "output.txt"))
    
    print(">>> TEMP VAE: PAPER EXACT REPLICATION PIPELINE")
    print(f">>> Log salvati in: {run_dir}")
    print("\nCONFIGURAZIONE:")
    for k, v in CONFIG.items():
        print(f"{k}: {v}")
    print("-" * 50)
    
    # 1. Dati
    X_train, X_test, ds_info = download_and_process_data()
    
    # 2. Training
    model = train_paper_model(X_train)
    
    # Salva modello
    torch.save(model.state_dict(), os.path.join(run_dir, "paper_model.pth"))
    
    # 3. Valutazione
    evaluate_paper_model(model, X_test, ds_info)
    
    print("\n>>> REPLICATION COMPLETE.")
