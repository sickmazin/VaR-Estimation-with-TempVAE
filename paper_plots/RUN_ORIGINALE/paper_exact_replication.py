import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2
from sklearn.decomposition import PCA
import os
from datetime import datetime
from TempVae import TempVAE # Import from TempVae.py

# ==========================================
# LOCAL IMPLEMENTATION OF INFERENCE UTILS
# ==========================================
def run_inference(model, x):
    """
    Extracts latent variables (mu) from TempVAE (Recurrent).
    Replicates the loop logic from TempVAE.forward to handle 
    context shifting and autoregressive Z dependency.
    """
    model.eval()
    batch_size, seq_len, _ = x.size()
    device = x.device
    
    with torch.no_grad():
        # --- 1. Bi-RNN Context ---
        # Output shape: [Batch, Seq, 2*Hidden] (e.g. 32)
        bi_rnn_out_raw, _ = model.inf_bi_rnn(x)
        
        fwd_out = bi_rnn_out_raw[:, :, :model.hidden_dim]
        bwd_out = bi_rnn_out_raw[:, :, model.hidden_dim:]

        # --- 2. Shifting Logic (Eq 15-16) ---
        zeros_fwd = torch.zeros(batch_size, 1, model.hidden_dim).to(device)
        h_arrow_right = torch.cat([zeros_fwd, fwd_out[:, :-1, :]], dim=1)

        zeros_bwd = torch.zeros(batch_size, 1, model.hidden_dim).to(device)
        h_arrow_left = torch.cat([bwd_out[:, 1:, :], zeros_bwd], dim=1)

        context_shifted = torch.cat([h_arrow_right, h_arrow_left], dim=2)

        # --- 3. Recurrent Inference Loop ---
        h_z = torch.zeros(batch_size, model.hidden_dim).to(device)
        z_prev = torch.zeros(batch_size, model.latent_dim).to(device)
        
        mu_z_list = []
        
        for t in range(seq_len):
            current_context = context_shifted[:, t, :] # [Batch, 2*Hidden]

            # Input to RNN_z is concatenation of prev Z and context
            rnn_input_inf = torch.cat([z_prev, current_context], dim=1)
            
            # RNN Cell update
            h_z = model.inf_rnn_z(rnn_input_inf, h_z)
            
            # MLP Inference -> q(Z_t)
            # inf_mlp returns (mu, sigma)
            q_mu, _ = model.inf_mlp(h_z)
            
            mu_z_list.append(q_mu)
            
            # Use mean for next step (deterministic inference path)
            z_prev = q_mu 
            
        return torch.stack(mu_z_list, dim=1)

def run_generation(model, z_seq):
    """
    Generates reconstruction parameters from Z sequence.
    Handles GRUCell state updates properly.
    """
    model.eval()
    
    # Handle case where input is [Batch, Latent] -> [Batch, 1, Latent]
    if z_seq.dim() == 2:
        z_seq = z_seq.unsqueeze(1)
        
    batch_size, seq_len, _ = z_seq.size()
    device = z_seq.device
    
    with torch.no_grad():
        h_gen_r = torch.zeros(batch_size, model.hidden_dim).to(device)
        
        mu_r_list = []
        diag_r_list = []
        
        for t in range(seq_len):
            z_t = z_seq[:, t, :]
            
            # Recurrent update
            h_gen_r = model.gen_rnn_r(z_t, h_gen_r)

            # Emission Parameters
            r_params = model.gen_mlp_r(h_gen_r)

            r_mu = r_params[:, :model.input_dim]
            r_diag_log = r_params[:, model.input_dim : 2*model.input_dim]
            r_diag = torch.exp(r_diag_log) + 1e-4
            
            mu_r_list.append(r_mu)
            diag_r_list.append(r_diag)
            
        return torch.stack(mu_r_list, dim=1), torch.stack(diag_r_list, dim=1)

# ==========================================
# CONFIGURAZIONE (Hyperparams from Paper)
# ==========================================
CONFIG = {
    'seq_length': 21,         # Finestra temporale M
    'latent_dim': 10,         # K=10 dimensioni latenti
    'hidden_dim': 16,         # H=32
    'batch_size': 256,        
    'epochs': 1000,
    'annealing_epochs': 100,
    'learning_rate': 1e-3,
    'l2_reg': 0.01,
    'data_path': 'dataset/log_returns.csv',
    'plot_dir': 'paper_plots',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': 42,
    'train_split': 0.66,
    'dropout_rate': 0.1      # Dropout 10% (Paper calibration)
}

torch.manual_seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])

if not os.path.exists(CONFIG['plot_dir']):
    os.makedirs(CONFIG['plot_dir'])

# ==========================================
# 1. DATA PREPROCESSING
# ==========================================
class FinancialDataset:
    def __init__(self, filepath, seq_len):
        self.df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        self.df = self.df.dropna()
        self.asset_names = self.df.columns.tolist()
        self.data = self.df.values.astype(np.float32)
        
        split_idx = int(len(self.data) * CONFIG['train_split'])
        self.train_raw = self.data[:split_idx]
        self.test_raw = self.data[split_idx:]
        
        self.mean = np.mean(self.train_raw, axis=0)
        self.std = np.std(self.train_raw, axis=0)
        self.std[self.std == 0] = 1.0 
        
        self.train_scaled = (self.train_raw - self.mean) / self.std
        self.test_scaled = (self.test_raw - self.mean) / self.std
        
        self.X_train = self._create_windows(self.train_scaled, seq_len)
        self.X_test = self._create_windows(self.test_scaled, seq_len)

        self.test_dates = self.df.index[split_idx + seq_len - 1 : split_idx + seq_len - 1 + len(self.X_test)]
        
    def _create_windows(self, data, seq_len):
        windows = []
        for i in range(len(data) - seq_len + 1):
            windows.append(data[i:i+seq_len])
        return torch.tensor(np.array(windows), dtype=torch.float32)


# ==========================================
# 3. TRAINING
# ==========================================
def train_tempvae():
    print(">>> 1. Loading Dataset...")
    dataset = FinancialDataset(CONFIG['data_path'], CONFIG['seq_length'])

    print("-" * 50)
    print(f"Data Analysis Start Date: {dataset.df.index[0].date()}")
    print(f"Data Analysis End Date:   {dataset.df.index[-1].date()}")
    print("-" * 50)

    # Nel dataset o prima del DataLoader
    if not torch.is_tensor(dataset.X_train):
        dataset.X_train = torch.tensor(dataset.X_train, dtype=torch.float32)

    # IMPORTANTE: drop_last=True evita batch incompleti che possono destabilizzare
    # le statistiche della Batch Normalization (se la usassi) o i calcoli della media
    train_loader = DataLoader(
        TensorDataset(dataset.X_train),
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        drop_last=True
    )

    # Istanziazione Modello
    model = TempVAE(
        input_dim=dataset.X_train.shape[-1],
        latent_dim=CONFIG['latent_dim'],
        hidden_dim=CONFIG['hidden_dim'],
        dropout=CONFIG['dropout_rate']
    ).to(CONFIG['device'])

    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['l2_reg'])

    print(">>> 2. Starting Training Loop...")
    model.train()

    for epoch in range(CONFIG['epochs']):
        # SICUREZZA 1: Gestione annealing sicuro
        if CONFIG['annealing_epochs'] > 0:
            beta = min(1.0, epoch / CONFIG['annealing_epochs'])
        else:
            beta = 1.0 # Nessun annealing, KL piena fin da subito

        epoch_loss = 0
        epoch_nll = 0
        epoch_kl = 0

        for batch_idx, batch in enumerate(train_loader):
            x = batch[0].to(CONFIG['device'])

            optimizer.zero_grad()

            # Forward
            nll_loss, kl_loss = model(x)

            # Calcolo Loss Totale
            loss = nll_loss + beta * kl_loss

            # SICUREZZA 2: Controllo NaN immediato
            if torch.isnan(loss):
                print(f"!!! CRITICAL ERROR: Loss is NaN at Epoch {epoch}, Batch {batch_idx}")
                print(f"NLL: {nll_loss.item()}, KL: {kl_loss.item()}")
                return model, dataset # Esci subito per debuggare

            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.5)

            optimizer.step()

            epoch_loss += loss.item()
            epoch_nll += nll_loss.item()
            epoch_kl += kl_loss.item()

        # Logging
        if (epoch+1) % 10 == 0:
            avg_loss = epoch_loss / len(train_loader)
            avg_nll = epoch_nll / len(train_loader)
            avg_kl = epoch_kl / len(train_loader)
            print(f"Epoch {epoch+1:04d} | Beta: {beta:.2f} | Loss: {avg_loss:.4f} | NLL: {avg_nll:.4f} | KL: {avg_kl:.4f}")

    return model, dataset

# ==========================================
# 4. COMPREHENSIVE STATISTICAL ANALYSIS
# ==========================================

def kupiec_pof_test(actual_returns, var_forecasts, alpha=0.05):
    """
    Esegue il test di Kupiec (Proportion of Failures).
    """
    print("\n--- Kupiec POF Test ---")
    violations = actual_returns < var_forecasts
    N = np.sum(violations)
    T = len(actual_returns)
    pi_obs = N / T
    pi_exp = alpha

    print(f"Totale Giorni: {T}")
    print(f"Violazioni Attese: {T * pi_exp:.1f}")
    print(f"Violazioni Osservate: {N}")
    print(f"Frequenza Osservata: {pi_obs:.4f} (Target: {pi_exp:.4f})")

    if N == 0:
        print("Zero violazioni! Il modello è troppo conservativo.")
        return

    numerator = (pi_exp ** N) * ((1 - pi_exp) ** (T - N))
    denominator = (pi_obs ** N) * ((1 - pi_obs) ** (T - N))
    lr_stat = -2 * np.log(numerator / denominator)
    p_value = 1 - chi2.cdf(lr_stat, 1)

    print(f"LR Statistic: {lr_stat:.4f} | P-Value: {p_value:.4f}")
    if p_value < 0.05:
        print("RISULTATO: RIGETTO H0. Il modello NON è calibrato correttamente.")
    else:
        print("RISULTATO: NON RIGETTO H0. Il modello è statisticamente Valido.")

def christoffersen_test(actual_returns, var_forecasts, alpha=0.05):
    """
    Esegue il test di Christoffersen (Conditional Coverage).
    """
    print("\n--- Christoffersen Conditional Coverage Test ---")
    hits = (actual_returns < var_forecasts).astype(int)
    hits_curr = hits[:-1]
    hits_next = hits[1:]

    n00 = np.sum((hits_curr == 0) & (hits_next == 0))
    n01 = np.sum((hits_curr == 0) & (hits_next == 1))
    n10 = np.sum((hits_curr == 1) & (hits_next == 0))
    n11 = np.sum((hits_curr == 1) & (hits_next == 1))

    pi_0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0
    pi_1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    def safe_log(x): return np.log(x) if x > 0 else 0
    logL_H0 = (n00 + n10) * safe_log(1 - pi) + (n01 + n11) * safe_log(pi)
    logL_H1 = (n00 * safe_log(1 - pi_0) + n01 * safe_log(pi_0) +
               n10 * safe_log(1 - pi_1) + n11 * safe_log(pi_1))

    lr_ind = -2 * (logL_H0 - logL_H1)
    p_val_ind = 1 - chi2.cdf(lr_ind, 1)

    print(f"Transizioni: n00={n00}, n01={n01}, n10={n10}, n11={n11}")
    print(f"LR Independence: {lr_ind:.4f} (p-value: {p_val_ind:.4f})")

    # UC part (Kupiec)
    T = len(hits)
    N = np.sum(hits)
    p_exp = alpha
    p_obs = N / T
    lr_uc = -2 * ((T-N)*safe_log(1-p_exp) + N*safe_log(p_exp) - ((T-N)*safe_log(1-p_obs) + N*safe_log(p_obs)))

    lr_cc = lr_uc + lr_ind
    p_val_cc = 1 - chi2.cdf(lr_cc, 2)

    print(f"LR Conditional Coverage (CC): {lr_cc:.4f} | P-Value CC: {p_val_cc:.4f}")
    if p_val_cc < 0.05:
        print("RISULTATO: RIGETTO H0. Il modello fallisce il test condizionale.")
    else:
        print("RISULTATO: NON RIGETTO H0. Il modello è Valido Condizionalmente.")

def correlation_analysis(Z, dataset, active_indices):
    """
    Correlazione tra Latenti Attivi e Rendimenti/Volatilità Asset
    """
    if len(active_indices) == 0:
        print("Skipping Correlation Analysis: No active units.")
        return

    real_returns = dataset.X_test[:, -1, :].cpu().numpy()
    
    # DataFrame Costruzione
    df_dict = {}
    for idx in active_indices:
        df_dict[f'Z_{idx}'] = Z[:, idx]
        
    for i, asset in enumerate(dataset.asset_names):
        df_dict[f'{asset}_Ret'] = real_returns[:, i]
        df_dict[f'{asset}_Vol'] = real_returns[:, i]**2 
        
    df = pd.DataFrame(df_dict)
    
    # Filter Correlation Matrix
    z_cols = [f'Z_{idx}' for idx in active_indices]
    asset_cols = [c for c in df.columns if c not in z_cols]
    
    corr = df.corr().loc[z_cols, asset_cols]
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt=".2f", linewidths=0.5)
    plt.title("Latent Space Semantic Map: Correlation with Returns & Volatility")
    plt.tight_layout()
    plt.savefig(f"{CONFIG['plot_dir']}/latent_heatmap.png")
    print(f"Correlation Heatmap saved to {CONFIG['plot_dir']}/latent_heatmap.png")

def analyze_latent_manifold(model, dataset):
    """
    PCA Projection colored by Volatility (Standard VAE Analysis)
    """
    model.eval()
    X_test = dataset.X_test.to(CONFIG['device'])
    
    # Use inference helper
    mu_z_seq = run_inference(model, X_test)
        
    # Extract Latents (N, Seq, K) -> (N, K) last step
    mu_z = mu_z_seq[:, -1, :].cpu().numpy()
    
    # Calculate Portfolio Volatility for coloring
    # Using realized volatility of the window (standard deviation of returns in window)
    realized_vol = torch.std(dataset.X_test, dim=1).mean(dim=1).numpy() # Avg std across assets
    log_vol = np.log(realized_vol + 1e-9)
    
    # PCA
    pca = PCA(n_components=2)
    z_pca = pca.fit_transform(mu_z)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(z_pca[:, 0], z_pca[:, 1], c=log_vol, cmap='magma', alpha=0.7, s=20)
    plt.colorbar(scatter, label='Log Realized Volatility')
    plt.title("Latent Manifold (PCA) colored by Market Volatility")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{CONFIG['plot_dir']}/latent_manifold.png")
    
    # Active Units Plot (Recalculated here for completeness)
    vars = np.var(mu_z, axis=0)
    plt.figure(figsize=(8,4))
    plt.bar(range(CONFIG['latent_dim']), vars)
    plt.axhline(0.01, color='r', linestyle='--')
    plt.title("Active Units Variance")
    plt.savefig(f"{CONFIG['plot_dir']}/active_units.png")
    
    # Call Correlation Analysis
    active_indices = np.where(vars > 0.01)[0]
    correlation_analysis(mu_z, dataset, active_indices)
    
    return mu_z

def plot_zoomed_regimes(actual, var, dates, alpha):
    """
    Plots VaR vs Actual Returns for specific stress periods:
    2. Bear Market Onset (2022)
    3. Recent (2024)
    """
    import matplotlib.dates as mdates
    
    periods = [
        ("Bear Market Onset (2022)", "2022-01-01", "2022-06-30"),
        ("Recent Period (2024)", "2024-01-01", "2024-12-31")
    ]
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    plt.subplots_adjust(hspace=0.4)
    
    # Ensure dates are datetime
    dates = pd.to_datetime(dates)
    
    for ax, (title, start, end) in zip(axes, periods):
        mask = (dates >= start) & (dates <= end)
        
        if not any(mask):
            ax.text(0.5, 0.5, "No Data for this Period", ha='center', transform=ax.transAxes)
            ax.set_title(title)
            continue
            
        dates_sub = dates[mask]
        actual_sub = actual[mask]
        var_sub = var[mask]
        
        ax.plot(dates_sub, actual_sub, color='grey', alpha=0.6, label='Portfolio Return', linewidth=1)
        ax.plot(dates_sub, var_sub, color='red', linewidth=1.5, label=f'VaR {1-alpha:.0%}')
        
        # Violations
        violations_idx = np.where(actual_sub < var_sub)[0]
        if len(violations_idx) > 0:
            ax.scatter(dates_sub[violations_idx], actual_sub[violations_idx], color='black', marker='x', s=40, zorder=5, label='Violation')
            
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        if ax == axes[0]: ax.legend(loc='upper right')
        
    plt.suptitle(f"Stress Testing VaR: Multi-Regime Analysis", fontsize=16)
    plt.savefig(f"{CONFIG['plot_dir']}/var_regimes_plot.png")
    print(f"Grafico regimi salvato in '{CONFIG['plot_dir']}/var_regimes_plot.png'")

def backtest_pipeline(model, dataset, alpha=0.05):
    model.eval()
    weights = np.ones(len(dataset.asset_names)) / len(dataset.asset_names)
    X_test = dataset.X_test.to(CONFIG['device'])
    var_forecasts, actual_returns = [], []
    MC = 1000
    
    print(f"\n>>> Generating {MC} Monte Carlo Scenarios for VaR...")
    with torch.no_grad():
        # 1. Inference to get Z distribution
        mu_z_seq = run_inference(model, X_test)
        
        # We assume sigma is fixed/small or we rely on mu for the mean trajectory.
        # But for Monte Carlo VaR, we need stochasticity.
        # TempVAE inference mlp outputs mu and logvar. run_inference currently returns only mu.
        # I should update run_inference to return logvar too if I want full MC on Z.
        # However, for simplicity given the constraints, let's assume we sample around mu_z
        # using the Generative model's stochasticity (sigma_r) primarily.
        
        # Get Z at last step T
        z_last = mu_z_seq[:, -1, :] # (B, Latent)
        
        # 2. Monte Carlo Sampling for R
        # We need to pass Z through decoder.
        # Ideally we should simulate the generative process p(R|Z).
        # We simply repeat z_last for MC samples.
        
        B = z_last.shape[0]
        z_expanded = z_last.unsqueeze(1).expand(B, MC, -1).reshape(B*MC, -1).unsqueeze(1) # (B*MC, 1, Latent)
        
        # Generate R parameters (mu, diag)
        # Note: run_generation expects a sequence. We give length 1.
        mu_r_flat, diag_r_flat = run_generation(model, z_expanded)
        
        mu_r = mu_r_flat.squeeze(1)
        diag_r = diag_r_flat.squeeze(1)
        
        # Sample R ~ N(mu, diag)
        eps = torch.randn_like(mu_r)
        r_sim = mu_r + torch.sqrt(diag_r) * eps
        
        # Reshape to (B, MC, D)
        r_sim = r_sim.reshape(B, MC, -1)
        r_sim = r_sim.cpu().numpy()
        
        for t in range(len(X_test)):
            # Destandardize
            sim_rets = r_sim[t] * dataset.std + dataset.mean
            sim_port = np.dot(sim_rets, weights)
            
            var_forecasts.append(np.percentile(sim_port, alpha*100))
            
            real_ret = np.dot(X_test[t, -1].cpu().numpy() * dataset.std + dataset.mean, weights)
            actual_returns.append(real_ret)
            
    var_forecasts = np.array(var_forecasts)
    actual_returns = np.array(actual_returns)
    
    # Plot Time Series
    plt.figure(figsize=(12, 6))
    plt.plot(actual_returns, color='gray', alpha=0.5, label='Actual')
    plt.plot(var_forecasts, color='red', label='VaR')
    violations = actual_returns < var_forecasts
    plt.scatter(np.where(violations)[0], actual_returns[violations], color='black', marker='x')
    plt.title("VaR Forecast vs Actual Returns")
    plt.savefig(f"{CONFIG['plot_dir']}/var_series.png")
    
    # EXECUTE ALL STATISTICAL TESTS
    kupiec_pof_test(actual_returns, var_forecasts, alpha)
    christoffersen_test(actual_returns, var_forecasts, alpha)
    
    # Plot Zoomed Regimes
    plot_zoomed_regimes(actual_returns, var_forecasts, dataset.test_dates, alpha)

if __name__ == "__main__":
    # Creazione sottocartella basata su timestamp
    base_plot_dir = CONFIG['plot_dir']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_plot_dir, f"run_{timestamp}")
    
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
        print(f">>> Risultati di questa esecuzione in: {run_dir}")
    
    # Aggiorniamo il CONFIG in modo che tutte le funzioni puntino alla nuova cartella
    CONFIG['plot_dir'] = run_dir

    model, dataset = train_tempvae()
    _ = analyze_latent_manifold(model, dataset)
    backtest_pipeline(model, dataset, alpha=0.05)
    print("\n>>> Analysis Complete.")
