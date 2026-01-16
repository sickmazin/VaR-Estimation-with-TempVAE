import os
import pandas as pd
import seaborn as sns
from scipy.stats import chi2
from sklearn.decomposition import PCA
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ==========================================
# EXTENSION: PIPELINE DI ANALISI (PAPER REPLICATION)
# ==========================================
import numpy as np
import torch
from matplotlib import pyplot as plt

from analisi import CONFIG


# Dataset
class FinancialDataset:
    def __init__(self, filepath, seq_len):
        self.df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        self.df = self.df.fillna(0.0)
        
        if CONFIG.get('selected_assets') is not None:
            self.df = self.df[CONFIG['selected_assets']]
            
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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence


class MLP(nn.Module):
    """Multi-Layer Perceptron con 2 hidden layers"""
    def __init__(self, input_dim, output_dim, hidden_dim=16, dropout_rate=0.0):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

        # Variance Scaling initialization (He initialization)
        for layer in [self.fc1, self.fc2, self.fc_out]:
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc_out(x)
        return x


class CovarianceOutput(nn.Module):
    """Output layer per matrice di covarianza"""
    def __init__(self, input_dim, output_dim, covariance_type='diagonal'):
        super().__init__()
        self.output_dim = output_dim
        self.covariance_type = covariance_type

        if covariance_type == 'diagonal':
            self.diag_layer = nn.Linear(input_dim, output_dim)
        elif covariance_type == 'rank1':
            # Rank-1 perturbation: diag + vv^T
            self.diag_layer = nn.Linear(input_dim, output_dim)
            self.v_layer = nn.Linear(input_dim, output_dim)

        # Inizializzazione
        nn.init.kaiming_normal_(self.diag_layer.weight, nonlinearity='relu')
        nn.init.zeros_(self.diag_layer.bias)
        if covariance_type == 'rank1':
            nn.init.kaiming_normal_(self.v_layer.weight, nonlinearity='relu')
            nn.init.zeros_(self.v_layer.bias)

    def forward(self, x):
        # Exponential activation per elementi diagonali
        diag = torch.exp(self.diag_layer(x))

        if self.covariance_type == 'diagonal':
            return diag
        elif self.covariance_type == 'rank1':
            v = self.v_layer(x)
            return diag, v


class GenerativeModel(nn.Module):
    """Modello Generativo: p_θ(Z) e p_θ(R|Z)"""
    def __init__(self, latent_dim=10, return_dim=1, hidden_dim=16, dropout_rate=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.return_dim = return_dim
        self.hidden_dim = hidden_dim

        # RNN per Z (prior)
        self.rnn_z = nn.GRU(latent_dim, hidden_dim, batch_first=True, dropout=dropout_rate)

        # MLP per parametri di Z
        self.mlp_mu_z = MLP(hidden_dim, latent_dim, hidden_dim)
        self.mlp_sigma_z = MLP(hidden_dim, latent_dim, hidden_dim)

        # RNN per R (likelihood)
        self.rnn_r = nn.GRU(latent_dim, hidden_dim, batch_first=True, dropout=dropout_rate)

        # MLP per parametri di R
        self.mlp_mu_r = MLP(hidden_dim, return_dim, hidden_dim)
        self.cov_r = CovarianceOutput(hidden_dim, return_dim, covariance_type='rank1')

    def forward_z(self, z_prev, h_z):
        """
        Equations (5), (6), (7):
        - h^z_t = RNN_G^z(h^z_{t-1}, Z_{t-1})
        - {μ^z_t, Σ^z_t} = MLP_G^z(h^z_t)
        """
        # RNN step
        _, h_z_new = self.rnn_z(z_prev.unsqueeze(1), h_z)
        h_z_out = h_z_new.squeeze(0)

        # MLP per μ e Σ
        mu_z = self.mlp_mu_z(h_z_out)
        log_sigma_z = self.mlp_sigma_z(h_z_out)
        sigma_z = torch.exp(log_sigma_z)

        return mu_z, sigma_z, h_z_new

    def forward_r(self, z_t, h_r):
        """
        Equations (8), (9), (10):
        - h^r_t = RNN_G^r(h^r_{t-1}, Z_t)
        - {μ^r_t, Σ^r_t} = MLP_G^r(h^r_t)
        """
        # RNN step
        _, h_r_new = self.rnn_r(z_t.unsqueeze(1), h_r)
        h_r_out = h_r_new.squeeze(0)

        # MLP per μ e Σ
        mu_r = self.mlp_mu_r(h_r_out)
        diag_r, v_r = self.cov_r(h_r_out)

        return mu_r, diag_r, v_r, h_r_new


class InferenceModel(nn.Module):
    """Modello di Inferenza: q_φ(Z|R)"""
    def __init__(self, latent_dim=10, return_dim=1, hidden_dim=16, dropout_rate=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.return_dim = return_dim
        self.hidden_dim = hidden_dim

        # Bidirectional RNN per R
        self.rnn_forward = nn.GRU(return_dim, hidden_dim, batch_first=True, dropout=dropout_rate)
        self.rnn_backward = nn.GRU(return_dim, hidden_dim, batch_first=True, dropout=dropout_rate)

        # RNN per Z con input concatenato
        self.rnn_z = nn.GRU(latent_dim + 2*hidden_dim, hidden_dim, batch_first=True, dropout=dropout_rate)

        # MLP per parametri di Z
        self.mlp_mu_z = MLP(hidden_dim, latent_dim, hidden_dim)
        self.mlp_sigma_z = MLP(hidden_dim, latent_dim, hidden_dim)

    def forward(self, R):
        """
        Equations (13), (14), (15), (16), (17):
        - Bidirectional RNN su R
        - RNN su Z con concatenazione
        """
        batch_size, seq_len, _ = R.shape

        # Forward e backward pass su R (equations 15, 16)
        h_forward, _ = self.rnn_forward(R)
        h_backward, _ = self.rnn_backward(torch.flip(R, [1]))
        h_backward = torch.flip(h_backward, [1])

        # Inizializzazione
        z_samples = []
        mu_z_list = []
        sigma_z_list = []

        z_t = torch.zeros(batch_size, self.latent_dim, device=R.device)
        h_z = torch.zeros(1, batch_size, self.hidden_dim, device=R.device)

        # Loop temporale
        for t in range(seq_len):
            # Concatena hidden states (equation 14)
            rnn_input = torch.cat([z_t, h_forward[:, t], h_backward[:, t]], dim=-1)
            _, h_z = self.rnn_z(rnn_input.unsqueeze(1), h_z)
            h_z_out = h_z.squeeze(0)

            # MLP per μ e Σ (equation 13)
            mu_z = self.mlp_mu_z(h_z_out)
            log_sigma_z = self.mlp_sigma_z(h_z_out)
            sigma_z = torch.exp(log_sigma_z)

            # Sampling (equation 17)
            dist = Normal(mu_z, sigma_z)
            z_t = dist.rsample()

            z_samples.append(z_t)
            mu_z_list.append(mu_z)
            sigma_z_list.append(sigma_z)

        z_samples = torch.stack(z_samples, dim=1)
        mu_z_list = torch.stack(mu_z_list, dim=1)
        sigma_z_list = torch.stack(sigma_z_list, dim=1)

        return z_samples, mu_z_list, sigma_z_list


class TempVAE(nn.Module):
    """
    Temporal Variational Autoencoder per Asset Returns
    Implementa il modello completo con generative e inference networks
    """
    def __init__(self, latent_dim=10, return_dim=1, hidden_dim=16,
                 dropout_rate=0.1, l2_lambda=0.01):
        super().__init__()
        self.latent_dim = latent_dim
        self.return_dim = return_dim
        self.l2_lambda = l2_lambda

        self.generative = GenerativeModel(latent_dim, return_dim, hidden_dim, dropout_rate)
        self.inference = InferenceModel(latent_dim, return_dim, hidden_dim, dropout_rate)

    def forward(self, R):
        """
        Forward pass: calcola ELBO

        Args:
            R: tensor di shape (batch_size, seq_len, return_dim)

        Returns:
            elbo, reconstruction_loss, kl_loss
        """
        batch_size, seq_len, _ = R.shape
        device = R.device

        # Inference: q_φ(Z|R)
        z_samples, q_mu_z, q_sigma_z = self.inference(R)

        # Generative model
        recon_loss = 0
        kl_loss = 0

        z_t = torch.zeros(batch_size, self.latent_dim, device=device)
        h_z = torch.zeros(1, batch_size, self.generative.hidden_dim, device=device)
        h_r = torch.zeros(1, batch_size, self.generative.hidden_dim, device=device)

        for t in range(seq_len):
            # Prior p_θ(Z_t | Z_{1:t-1})
            p_mu_z, p_sigma_z, h_z = self.generative.forward_z(z_t, h_z)

            # Sample da q
            z_t = z_samples[:, t]

            # Likelihood p_θ(R_t | Z_{1:t})
            mu_r, diag_r, v_r, h_r = self.generative.forward_r(z_t, h_r)

            # Reconstruction loss (equation 18-19)
            # Rank-1 covariance: Σ = diag + vv^T
            sigma_r = diag_r + 1e-6  # Stabilità numerica
            dist_r = Normal(mu_r, torch.sqrt(sigma_r))
            recon_loss += -dist_r.log_prob(R[:, t]).sum(dim=-1).mean()

            # KL divergence (equation 20)
            q_dist = Normal(q_mu_z[:, t], q_sigma_z[:, t])
            p_dist = Normal(p_mu_z, p_sigma_z)
            kl = kl_divergence(q_dist, p_dist).sum(dim=-1).mean()
            kl_loss += kl

        # L2 regularization
        l2_reg = 0
        for name, param in self.named_parameters():
            if 'weight' in name and 'mlp' in name:
                l2_reg += torch.norm(param, p=2)
        l2_reg *= self.l2_lambda

        # ELBO (equation 19)
        elbo = -recon_loss - kl_loss - l2_reg

        return elbo, recon_loss, kl_loss

    def sample(self, seq_len, num_samples=1):
        """
        Genera campioni dal modello generativo

        Args:
            seq_len: lunghezza sequenza
            num_samples: numero di campioni

        Returns:
            R_samples: tensor (num_samples, seq_len, return_dim)
        """
        device = next(self.parameters()).device
        R_samples = []

        with torch.no_grad():
            z_t = torch.zeros(num_samples, self.latent_dim, device=device)
            h_z = torch.zeros(1, num_samples, self.generative.hidden_dim, device=device)
            h_r = torch.zeros(1, num_samples, self.generative.hidden_dim, device=device)

            for t in range(seq_len):
                # Sample Z_t
                mu_z, sigma_z, h_z = self.generative.forward_z(z_t, h_z)
                dist_z = Normal(mu_z, sigma_z)
                z_t = dist_z.sample()

                # Sample R_t
                mu_r, diag_r, v_r, h_r = self.generative.forward_r(z_t, h_r)
                sigma_r = torch.sqrt(diag_r + 1e-6)
                dist_r = Normal(mu_r, sigma_r)
                r_t = dist_r.sample()

                R_samples.append(r_t)

        return torch.stack(R_samples, dim=1)


# Stat Tests
def kupiec_pof_test(actual_returns, var_forecasts, alpha=0.05):
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

# Plots
def correlation_analysis(Z, dataset, active_indices):
    if len(active_indices) == 0: return

    real_returns = dataset.X_test[:, -1, :].cpu().numpy()
    
    df_dict = {}
    for idx in active_indices:
        df_dict[f'Z_{idx}'] = Z[:, idx]
        
    for i, asset in enumerate(dataset.asset_names):
        df_dict[f'{asset}_Ret'] = real_returns[:, i]
        df_dict[f'{asset}_Vol'] = real_returns[:, i]**2 
        
    df = pd.DataFrame(df_dict)
    
    z_cols = [f'Z_{idx}' for idx in active_indices]
    asset_cols = [c for c in df.columns if c not in z_cols]
    
    corr = df.corr().loc[z_cols, asset_cols]
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt=".2f", linewidths=0.5)
    plt.title("Latent Space Semantic Map")
    plt.tight_layout()
    plt.savefig(f"{CONFIG['plot_dir']}/latent_heatmap.png")
    print(f"Correlation Heatmap saved.")

def analyze_latent_manifold(model, dataset):
    model.eval()
    X_test = dataset.X_test.to(CONFIG['device'])
    
    # Extract Latents using Inference Model
    # model.inference returns (z_samples, mu_z_list, sigma_z_list)
    with torch.no_grad():
        _, mu_z_list, _ = model.inference(X_test)
        
    # Take last time step
    mu_z = mu_z_list[:, -1, :].cpu().numpy()
    
    # PCA
    realized_vol = torch.std(dataset.X_test, dim=1).mean(dim=1).numpy()
    log_vol = np.log(realized_vol + 1e-9)
    
    pca = PCA(n_components=2)
    z_pca = pca.fit_transform(mu_z)
    
    plt.figure(figsize=(10, 8))
    plt.scatter(z_pca[:, 0], z_pca[:, 1], c=log_vol, cmap='magma', alpha=0.7, s=20)
    plt.colorbar(label='Log Realized Volatility')
    plt.title("Latent Manifold (PCA)")
    plt.savefig(f"{CONFIG['plot_dir']}/latent_manifold.png")
    
    # Active Units
    vars = np.var(mu_z, axis=0)
    plt.figure(figsize=(8,4))
    plt.bar(range(CONFIG['latent_dim']), vars)
    plt.axhline(0.01, color='r', linestyle='--')
    plt.title("Active Units Variance")
    plt.savefig(f"{CONFIG['plot_dir']}/active_units.png")
    
    active_indices = np.where(vars > 0.01)[0]
    correlation_analysis(mu_z, dataset, active_indices)
    
    return mu_z

def plot_zoomed_regimes(actual, var, dates, alpha):
    import matplotlib.dates as mdates
    periods = [
        ("Covid Crash (2020)", "2020-01-01", "2020-06-30"),
        ("Bear Market Onset (2022)", "2022-01-01", "2022-06-30"),
        ("Recent Period (2024)", "2024-01-01", "2024-12-31")
    ]
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    plt.subplots_adjust(hspace=0.4)
    dates = pd.to_datetime(dates)
    
    for ax, (title, start, end) in zip(axes, periods):
        mask = (dates >= start) & (dates <= end)
        if not any(mask):
            ax.set_title(title + " (No Data)")
            continue
            
        dates_sub = dates[mask]
        actual_sub = actual[mask]
        var_sub = var[mask]
        
        ax.plot(dates_sub, actual_sub, color='grey', alpha=0.6)
        ax.plot(dates_sub, var_sub, color='red', linewidth=1.5)
        
        violations = actual_sub < var_sub
        if np.any(violations):
            ax.scatter(dates_sub[violations], actual_sub[violations], color='black', marker='x')
            
        ax.set_title(title)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
    plt.savefig(f"{CONFIG['plot_dir']}/var_regimes_plot.png")

def backtest_pipeline(model, dataset, alpha=0.05):
    model.eval()
    weights = np.ones(len(dataset.asset_names)) / len(dataset.asset_names)
    X_test = dataset.X_test.to(CONFIG['device'])
    var_forecasts, actual_returns = [], []
    MC = 1000 # Reduced for speed if needed
    
    print(f"\n>>> Generating Monte Carlo Scenarios for VaR...")
    
    with torch.no_grad():
        # 1. Get Latent Distribution q(Z_{1:T} | X)
        z_samples_seq, mu_z_seq, sigma_z_seq = model.inference(X_test)
        
        # We need the predictive distribution for the LAST step T
        # Or rather, we sample Z_T ~ q(Z_T | X) and then R_T ~ p(R_T | Z_T)?
        # Standard VaR is predictive. 
        # Here we do reconstruction VaR (In-Sample at time t) as in the roadmap code.
        
        # Get parameters for the last time step T
        mu_z = mu_z_seq[:, -1, :]      # (B, Latent)
        sigma_z = sigma_z_seq[:, -1, :]
        
        # 2. Monte Carlo Sampling
        # Sample Z ~ N(mu_z, sigma_z)
        # We need MC samples per batch item.
        # Shape: (B, MC, Latent)
        
        B = mu_z.shape[0]
        # Expand for MC
        mu_z_exp = mu_z.unsqueeze(1).expand(B, MC, -1)
        sigma_z_exp = sigma_z.unsqueeze(1).expand(B, MC, -1)
        
        eps = torch.randn_like(mu_z_exp)
        z_sim = mu_z_exp + sigma_z_exp * eps # (B, MC, Latent)
        
        # 3. Decode to Market Space
        # p(R | Z). We need to pass Z through Generative Model.
        # However, Generative Model is recurrent (h_r).
        # Approximating: We assume the hidden state h_r is determined by the trajectory.
        # BUT, for efficiency, can we just use the generative MLP if we ignore temporal dependency for the immediate step?
        # The model defines forward_r(z_t, h_r). h_r is stateful.
        # To do this correctly, we should have run the generative RNN over the sequence z_samples_seq to get h_r at T-1.
        
        # Let's run the generative RNN over the mean Z sequence to get context
        h_r = torch.zeros(1, B, model.generative.hidden_dim, device=CONFIG['device'])
        for t in range(CONFIG['seq_length'] - 1):
            z_t = mu_z_seq[:, t, :]
            _, _, _, h_r = model.generative.forward_r(z_t, h_r)
            
        # Now h_r contains context up to T-1.
        # Expand h_r for MC: (1, B*MC, H) -> This is tricky with RNN API.
        # Alternative: Loop over MC? Slow.
        # Alternative: Reshape batch.
        
        # Flatten B*MC
        z_sim_flat = z_sim.reshape(B*MC, -1)
        h_r_rep = h_r.repeat(1, MC, 1).reshape(1, B*MC, -1) # Repeat for MC
        
        # One step generation
        mu_r, diag_r, v_r, _ = model.generative.forward_r(z_sim_flat, h_r_rep)
        
        # Sample R ~ N(mu_r, Sigma_r)
        # Sigma = diag + vv^T? (The model code has a bug/simplification where it uses only diag in loss)
        # We will use what the model outputs.
        # Note: CovarianceOutput returns (diag, v) for rank1.
        
        # For simplicity/speed and given the loss bug, use diagonal for sampling or try rank1
        # eps_r = torch.randn_like(mu_r)
        # r_sim = mu_r + torch.sqrt(diag_r) * eps_r 
        
        # Correct Rank-1 Sampling:
        # R = mu + D^0.5 * eps1 + v * eps2
        eps1 = torch.randn_like(mu_r)
        eps2 = torch.randn(B*MC, 1, device=CONFIG['device'])
        
        r_sim = mu_r + torch.sqrt(diag_r) * eps1 + v_r * eps2
        
        # Reshape back to (B, MC, D)
        r_sim = r_sim.reshape(B, MC, -1)
        
        # 4. Calculate VaR per batch item
        r_sim = r_sim.cpu().numpy()
        
        for i in range(B):
            # Destandardize
            # Note: window i corresponds to dataset.std
            # Using global mean/std from dataset object
            sim_rets = r_sim[i] * dataset.std + dataset.mean
            
            # Portfolio
            sim_port = sim_rets.mean(axis=1) # Equi-weighted (mean across assets)
            
            var_forecasts.append(np.percentile(sim_port, alpha*100))
            
            # Actual Return
            real_ret_scaled = X_test[i, -1, :].cpu().numpy()
            real_ret = np.dot(real_ret_scaled * dataset.std + dataset.mean, weights)
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
    
    kupiec_pof_test(actual_returns, var_forecasts, alpha)
    christoffersen_test(actual_returns, var_forecasts, alpha)
    plot_zoomed_regimes(actual_returns, var_forecasts, dataset.test_dates, alpha)

# Training Wrapper
def train_model_wrapper():
    print(">>> 1. Loading Dataset...")
    dataset = FinancialDataset(CONFIG['data_path'], CONFIG['seq_length'])
    print(f"Data Analysis Start: {dataset.df.index[0]}")
    print(f"Data Analysis End:   {dataset.df.index[-1]}")
    
    train_loader = DataLoader(TensorDataset(dataset.X_train), batch_size=CONFIG['batch_size'], shuffle=True)
    
    input_dim = dataset.X_train.shape[-1]
    model = TempVAE(
        latent_dim=CONFIG['latent_dim'],
        return_dim=input_dim,
        hidden_dim=CONFIG['hidden_dim'],
        dropout_rate=CONFIG['dropout_rate'],
        l2_lambda=CONFIG['l2_reg']
    ).to(CONFIG['device'])
    
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    print(">>> 2. Training...")
    model.train()
    for epoch in range(CONFIG['epochs']):
        total_loss = 0
        for batch in train_loader:
            x = batch[0].to(CONFIG['device'])
            optimizer.zero_grad()
            
            elbo, _, _ = model(x)
            loss = -elbo # Maximize ELBO -> Minimize -ELBO
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{CONFIG['epochs']} | Loss: {total_loss/len(train_loader):.4f}")
            
    return model, dataset

if __name__ == "__main__":
    if os.path.exists(CONFIG['plot_dir']):
        import shutil
        shutil.rmtree(CONFIG['plot_dir'])
    os.makedirs(CONFIG['plot_dir'])
    
    model, dataset = train_model_wrapper()
    _ = analyze_latent_manifold(model, dataset)
    backtest_pipeline(model, dataset)
    print("DONE.")
