import optuna
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import os
import sys
import copy

# Importiamo il modello originale
from TempVae import TempVAE

# ==========================================
# CONFIGURAZIONE AVANZATA
# ==========================================
BASE_CONFIG = {
    'data_path': 'dataset/log_returns_paper.csv',
    'train_split': 0.66,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # PARAMETRI PER LA ROBUSTEZZA
    'epochs_per_fold': 100,      # Numero corretto per una convergenza vera
    'patience': 15,             # Early Stopping: ferma se non migliora per 15 epoche
    'n_splits': 3,              # Manteniamo 3 fold temporali
    
    # Parametri fissi del modello base
    'latent_dim': 10,       
    'dropout_rate': 0.1,    
    'free_bits': 2.0,
    'annealing_steps': 20   # Annealing come paper lento per stabilizzare la KL
}

# ==========================================
# GESTIONE DATI DINAMICA
# ==========================================
class DynamicFinancialDataset:
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        self.df = self.df.dropna()
        self.raw_data = self.df.values.astype(np.float32)
        
        # Standardizzazione basata SOLO sulla prima porzione di dati (Train Split teorico)
        # per evitare data leakage globale.
        split_idx = int(len(self.raw_data) * BASE_CONFIG['train_split'])
        train_raw = self.raw_data[:split_idx]
        
        self.mean = np.mean(train_raw, axis=0)
        self.std = np.std(train_raw, axis=0)
        self.std[self.std == 0] = 1.0
        
        self.data_scaled = (self.raw_data - self.mean) / self.std

    def get_windows(self, seq_len):
        windows = []
        # Usiamo i dati fino al train split per l'ottimizzazione
        # (Simuliamo che il test set finale non esista ancora)
        limit = int(len(self.data_scaled) * BASE_CONFIG['train_split'])
        data_subset = self.data_scaled[:limit]
        
        for i in range(len(data_subset) - seq_len + 1):
            windows.append(data_subset[i:i+seq_len])
        
        return torch.tensor(np.array(windows), dtype=torch.float32)

print(">>> Caricamento Dataset in memoria...")
GLOBAL_DATASET = DynamicFinancialDataset(BASE_CONFIG['data_path'])
print(">>> Dataset caricato.")

# ==========================================
# UTILS: EARLY STOPPING
# ==========================================
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# ==========================================
# TRAINING LOOP DEL FOLD
# ==========================================
def train_and_evaluate_fold(model, train_loader, val_loader, optimizer, gradient_clip, device, trial, fold_idx):
    
    early_stopper = EarlyStopping(patience=BASE_CONFIG['patience'])
    best_fold_loss = float('inf')
    
    model.train() # Set model to training mode
    
    for epoch in range(BASE_CONFIG['epochs_per_fold']):
        # Annealing
        beta = min(1.0, epoch / BASE_CONFIG['annealing_steps'])
        
        # --- TRAIN ---
        model.train() # Ensure model is in training mode for this epoch
        train_loss_accum = 0.0
        for batch in train_loader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            
            nll_loss, kl_loss = model(x)
            
            # Free bits hinge loss
            kl_effective = torch.max(kl_loss, torch.tensor(BASE_CONFIG['free_bits']).to(device))
            loss = nll_loss + beta * kl_effective
            
            if torch.isnan(loss):
                return float('inf') 
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            train_loss_accum += loss.item()

        # --- VALIDATION ---
        model.eval() # Set model to evaluation mode
        val_loss_accum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                nll, kl = model(x)
                # Validation Loss "Pura" (senza beta annealing dinamico, beta=1 full objective)
                val_loss = nll + kl 
                val_loss_accum += val_loss.item()
        
        avg_val_loss = val_loss_accum / len(val_loader)
        
        # Aggiorniamo la miglior loss vista in questo fold
        if avg_val_loss < best_fold_loss:
            best_fold_loss = avg_val_loss
            
        # Check Early Stopping
        early_stopper(avg_val_loss)
        if early_stopper.early_stop:
            # print(f"    Fold {fold_idx}: Early stopping at epoch {epoch}")
            break
            
        # Reporting intermedio a Optuna per Pruning globale (opzionale, ma utile)
        # Riportiamo solo se siamo in un epoch significativa per non rallentare troppo
        if fold_idx == 0: # Pruning basato principalmente sul primo fold per efficienza
            trial.report(avg_val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    return best_fold_loss

# ==========================================
# OBIETTIVO OPTUNA
# ==========================================
def objective(trial):
    # --- 1. SPAZIO DI RICERCA (Più ampio e granulare) ---
    
    seq_length = trial.suggest_int('seq_length', 15, 60, step=1)
    hidden_dim = trial.suggest_int('hidden_dim', 16, 64, step=1)
    batch_size = trial.suggest_int('batch_size', 32, 256, step=16)
    gradient_clip = trial.suggest_float('gradient_clip', 2.5, 5.0)
    
    # Learning Rate (CRUCIALE: ottimizzarlo insieme al batch size)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 5e-3, log=True)

    use_l2 = trial.suggest_categorical('use_l2', [True, False])
    if use_l2:
        l2_reg = trial.suggest_float('l2_reg', 1e-6, 1e-3, log=True)
    else:
        l2_reg = 0.0

    # --- 2. PREPARAZIONE DATI ---
    # Rigenera finestre con la seq_len corrente
    X_data = GLOBAL_DATASET.get_windows(seq_length)
    
    # --- 3. CROSS VALIDATION ---
    tscv = TimeSeriesSplit(n_splits=BASE_CONFIG['n_splits'])
    fold_losses = []
    
    try:
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_data)):
            # Dataset
            X_train_fold = X_data[train_idx]
            X_val_fold = X_data[val_idx]
            
            # DataLoader
            train_loader = DataLoader(TensorDataset(X_train_fold), batch_size=batch_size, shuffle=True, drop_last=True)
            val_loader = DataLoader(TensorDataset(X_val_fold), batch_size=batch_size, shuffle=False, drop_last=False)
            
            # Model Init
            model = TempVAE(
                input_dim=X_data.shape[-1],
                latent_dim=BASE_CONFIG['latent_dim'],
                hidden_dim=hidden_dim,
                dropout=BASE_CONFIG['dropout_rate']
            ).to(BASE_CONFIG['device'])
            
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_reg)
            
            # Training Robusto
            val_loss = train_and_evaluate_fold(
                model, train_loader, val_loader, optimizer, gradient_clip, BASE_CONFIG['device'], trial, fold_idx
            )
            
            if val_loss == float('inf'): # Se esplode
                return float('inf')
                
            fold_losses.append(val_loss)

    except RuntimeError as e:
        print(f"CUDA/Runtime Error: {e}")
        return float('inf')

    # Ritorniamo la media delle migliori loss dei fold
    return np.mean(fold_losses)

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    print(f"Device: {BASE_CONFIG['device']}")
    print("AVVIO OTTIMIZZAZIONE ROBUSTA (Long Run)...")
    
    # Pruner: Hyperband è spesso migliore del Median per training lunghi, 
    # ma Median è più conservativo. Teniamo MedianPruner ma con warmup.
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))
    
    # Aumentiamo i trial per esplorare meglio lo spazio combinatorio
    N_TRIALS = 100 
    print(f"Target: {N_TRIALS} Trials con Early Stopping.")
    
    study.optimize(objective, n_trials=N_TRIALS)

    print("\n" + "="*50)
    print("RISULTATI FINALI")
    print("="*50)
    
    print("Migliori Parametri:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
        
    print(f"\nBest Validation Loss: {study.best_value:.4f}")
    
    # Salvataggio
    with open("best_hyperparameters_robust.txt", "w") as f:
        f.write("MIGLIORI PARAMETRI (ROBUST SCAN):\n")
        f.write(str(study.best_params))
        f.write(f"\n\nBest Loss: {study.best_value}")
    
    print("\nSalvato in 'best_hyperparameters_robust.txt'")