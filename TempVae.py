import torch
import torch.nn as nn
import torch.distributions as dist

class MLP(nn.Module):
    """
    Modulo MLP generico come descritto:
    - 2 hidden layers con dimensione 16
    - Attivazione ReLU
    - Inizializzazione 'Variance Scaling' (He Initialization)
    """
    def __init__(self, input_dim, latent_dim, hidden_dim=16):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # 2. Media (nessuna attivazione, può essere negativa)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)

        # 3. Diagonale della Covarianza
        # output ancora "grezzo" (pre-attivazione)
        self.fc_cov_diag = nn.Linear(hidden_dim, latent_dim)

        # He Initialization (Variance Scaling)
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # He initialization (Variance Scaling)
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        h = self.net(x)

        # Calcolo della media
        mu = self.fc_mu(h)

        # Calcolo della diagonale grezza
        raw_diag = self.fc_cov_diag(h)

        # --- APPLICAZIONE DELLA RICHIESTA DEL PAPER ---
        # Applichiamo l'esponenziale per rendere i valori strettamente positivi, "exponential activation".
        diag_entries = torch.exp(raw_diag)

        return mu, diag_entries

class TempVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=10, hidden_dim=16, dropout=0.1):
        """
        Args:
            input_dim (int): Dimensione dei rendimenti degli asset (d)
            latent_dim (int): Dimensione dello spazio latente (kappa)
            hidden_dim (int): Dimensione hidden per RNN e MLP (default 16)
            dropout (float): Rate di dropout per le RNN (default 10%)
        """
        super(TempVAE, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # --- INFERENCE MODEL (Encoder) ---
        # Eq (15) & (16): Bidirectional RNN per estrarre il contesto da R
        # Input: R_t, Output: [h_forward, h_backward] (dimensione 2 * hidden_dim)
        self.inf_bi_rnn = nn.GRU(input_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.inf_dropout = nn.Dropout(dropout)

        # Eq (14): RNN inferenza autoregressiva
        # Input: concat(h^z_{t-1}, Z_{t-1}, Context_t)
        # Context_t è l'output della BiRNN (2*hidden)
        self.inf_rnn_z = nn.GRUCell(latent_dim + 2 * hidden_dim, hidden_dim)

        # Eq (13): MLP per parametri q(Z)
        # Output: mu (kappa) + sigma_diag (kappa)
        self.inf_mlp = MLP(hidden_dim, latent_dim, hidden_dim)

        # --- GENERATIVE MODEL (Decoder) ---
        # Eq (6): RNN Prior
        # Input: Z_{t-1}, Hidden: h^z_{t-1}
        self.gen_rnn_z = nn.GRUCell(latent_dim, hidden_dim)

        # Eq (5): MLP Prior
        # Output: mu (kappa) + sigma_diag (kappa)
        self.gen_mlp_z = MLP(hidden_dim, latent_dim, hidden_dim)

        # --- MODIFICA PRIOR (Paper Replication) ---
        # Inizializzazione e Congelamento (Freeze) per Prior Fissa
        for m in [self.gen_rnn_z, self.gen_mlp_z]:
            # Inizializzazione custom
            for name, param in m.named_parameters():
                if 'weight' in name:
                    nn.init.kaiming_normal_(param.data)
                elif 'bias' in name:
                    nn.init.zeros_(param.data)
            
            # Congeliamo i pesi
            for param in m.parameters():
                param.requires_grad = False

        # Eq (9): RNN Emissione
        # Input: Z_t, Hidden: h^r_{t-1}
        self.gen_rnn_r = nn.GRUCell(latent_dim, hidden_dim)

        # Eq (8): MLP Emissione
        # Output per Rank-1 Perturbation:
        #  Diagonale D (input_dim)
        #  Vettore fattore u (input_dim) -> per uu^Trasposta
        #  Media mu (input_dim)
        # Totale: input_dim * 3
        self.gen_mlp_r = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim * 3) # Un unico vettore lungo che poi tagliamo
        )
        self.gen_dropout = nn.Dropout(dropout)


    def forward(self, x):
        """
        Args:
            x: Tensore [Batch, Seq_Len, Input_Dim] (R_{1:T})
        Returns:
            recon_loss: Negativa Log-Likelihood
            kl_loss: KL Divergence totale
        """
        batch_size, seq_len, _ = x.size()
        device = x.device

        # --- 1. Bi-RNN Standard su R ---
        # Output shape: [Batch, Seq, 2*Hidden]
        # Contiene [h_fwd_t, h_bwd_t] allineati con R_t
        bi_rnn_out_raw, _ = self.inf_bi_rnn(x)
        bi_rnn_out_raw = self.inf_dropout(bi_rnn_out_raw)

        # Separazione h forward e backward
        # fwd_out[t] contiene info su R_0...R_t
        # bwd_out[t] contiene info su R_t...R_T
        fwd_out = bi_rnn_out_raw[:, :, :self.hidden_dim]
        bwd_out = bi_rnn_out_raw[:, :, self.hidden_dim:]

        # --- 2. Implementazione Shifting (Eq 15 e 16) ---

        # Eq (15): h_forward_t deve dipendere da R_{t-1}.
        # Al tempo t=0, non c'è R_{-1}, quindi usiamo uno stato zero (padding).
        # Al tempo t=k, usiamo fwd_out[k-1].
        zeros_fwd = torch.zeros(batch_size, 1, self.hidden_dim).to(device)
        # Shift a destra: [0, h_0, h_1, ..., h_{T-2}]
        h_arrow_right = torch.cat([zeros_fwd, fwd_out[:, :-1, :]], dim=1)

        # Eq (16): h_backward_t deve dipendere da R_{t+1}.
        # Al tempo t=T-1 (ultimo), non c'è R_{T}, usiamo zero.
        # Al tempo t=k, usiamo bwd_out[k+1].
        zeros_bwd = torch.zeros(batch_size, 1, self.hidden_dim).to(device)
        # Shift a sinistra: [h_1, h_2, ..., h_{T-1}, 0]
        h_arrow_left = torch.cat([bwd_out[:, 1:, :], zeros_bwd], dim=1)

        # Concateniamo i contesti corretti per l'equazione (14)
        # Ora context_shifted[:, t] contiene esattamente [h->_{t}, h<-_{t}] come da Eq 15-16
        context_shifted = torch.cat([h_arrow_right, h_arrow_left], dim=2)

        # ... inizializzazione stati ...
        h_z = torch.zeros(batch_size, self.hidden_dim).to(device)
        # h_gen_z per la Prior (inizializzato a zero)
        h_gen_z = torch.zeros(batch_size, self.hidden_dim).to(device)
        h_gen_r = torch.zeros(batch_size, self.hidden_dim).to(device)
        z_prev = torch.zeros(batch_size, self.latent_dim).to(device)

        kl_loss = 0
        nll_loss = 0

        for t in range(seq_len):
            # --- INFERENCE (q) ---

            # Eq (14): RNN_I^z prende (h_{t-1}, Z_{t-1}, [h->_t, h<-_t])
            # Usiamo il contesto shiftato calcolato sopra
            current_context = context_shifted[:, t, :] # [Batch, 2*Hidden]

            rnn_input_inf = torch.cat([z_prev, current_context], dim=1)
            h_z = self.inf_rnn_z(rnn_input_inf, h_z)
            h_z = self.inf_dropout(h_z)

            # Eq (13): Ottieni parametri q(Z_t | ...)
            q_mu, q_sigma = self.inf_mlp(h_z)

            # --- GENERATIVE PRIOR (p(Z)) ---
            # Prior Dinamica Fissa: p(Z_t | Z_{1:t-1})
            # Aggiornamento hidden state usando Z_{t-1} (z_prev)
            h_gen_z = self.gen_rnn_z(z_prev, h_gen_z)
            
            # Calcolo parametri prior per lo step t
            p_mu, p_sigma = self.gen_mlp_z(h_gen_z)

            # Calcolo KL Divergence analitica per step t (Eq 20 interna)
            # D_KL(N(q_mu, q_var) || N(p_mu, p_var))
            q_dist = dist.Normal(q_mu, q_sigma)
            p_dist = dist.Normal(p_mu, p_sigma)
            kl_loss += dist.kl_divergence(q_dist, p_dist).sum(dim=1).mean()

            # Sample Z_t (Eq 17)
            z_t = q_dist.rsample()

            # --- GENERATIVE EMISSION (p(R|Z)) ---
            # Eq (9): Aggiorno stato nascosto generazione usando Z_t ATTUALE
            h_gen_r = self.gen_rnn_r(z_t, h_gen_r)

            h_gen_r= self.gen_dropout(h_gen_r)

            # Eq (8): Parametri generazione
            r_params = self.gen_mlp_r(h_gen_r)

            # Decomposizione parametri per Rank-1 Covariance (LowRankMultivariateNormal)
            # mu: [Batch, d]
            # diag_log: [Batch, d] (log della diagonale)
            # factor: [Batch, d] (vettore u per uu^T)
            r_mu = r_params[:, :self.input_dim]
            r_diag_log = r_params[:, self.input_dim : 2*self.input_dim]
            r_factor = r_params[:, 2*self.input_dim :].unsqueeze(-1) # [Batch, d, 1]

            # Attivazione esponenziale per la diagonale
            r_diag = torch.exp(r_diag_log) + 1e-4

            # Distribuzione Multivariate Normal con struttura Low Rank
            # Covariance = diag(r_diag) + r_factor @ r_factor.T
            # Questa classe gestisce efficientemente il calcolo della log_prob
            r_dist = dist.LowRankMultivariateNormal(
                loc=r_mu,
                cov_factor=r_factor,
                cov_diag=r_diag
            )

            # Calcolo Log Likelihood (Eq 10 -> Eq 19)
            # MASSIMIZZARE la log prob, quindi per la loss prendiamo il negativo
            nll_loss -= r_dist.log_prob(x[:, t, :]).mean()

            # Preparazione per il prossimo step
            z_prev = z_t

        return nll_loss, kl_loss
