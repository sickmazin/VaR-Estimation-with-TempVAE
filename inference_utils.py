import torch

def run_inference(model, x):
    """
    Extracts latent variables (mu) from Vectorized TempVAE.
    Args:
        model: TempVAE instance
        x: Input tensor [Batch, Seq, Dim]
    Returns:
        mu_z: Tensor [Batch, Seq, Latent]
    """
    model.eval()
    with torch.no_grad():
        # 1. Context (Bi-RNN)
        context, _ = model.inf_bi_rnn(x)
        
        # 2. Inference MLP -> q(Z)
        inf_params = model.inf_mlp(context)
        q_mu, _ = torch.chunk(inf_params, 2, dim=2)
        
        return q_mu

def run_generation(model, z_seq):
    """
    Generates reconstruction parameters from Z sequence.
    Args:
        model: TempVAE instance
        z_seq: Tensor [Batch, Seq, Latent] (or [Batch*MC, 1, Latent])
    Returns:
        mu_r: Tensor [Batch, Seq, Dim]
        diag_r: Tensor [Batch, Seq, Dim]
    """
    model.eval()
    with torch.no_grad():
        # Generation RNN
        emission_rnn_out, _ = model.gen_rnn_r(z_seq)
        
        # Emission MLP
        r_params = model.gen_mlp_r(emission_rnn_out)
        
        r_mu = r_params[:, :, :model.input_dim]
        r_diag_log = r_params[:, :, model.input_dim : 2*model.input_dim]
        
        return r_mu, torch.exp(r_diag_log) + 1e-4