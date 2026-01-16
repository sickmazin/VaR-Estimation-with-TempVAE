# TempVAE: Deep Stochastic Volatility for Value-at-Risk

## 1. Scientific Abstract & Objective

This project implements a **Temporal Variational Autoencoder (TempVAE)** for multivariate financial time series analysis. The core objective is to model the joint distribution of asset returns $R_t$ by inferring a latent stochastic process $Z_t$ that represents unobserved market factors (regimes).

Unlike deterministic models, this architecture treats the latent state as a random variable, allowing for a probabilistic estimation of risk measures like **Value-at-Risk (VaR)**. The model optimizes the Evidence Lower Bound (ELBO) of the log-likelihood, effectively balancing the reconstruction accuracy of returns against the complexity of the latent dynamics.

---

## 2. Mathematical Framework

The model is defined by a state-space formulation where the generative process factorizes as follows:

$$
p_\theta(R_{1:T}, Z_{1:T}) = \prod_{t=1}^{T} p_\theta(R_t \mid Z_t) p_\theta(Z_t \mid Z_{1:t-1})
$$

### Generative Model (Decoder)
The conditional distribution of returns is parameterized (typically as a Gaussian) by a neural network:

$$R_t \mid Z_t \sim \mathcal{N}(\mu_\theta(Z_t), \Sigma_\theta(Z_t))
$$

### Inference Model (Encoder)
We approximate the posterior using a variational distribution $q_\phi$ parameterized by a **Bidirectional Recurrent Neural Network (Bi-GRU)** that captures future and past context:

$$q_\phi(Z_t \mid Z_{1:t-1}, R_{1:T})
$$

---

## 3. Paper Replication: "Estimating the Value-at-Risk by Temporal VAE"

This repository provides an exact replication of the methodology described in the reference paper:
> **"Estimating the Value-at-Risk by Temporal VAE"** (Available in `paper/EstimatingVaR_TempVAE.pdf`).

The results stored in **`final_data/paper_run_result`** represent the benchmark performance achieved by following the hyperparameters and architectural constraints defined in this research.

### Key Replication Artifacts (`final_data/paper_run_result`)
*   **`var_series.png`**: Predicted VaR levels vs actual log-returns, showing the model's response to volatility shocks.
*   **`latent_manifold.png`**: 2D projection of the latent space $Z_t$, identifying market regime clusters.
*   **`latent_heatmap.png`**: Correlation analysis of latent dimensions to ensure factor disentanglement.
*   **`active_units.png`**: KL-divergence per latent unit, monitoring "posterior collapse" to ensure efficient latent space utilization.

---

## 4. Project Structure & File Roles

### 🧠 Model Architecture
*   **`TempVae.py`**: **THE MODEL**. Defines the `TempVAE` class architecture, including the Bi-GRU Encoder and the Autoregressive Decoder.

### ⚙️ Execution & Data
*   **`main.py`**: **ENTRY POINT (Training)**. Primary script for training and full analysis. Includes the `FinancialDataset` class and the `CONFIG` dictionary.
*   **`inference_comparison.py`**: **INFERENCE ENGINE**. Use this script to load a pre-trained `best_model.pth` and perform backtesting/comparisons without training the network again.

### 🔬 Research & Replication
*   **`paper_exact_replication.py`**: **BENCHMARK SCRIPT**. Specialized for reproducing the results of the paper using fixed seeds and specific initialization schemes.

### 🛠 Utilities
*   **`inference_utils.py`**: Post-training tools for VaR calculation and distribution sampling.
*   **`visualize_graph.py`**: Utility to visualize the neural network architecture graph.

---

## 5. Setup & Usage

### Installation
```bash
pip install -r requirements.txt
```

### Option A: Train from Scratch
To train the model and generate a new `best_model.pth`:
```bash
python main.py
```

### Option B: Run Inference (Using Saved Weights)
To perform analysis using the already trained model:
```bash
python inference_comparison.py
```
*This script loads the weights (e.g., from `final_data/paper_run_result/best_model.pth`), sets the model to `eval()` mode, and executes the backtesting pipeline.*

---

*Author: Mattia