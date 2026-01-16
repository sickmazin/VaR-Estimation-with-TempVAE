# TempVAE for Value-at-Risk (VaR) Estimation

## 1. Project Overview & Scientific Context

This repository contains a PyTorch implementation of a **Temporal Variational Autoencoder (TempVAE)** designed for the stochastic modeling of multivariate financial time series. The primary objective is the estimation of **Value-at-Risk (VaR)** through a deep generative approach that captures non-linear dependencies and latent market regimes.

Unlike traditional econometric models (e.g., GARCH, DCC) or deterministic neural networks (e.g., standard LSTMs), this framework treats the latent state of the market as a stochastic variable $Z_t$. The model learns the joint probability distribution of asset returns by maximizing the Evidence Lower Bound (ELBO) on the log-likelihood of the data.

### Key Features
*   **Stochastic Latent Dynamics:** Models market regimes via autoregressive latent variables $Z_t$ conditioned on past states $Z_{<t}$ and past observations $R_{<t}$.
*   **Variational Inference:** Utilizes a Bidirectional GRU encoder to approximate the posterior distribution $q(Z_{1:T} | R_{1:T})$.
*   **Generative Decoder:** Reconstructs the distribution of returns using an MLP parameterization for the conditional means and covariances.
*   **Paper Replication:** Includes specific scripts to replicate the benchmarks and results discussed in the reference literature ("A Statistical Neural Network Approach for Value-at-Risk Analysis").

---

## 2. Repository Structure

The codebase is organized to separate the model architecture, data pipeline, and analysis logic.

### Core Modules
| File | Description |
|------|-------------|
| `TempVae.py` | **Model Architecture**. Contains the `TempVAE` class, defining the inference network (Bi-GRU), the generative network (Autoregressive MLP), and the reparameterization logic. Implements the specific prior and posterior transitions. |
| `vae.py` | **Data & Baselines**. Contains the `FinancialDataset` class for sliding-window sequences processing and legacy training loops. Defines the `MLP` and covariance layers. |
| `analisi.py` | **Main Analysis Pipeline**. The primary driver for training the model, running inference, computing metrics, and generating visualizations. It orchestrates the flow from raw data to latent space inspection. |
| `inference_utils.py` | **Utility Functions**. Helper functions for performing step-by-step inference, handling latent sampling, and transforming outputs. |

### Replication & Experiments
| File | Description |
|------|-------------|
| `paper_exact_replication.py` | Scripts specifically tuned to reproduce the hyperparameters and experimental setup of the reference paper. |
| `paper_exact_replication_updated.py` | An updated version of the replication script with potential optimizations or adjustments for current datasets. |
| `inference_comparison.py` | Tools to compare the TempVAE performance against baselines or different model configurations. |

### Data Directories
*   `dataset/`: Contains raw CSV files of financial log-returns and market data (e.g., `log_returns.csv`, `market_data.csv`).
*   `data_processed/`: serialized PyTorch tensors (`.pt`) for efficient loading of training and testing splits.
*   `comparison_results/` & `paper_plots/`: Output directories for generated plots (latent manifolds, VaR violations, correlation heatmaps).

---

## 3. Mathematical Framework

The model assumes the data generation process follows:

$$ p(R_{1:T}, Z_{1:T}) = p(Z_1) p(R_1|Z_1) \prod_{t=2}^T p(Z_t | Z_{<t}, R_{<t}) p(R_t | Z_t) $$ 

Where:
*   $R_t \in \mathbb{R}^d$: Observed log-returns at time $t$.
*   $Z_t \in \mathbb{R}^k$: Latent stochastic variables (market factors).

The **Inference Model** (Encoder) approximates the posterior:
$$ q(Z_t | Z_{t-1}, R_{1:T}) $$ 

implemented via a Bidirectional GRU to capture global context.

The **Generative Model** (Decoder) parameterizes the conditional distribution of returns, typically as a Multivariate Normal:
$$ p(R_t | Z_t) = \mathcal{N}(\mu_\theta(Z_t), \Sigma_\theta(Z_t)) $$ 

---

## 4. Installation & Setup

### Prerequisites
*   Python 3.8+
*   PyTorch (with CUDA support recommended)

### Setup
1.  **Clone the repository:**
    ```bash
    git clone <repo-url>
    cd VaR_NN
    ```

2.  **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # Unix/MacOS
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 5. Usage

### Data Preparation
Ensure your dataset is placed in `dataset/log_returns.csv`. The `FinancialDataset` class in `vae.py` handles the preprocessing, including standardization and sliding window creation.

### Running the Analysis
To train the model and generate analysis plots:
```bash
python main.py
```
*Check `analisi.py` to modify `CONFIG` parameters (batch size, latent dimension, epochs).*

### Paper Replication
To run the specific replication experiment:
```bash
python paper_exact_replication.py
```

---

## 6. Results & Artifacts
Upon execution, the system generates artifacts in `extracted_outputs/` and `paper_plots/`. Key visualizations include:
*   **Latent Space PCA/t-SNE:** Visualizing how the model clusters different market regimes (e.g., high vs. low volatility).
*   **Reconstruction Analysis:** Comparing original returns vs. model-generated samples.
*   **VaR Violations:** Time-series plots showing the predicted Value-at-Risk against actual losses.

---
*Author: Mattia*
*Project: Advanced Machine Learning for Finance*