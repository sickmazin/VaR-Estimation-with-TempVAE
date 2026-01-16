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

$$ R_t \mid Z_t \sim \mathcal{N}(\mu_\theta(Z_t), \Sigma_\theta(Z_t)) $$

### Inference Model (Encoder)
The true posterior is approximated via a variational distribution $q_\phi$ using a **Bidirectional GRU**:

$$ q_\phi(Z_t \mid Z_{1:t-1}, R_{1:T}) $$

---

## 3. Paper Replication: "A Statistical Neural Network Approach for Value-at-Risk Analysis"

This repository provides an exact replication of the methodology described in:
> *A Statistical Neural Network Approach for Value-at-Risk Analysis* (2022).

The results stored in `final_data/paper_run_result` represent the benchmark performance on the original dataset, achieving stable convergence and capturing significant market volatility regimes.

### Key Replication Artifacts (`final_data/paper_run_result`)
*   **`var_series.png`**: Visualization of the predicted VaR levels against actual log-returns. It demonstrates the model's ability to adjust risk thresholds during periods of high volatility (e.g., market crashes).
*   **`latent_manifold.png`**: A 2D projection of the latent space $Z_t$, showing how the model clusters different market regimes.
*   **`latent_heatmap.png`**: Correlation analysis of the latent dimensions, ensuring that the model learns disentangled or structured representations of risk factors.
*   **`active_units.png`**: Analysis of the KL-divergence per latent unit, used to monitor "posterior collapse" and ensure all dimensions of $Z$ are contributing to the reconstruction.
*   **`output.txt`**: Detailed logs of the replication run, including ELBO decomposition (Reconstruction Loss vs. KL Divergence).

---

## 4. Project Structure & File Roles

### 🧠 Model Architecture
*   **`TempVae.py`**: **THE MODEL**. Defines the `TempVAE` class, the Bi-GRU Encoder, and the Autoregressive Decoder.

### ⚙️ Main Execution & Data
*   **`main.py`**: **ENTRY POINT**. Primary script for training and analysis. Includes the `FinancialDataset` class and `CONFIG` parameters.

### 🔬 Research & Replication
*   **`paper_exact_replication.py`**: **BENCHMARK SCRIPT**. Standalone script optimized to reproduce the 2022 paper's results using the `final_data/ORIGINALE_2` configuration.

### 🛠 Utilities
*   **`inference_utils.py`**: Post-training tools for VaR calculation and sampling.
*   **`visualize_graph.py`**: Utility to visualize the neural network architecture.

---

## 5. Setup & Usage

1.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run Standard Analysis**:
    ```bash
    python main.py
    ```
3.  **Run Exact Paper Replication**:
    ```bash
    python paper_exact_replication.py
    ```

---

*Author: Mattia*