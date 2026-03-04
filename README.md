# Unsupervised Learning: Recommender Systems and Anomaly Detection

This repository contains two core implementations of Unsupervised Learning algorithms from scratch using NumPy and SciPy. These projects demonstrate the ability to handle statistical inference and matrix factorization without high-level ML frameworks.

## 1. Anomaly Detection (Server Monitoring)
Implemented a Gaussian distribution model to identify failing servers on a network by analyzing throughput (mb/s) and latency (ms).
- **Core Algorithm:** Multivariate Gaussian probability density function to compute $p(x)$.
- **Optimization:** Automated threshold ($\varepsilon$) selection using the $F_1$ score to balance Precision and Recall.
- **Goal:** Identifying anomalies that fall outside the learned high-probability density regions.

## 2. Movie Recommender System (Collaborative Filtering)
Built a personalized recommendation engine using the MovieLens dataset (1,682 movies, 943 users).
- **Model:** Matrix Factorization where user preferences ($\Theta$) and movie features ($X$) are learned simultaneously.
- **Regularization:** L2 regularization (Weight Decay) applied to prevent overfitting in sparse rating matrices.
- **Technique:** Vectorized implementation of the cost function and gradients for efficient computation.

## Technical Stack
- **Languages:** Python (Jupyter Notebook)
- **Scientific Computing:** NumPy, SciPy (for `.mat` file loading)
- **Data Visualization:** Matplotlib
