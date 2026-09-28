# LLM-assisted
# Tool: ChatGPT, GPT-5.6 Sol (OpenAI, September 2026)
# Level: 4 - Substantial
# Role: Suggested and substantially assisted with an additional condition-number
# analysis comparing unscaled and standardized polynomial design matrices.
# The suggestion was reviewed and retained by the project authors to provide
# quantitative support for the discussion of standardization and numerical stability.
# Verification: Reviewed, executed and interpreted by the project authors.

import numpy as np
import matplotlib.pyplot as plt

from src.data import generate_data, design_matrix, scale_matrix, split_data
from src.plotting import save_fig


x, y, y_true = generate_data(n=100, sigma=0.1)
x_train, x_test, y_train, y_test = split_data(x, y)

degrees = range(1, 16)
cond_raw, cond_scaled = [], []

for d in degrees:
    X = design_matrix(x_train, d)
    X_s = scale_matrix(X)
    cond_raw.append(np.linalg.cond(X))
    cond_scaled.append(np.linalg.cond(X_s))

plt.figure(figsize=(9, 6))
plt.semilogy(degrees, cond_raw, "o-", linewidth=2, label="Unscaled")
plt.semilogy(degrees, cond_scaled, "o-", linewidth=2, label="Standardized")
plt.xlabel("Polynomial degree")
plt.ylabel("Condition number")
plt.title("OLS: Effect of standardization on numerical conditioning")
plt.xticks(degrees)
plt.legend()
plt.tight_layout()
save_fig("ols_conditioning")
