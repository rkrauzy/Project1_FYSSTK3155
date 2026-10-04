# LLM-assisted
# Tool: ChatGPT, GPT-5.6 Sol (OpenAI, September 2026)
# Level: 2 - Snippet
# Role: Assisted with parts of the singular-value shrinkage analysis
# used to illustrate how Ridge regularization stabilizes the model.
# Verification: Reviewed and tested by the project authors.

import numpy as np
import matplotlib.pyplot as plt

from src.data import generate_data, design_matrix, scale_matrix, split_data
from src.plotting import save_fig


x, y, y_true = generate_data(n=100, sigma=0.1)
x_train, x_test, y_train, y_test = split_data(x, y)

degree = 15
lambdas = [1e-6, 1e-4, 1e-2, 1, 100]

X_train = design_matrix(x_train, degree)
X_train_s = scale_matrix(X_train)

n_train = len(y_train)
singular_values = np.linalg.svd(X_train_s, compute_uv=False)
mode_numbers = np.arange(1, len(singular_values) + 1)

plt.figure(figsize=(9, 6))

for lmb in lambdas:
    # Convention B: closed form uses (X^TX + n*lambda*I), so shrinkage is s²/(s² + n*lambda)
    shrinkage = singular_values**2 / (singular_values**2 + n_train * lmb)

    plt.plot(mode_numbers, shrinkage, "o-", linewidth=2, label=f"λ={lmb:g}")

plt.xlabel("Singular-value mode")
plt.ylabel("Shrinkage factor")
plt.title("Ridge: Shrinkage of singular-value modes")
plt.xticks(mode_numbers)
plt.ylim(-0.05, 1.05)
plt.legend()
plt.tight_layout()
save_fig("ridge_singular_values")
