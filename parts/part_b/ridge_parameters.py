# LLM-assisted
# Tool: ChatGPT, GPT-5.6 Sol (OpenAI, September 2026)
# Level: 2 - Snippet
# Role: Assisted with parts of the Ridge coefficient analysis,
# including evaluation across lambda values and plotting.
# Verification: Reviewed and tested by the project authors.

import numpy as np
import matplotlib.pyplot as plt

from src.data import generate_data, design_matrix, scale_matrix, center_y, split_data
from src.models import ridge
from src.plotting import save_fig


x, y, y_true = generate_data(n=100, sigma=0.1)
x_train, x_test, y_train, y_test = split_data(x, y)
y_train_c, y_mean = center_y(y_train)

degree = 15
lambdas = np.logspace(-8, 2, 11)

X_train = design_matrix(x_train, degree)
X_train_s = scale_matrix(X_train)

theta_values = np.zeros((len(lambdas), X_train_s.shape[1]))

for i, lmb in enumerate(lambdas):
    theta_values[i] = ridge(X_train_s, y_train_c, lmb)

plt.figure(figsize=(9, 7))
colors = plt.cm.viridis(np.linspace(0, 1, theta_values.shape[1]))   # one colour per power, light to dark

for k in range(theta_values.shape[1]):
    plt.plot(lambdas, theta_values[:, k], "o-", linewidth=2, color=colors[k], label=rf"$\theta_{{{k + 1}}}$")

plt.xscale("log")
plt.yscale("symlog", linthresh=1)
plt.xlabel("λ")
plt.ylabel("Coefficient value")
plt.title("Ridge: Coefficients as a function of λ (degree 15)")
sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(1, theta_values.shape[1]))
plt.colorbar(sm, ax=plt.gca(), label=r"Index $j$ of $\theta_j$")   # replaces a legend with one entry per coefficient
plt.tight_layout()
save_fig("ridge_parameters")
