import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from src.data import generate_data, design_matrix


x, y, y_true = generate_data(n=100, sigma=0.1, seed=42)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42
)

degree = 15
lambdas = [1e-6, 1e-4, 1e-2, 1, 100]

X_train = design_matrix(x_train, degree)

mean_X = np.mean(X_train[:, 1:], axis=0)
X_train[:, 1:] = X_train[:, 1:] - mean_X

X_features = X_train[:, 1:]

singular_values = np.linalg.svd(
    X_features,
    compute_uv=False
)

mode_numbers = np.arange(1, len(singular_values) + 1)

plt.figure(figsize=(9, 6))

for lmbda in lambdas:
    shrinkage = singular_values**2 / (
        singular_values**2 + lmbda
    )

    plt.plot(
        mode_numbers,
        shrinkage,
        marker="o",
        linewidth=2,
        label=f"lambda = {lmbda:g}"
    )

plt.xlabel("Singular-value mode")
plt.ylabel("Shrinkage factor")
plt.title("Ridge: Shrinkage of singular-value modes")
plt.xticks(mode_numbers)
plt.ylim(-0.05, 1.05)
plt.legend()
plt.tight_layout()
plt.show()
