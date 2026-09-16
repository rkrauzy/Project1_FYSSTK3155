import numpy as np
import matplotlib.pyplot as plt

data = np.load("results_part_a.npz")

degrees = data["degrees"]
mse_train = data["mse_train"]
mse_test = data["mse_test"]
sigma = data["sigma"]

plt.figure(figsize=(8, 5))

plt.plot(degrees, mse_train, marker="o", label="Training MSE")
plt.plot(degrees, mse_test, marker="o", label="Test MSE")
plt.axhline(sigma**2, linestyle="--", label=r"Noise variance $\sigma^2$")

plt.xlabel("Polynomial degree")
plt.ylabel("MSE")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
