import numpy as np
import matplotlib.pyplot as plt

data = np.load("results_part_a.npz")

degrees = data["degrees"]
condition_numbers = data["condition_numbers"]
condition_numbers_uncentered = data["condition_numbers_uncentered"]

plt.figure(figsize=(8, 5))

plt.semilogy(
    degrees, 
    condition_numbers_uncentered, 
    marker="o", 
    label="Uncentered"
    )
plt.semilogy(
    degrees, 
    condition_numbers, 
    marker="o", 
    label="Centered"
    )

plt.xlabel("Polynomial degree")
plt.ylabel("Condition number")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
