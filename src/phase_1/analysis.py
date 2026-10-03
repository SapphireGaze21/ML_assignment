import matplotlib.pyplot as plt

degrees = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

rmse = [
    3.0226931782456314,
    1.7319804618671903,
    0.9676771993564193,
    0.8633039327752228,
    1.1328880229046452,
    10.508236351440685,
    2.886927246924341,
    2.1913402511146876,
    2.131886989505784,
    1.971326217153996
]

plt.plot(degrees, rmse, marker="o")

plt.xlabel("Polynomial Degree")
plt.ylabel("Average CV RMSE")
plt.title("Polynomial Degree vs Cross-Validation RMSE")

plt.xticks(degrees)
plt.grid(True)

plt.show()

r2 = [
    0.11836760994329595,
    0.7100202897931368,
    0.9093837556618982,
    0.9278737004877794,
    0.875431169137354,
    -9.701464469156658,
    0.17396625658097792,
    0.5356743245208152,
    0.5593712738997785,
    0.624382502021564
]

plt.plot(degrees, r2, marker="o")

plt.xlabel("Polynomial Degree")
plt.ylabel("Average CV R²")
plt.title("Polynomial Degree vs Cross-Validation R²")

plt.xticks(degrees)
plt.grid(True)

plt.show()