import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np

train = pd.read_csv("BT2024076_train_var1.csv")
test = pd.read_csv("BT2024076_test_var1.csv")

X = train[["x1", "x2", "x3", "x4", "x5", "x6"]]
y = train["y"]
X_test = test[["x1", "x2", "x3", "x4", "x5", "x6"]]

degree = 4
poly = PolynomialFeatures(degree=degree)
X_train_poly = poly.fit_transform(X)
X_test_poly = poly.transform(X_test)

model = LinearRegression()

model.fit(X_train_poly, y)
y_pred = model.predict(X_test_poly) 

result = pd.DataFrame({"y": y_pred})
result.to_csv("BT2024076_test_var1_predictions.csv", index=False)
