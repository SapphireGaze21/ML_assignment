import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

train = pd.read_csv("BT2024076_train_var2.csv")
test = pd.read_csv("BT2024076_test_var2.csv")

X = train[["x1", "x2", "x3"]]
y = train["y"]
X_test = test[["x1", "x2", "x3"]]

# print(X.shape)
# print(y.shape)
# print(X_test.shape)

#create KFold splitter
kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = []

for degree in range(1,21):
    fold_rmse = []
    fold_r2 = []
    for fold, (train_index, val_index) in enumerate(kf.split(X)):
        X_train_fold = X.iloc[train_index]
        y_train_fold = y.iloc[train_index]

        X_val_fold = X.iloc[val_index]
        y_val_fold = y.iloc[val_index]

        poly = PolynomialFeatures(degree=degree)

        X_train_poly = poly.fit_transform(X_train_fold)
        X_val_poly = poly.transform(X_val_fold)

        model = LinearRegression()

        model.fit(X_train_poly,y_train_fold)
        y_pred = model.predict(X_val_poly)

        rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred))
        r2 = r2_score(y_val_fold, y_pred)

        fold_rmse.append(rmse)
        fold_r2.append(r2)


    average_rmse = np.mean(fold_rmse)
    average_r2 = np.mean(fold_r2)
    results.append((degree, average_rmse, average_r2))
    print("Degree", degree, "Avg RMSE:", average_rmse, "Avg R2:", average_r2)

        # print(f"Fold: {fold + 1}")
        # print("X_train:", X_train_fold.shape)
        # print("X_val:", X_val_fold.shape)


