import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

train = pd.read_csv("BT2024076_train_var1.csv")
test  = pd.read_csv("BT2024076_test_var1.csv")

X = train[["x1", "x2", "x3", "x4", "x5", "x6"]]
y = train["y"]

kf = KFold(n_splits=5, shuffle=True, random_state=42)

alphas = [2.4,2.6,2.8,3.3,4]

for alpha in alphas:
    fold_rmse = []
    fold_r2 = []

    for (fold,(train_idx,val_idx)) in enumerate(kf.split(X)):
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]

        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]

        poly = PolynomialFeatures(degree=4)

        X_train_poly = poly.fit_transform(X_train_fold)
        X_val_poly = poly.transform(X_val_fold)

        model = Ridge(alpha=alpha)

        model.fit(X_train_poly, y_train_fold)
        y_pred = model.predict(X_val_poly)

        rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred))
        r2 = r2_score(y_val_fold, y_pred)

        fold_rmse.append(rmse)
        fold_r2.append(r2)

    average_rmse = np.mean(fold_rmse)
    average_r2 = np.mean(fold_r2)

    print("Alpha:", alpha,
        "Avg RMSE:", average_rmse,
        "Avg R2:", average_r2)

