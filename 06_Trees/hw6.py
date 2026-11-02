"""Homework 6: decision trees, random forests and XGBoost on car fuel efficiency."""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

TARGET = "fuel_efficiency_mpg"

# Data preparation
df = pd.read_csv("car_fuel_efficiency_2026.csv").fillna(0)

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

y_train = df_train[TARGET].values
y_val = df_val[TARGET].values

dv = DictVectorizer(sparse=True)
X_train = dv.fit_transform(df_train.drop(columns=TARGET).to_dict(orient="records"))
X_val = dv.transform(df_val.drop(columns=TARGET).to_dict(orient="records"))
features = list(dv.get_feature_names_out())


def forest_rmse(**params):
    model = RandomForestRegressor(random_state=1, n_jobs=-1, **params)
    model.fit(X_train, y_train)
    return model, root_mean_squared_error(y_val, model.predict(X_val))


# Q1: feature used by a depth-1 tree
tree = DecisionTreeRegressor(max_depth=1).fit(X_train, y_train)
print("Q1 split feature:", features[tree.tree_.feature[0]])

# Q2: random forest baseline
print("Q2 RMSE:", round(forest_rmse(n_estimators=10)[1], 4))

# Q3: number of trees
for n in [10, 50, 100, 150]:
    print(f"Q3 n_estimators={n}: {forest_rmse(n_estimators=n)[1]:.3f}")

# Q4: max_depth, averaged over n_estimators
for depth in [10, 15, 20, 25]:
    scores = [
        forest_rmse(n_estimators=n, max_depth=depth)[1] for n in [10, 50, 100, 150]
    ]
    print(f"Q4 max_depth={depth}: mean RMSE {np.mean(scores):.4f}")

# Q5: feature importance
model, _ = forest_rmse(n_estimators=10, max_depth=20)
ranked = sorted(zip(features, model.feature_importances_), key=lambda t: -t[1])
for name, score in ranked[:5]:
    print(f"Q5 {name}: {score:.4f}")

# Q6: XGBoost, eta 0.3 vs 0.1
d_train = xgb.DMatrix(X_train, label=y_train, feature_names=features)
d_val = xgb.DMatrix(X_val, label=y_val, feature_names=features)

for eta in [0.3, 0.1]:
    params = {
        "eta": eta,
        "max_depth": 6,
        "min_child_weight": 1,
        "objective": "reg:squarederror",
        "nthread": 8,
        "seed": 1,
        "verbosity": 1,
    }
    history = {}
    xgb.train(
        params,
        d_train,
        num_boost_round=100,
        evals=[(d_train, "train"), (d_val, "val")],
        evals_result=history,
        verbose_eval=False,
    )
    print(f"Q6 eta={eta}: final val RMSE {history['val']['rmse'][-1]:.4f}")
