import numpy as np, pandas as pd, xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error as rmse

df = pd.read_csv('car_fuel_efficiency_2026.csv').fillna(0)
dfft, dft = train_test_split(df, test_size=0.2, random_state=1)
dftr, dfv = train_test_split(dfft, test_size=0.25, random_state=1)
y = 'fuel_efficiency_mpg'
ytr, yv = dftr[y].values, dfv[y].values
dv = DictVectorizer(sparse=True)
Xtr = dv.fit_transform(dftr.drop(columns=y).to_dict(orient='records'))
Xv = dv.transform(dfv.drop(columns=y).to_dict(orient='records'))
fn = list(dv.get_feature_names_out())

dt = DecisionTreeRegressor(max_depth=1).fit(Xtr, ytr)
print('Q1', fn[dt.tree_.feature[0]])
rf = RandomForestRegressor(n_estimators=10, random_state=1, n_jobs=-1).fit(Xtr, ytr)
print('Q2', rmse(yv, rf.predict(Xv)))
for n in [10,50,100,150]:
    m = RandomForestRegressor(n_estimators=n, random_state=1, n_jobs=-1).fit(Xtr, ytr)
    print('Q3', n, round(rmse(yv, m.predict(Xv)),3))
for d in [10,15,20,25]:
    s=[]
    for n in [10,50,100,150]:
        m = RandomForestRegressor(n_estimators=n, max_depth=d, random_state=1, n_jobs=-1).fit(Xtr, ytr)
        s.append(rmse(yv, m.predict(Xv)))
    print('Q4', d, np.mean(s))
m = RandomForestRegressor(n_estimators=10, max_depth=20, random_state=1, n_jobs=-1).fit(Xtr, ytr)
imp = sorted(zip(fn, m.feature_importances_), key=lambda t:-t[1])
print('Q5', imp[:5])
dtr, dva = xgb.DMatrix(Xtr, label=ytr, feature_names=fn), xgb.DMatrix(Xv, label=yv, feature_names=fn)
for eta in [0.3, 0.1]:
    p = {'eta':eta,'max_depth':6,'min_child_weight':1,'objective':'reg:squarederror','nthread':8,'seed':1,'verbosity':1}
    ev={}
    xgb.train(p, dtr, 100, evals=[(dtr,'train'),(dva,'val')], evals_result=ev, verbose_eval=False)
    print('Q6', eta, ev['val']['rmse'][-1], min(ev['val']['rmse']))
