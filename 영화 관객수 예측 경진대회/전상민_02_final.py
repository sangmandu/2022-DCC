import pandas as pd
import numpy as np

from collections import defaultdict

from catboost import CatBoostClassifier, CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from ngboost import NGBRegressor

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold

train_df = pd.read_csv('new_movies_train.csv')
test_df = pd.read_csv('new_movies_test.csv')
submit = pd.read_csv('submission.csv')

model_list = [
    (CatBoostRegressor, {'loss_function':'RMSE', 'silent':True,}),
    (LGBMRegressor, {'verbose':-1, }),
    (XGBRegressor, {'silent':True, 'verbosity':0, }),
    (GradientBoostingRegressor, {},),
    (RandomForestRegressor, {},),
    (NGBRegressor, {'verbose':0,},),
]

xgb_args = [{}, {'subsample': 0.95}, {'subsample': 0.9}, {'colsample_bytree': 0.95}, {'colsample_bytree': 0.9}]

seeds = [42, 41, 1024, 4242, 2022]

result = defaultdict(list)

def run(train_x, train_y, valid_x, valid_y, test):
    ens_pred = np.zeros((test.shape[0]))
    
    for m, arg in model_list:
        name = str(m)
        
        avg_score = 0
        for idx, seed in enumerate(seeds):
            if name == XGBRegressor:
                model = m(**arg, **xgb_args[idx])
            else:
                model = m(**arg, random_state=seed)
                
            model.fit(train_x, train_y)

            pred = model.predict(valid_x)
            pred = [p if p >= 0 else 0 for p in pred]
            
            score = mean_squared_error(valid_y.values, pred) ** 0.5
            avg_score += score
            result[name].append(score)
            
            pred = model.predict(test)
            pred = [p if p >= 0 else 0 for p in pred]
            ens_pred += pred
        
        print(f"{name} | score {avg_score / len(seeds)}")
    
    ens_pred /= len(model_list) * 5
    return ens_pred
    

_train_df = train_df.copy()
_test_df = test_df.copy()

exclude_cols = ['distributor_3way', 'distributor_2way', 'distributor_complex', 'dir_prev_bfnum2']

_train_df.drop(columns=exclude_cols, inplace=True)
_test_df.drop(columns=exclude_cols, inplace=True)

mean_dir_prev_bfnum = train_df[~train_df.dir_prev_bfnum.isna()].dir_prev_bfnum.mean()

_train_df.loc[train_df.dir_prev_bfnum.isna(), 'dir_prev_bfnum'] = mean_dir_prev_bfnum
_test_df.loc[test_df.dir_prev_bfnum.isna(), 'dir_prev_bfnum'] = mean_dir_prev_bfnum

x, y = _train_df.drop(columns=['box_off_num']), _train_df['box_off_num']

n_folds = 10
kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
for idx, (train_idx, valid_idx) in enumerate(kf.split(x, y), 1):
    print(f"Fold {idx}")
    train_x, train_y = x.iloc[train_idx], y.iloc[train_idx]
    valid_x, valid_y = x.iloc[valid_idx], y.iloc[valid_idx]

    submit['box_off_num'] += run(train_x, train_y, valid_x, valid_y, _test_df)

print("\n Average score of model")
for m, _,in model_list:
    name = str(m)
    print(name, sum(result[name]) / len(result[name]))

submit['box_off_num'] /= n_folds
submit.to_csv(f'ensemble_{len(seeds)}seeds_{n_folds}folds.csv', index=False)