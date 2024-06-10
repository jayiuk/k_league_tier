import xgboost as xgb
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split


from dotenv import load_dotenv
load_dotenv(verbose = True)
path = os.getenv('url')

data = pd.read_csv(os.path.join(path, 'tier.csv'))
y_data = data['target']
x_data = data.drop(['target'], axis = 1)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.2, random_state = 42)

dtrain = xgb.DMatrix(data = x_train, label = y_train)
dtest = xgb.DMatrix(data = x_test, label = y_test)

params = {'max_depth' : 6, 'eta' : 0.5, 'objective' : 'multi:softmax', 'eval_metrics' : 'merror', 'num_class' : 5}

num_rounds = 500

xlist = [(dtrain, 'train'), (dtest, 'eval')]
xgb_m = xgb.train(params = params, dtrain = dtrain, num_boost_round = num_rounds,  evals = xlist)

name = 'xgb_tier.model'
pickle.dump(xgb_m, open(name, 'wb'))