import pandas as pd
import sklearn
from sklearn.cluster import KMeans
import pickle

import os
path = 'C:/k_league_tier'
data = pd.read_csv(os.path.join(path, 'total_ranking.csv'))

km = KMeans(n_clusters = 10)
km.fit(data)
pred = km.predict(data)

with open('tier_model.pkl', 'wb') as f:
    pickle.dump(km, f)
    
c_data = data.copy()
c_data['티어'] = pred
c_data