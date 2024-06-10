import pandas as pd
from sklearn.cluster import KMeans
import pickle

import os
path = 'C:/k_league_tier'
data = pd.read_csv(os.path.join(path, 'total_ranking.csv'))

km = KMeans(n_clusters = 5)
km.fit(data)
pred = km.predict(data)


c_data = data.copy()
c_data['target'] = pred
c_data
c_data.to_csv('tier.csv', index = False)

