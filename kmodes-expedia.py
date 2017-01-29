#!/usr/bin/env python

import numpy as np
from kmodes import kprototypes

# stocks with their market caps, sectors and countries
syms = np.genfromtxt('train.csv', dtype=str, delimiter=',')[:, 0:15]
X = np.genfromtxt('train.csv', dtype=object, delimiter=',')[:, 1:15]

kproto = kmodes.KModes(n_clusters=100, init='Cao', verbose=2)
clusters = kproto.fit_predict(X, categorical=[1, 2])

# Print cluster centroids of the trained model.
print(kproto.cluster_centroids_)
# Print training statistics
print(kproto.cost_)
print(kproto.n_iter_)
f = open('/home/chirag/Music/clustersKmodes','w')
f.write("user_id,date_time,site_name,posa_continent,user_location_country,user_location_region,user_location_city,orig_destination_distance,is_mobile,is_package,channel,srch_ci,srch_co,srch_adults_cnt,srch_children_cnt,srch_rm_cnt")
for s, c in zip(syms, clusters):
    f.write("Symbol, cluster:{}".format(s, c))
