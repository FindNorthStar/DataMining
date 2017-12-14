import numpy as np
from kmodes.kprototypes import KPrototypes



lines = open('data/temp_song.csv','r').readlines()
open('data/temp_song2.csv','w').writelines(lines[1:])

# stocks with their market caps, sectors and countries
syms = np.genfromtxt('data/temp_song2.csv', dtype=str, delimiter=',')[:, 1]
X = np.genfromtxt('data/temp_song2.csv', dtype=object, delimiter=',')[:, 2:]

X[:, 1:] = X[:, 1:].astype(float)

kproto = KPrototypes(n_clusters=50, init='Cao', verbose=2)

clusters = kproto.fit_predict(X,categorical=[0])
# Print cluster centroids of the trained model.
print(kproto.cluster_centroids_)
# Print training statistics
print(kproto.cost_)
print(kproto.n_iter_)

output = open('data/kmeans_song.csv', 'w')
output.writelines("msno,clusterId\n")
for s, c in zip(syms, clusters):
    print("Symbol: {}, cluster:{}".format(s, c))
    output.writelines("{},{}\n".format(s, c))

output.close()
