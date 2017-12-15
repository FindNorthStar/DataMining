import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from sklearn.manifold import TSNE

def scatter(x, colors,n):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", n))
    # We create a scatter plot.
    f = plt.figure(figsize=(16, 16))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[colors.astype(np.int)])
    plt.xlim(-50, 50)
    plt.ylim(-50, 50)
    ax.axis('off')
    ax.axis('tight')
    # We add the labels for each digit.
    txts = []
    for i in range(n):
        # Position of each label.
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)
    return f, ax, sc, txts


#===========================songs============================#

cluster_songs_df = pd.read_csv('data/kmeans_song.csv', nrows=50000)
songs_df = pd.read_csv('data/temp_song.csv', nrows=50000)

songs_df = songs_df.merge(cluster_songs_df, on='song_id', how='left')
songs_df = songs_df.sort_values(by=["clusterId"],ascending=True)
X = songs_df[['genre_ids', 'genre_count', 'artistId', 'artist_count', 'composerId', 'composer_count', 'lyricistId', 'lyricist_count', 'song_year','language','song_length', 'play_count','replay_count', 'replay_pb']]
#X = songs_df[['genre_ids', 'genre_count','artist_count', 'composer_count','lyricist_count', 'song_year','language','song_length', 'play_count','replay_count', 'replay_pb']]
y = songs_df['clusterId']
X = np.array(X)
y = np.array(y)

print(X[0])
print(y)
print(type(X))
print(type(y))

RS = 20150101
songs_proj = TSNE(random_state=RS).fit_transform(X)
print(songs_proj[0])
print(y)
print(type(songs_proj))
print(type(y))
scatter(songs_proj, y,10)
plt.show()
plt.savefig('songs_tsne-generated.png', dpi=120)

cluster_songs_df = cluster_songs_df[['clusterId','song_id']].groupby('clusterId').agg(['count']).reset_index()
cluster_songs_df.columns = ['clusterId', 'count']

g = sns.barplot(x='clusterId', y='count', data=cluster_songs_df, order=cluster_songs_df.sort_values(by=['count'], ascending=False)['clusterId'])
g.set_yscale('log', nonposy='clip')
# plt.show()
plt.savefig('songs_clusterId.png', dpi=120)
