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
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[colors.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
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


#===========================members============================#

cluster_member_df = pd.read_csv('data/kmeans_member.csv')
member_df = pd.read_csv('data/temp_member.csv')

member_df = member_df.merge(cluster_member_df, on='msno', how='left')
member_df = member_df.sort_values(by=["clusterId"],ascending=True)
X = member_df[['city','bd','gender','registered_via','registration_year','registration_month','registration_day','expiration_year','expiration_month','expiration_day','membership_days','replay_pb','play_count','replay_count']]
y = member_df['clusterId']
X = np.array(X)
y = np.array(y)

print(X[0])
print(y)
print(type(X))
print(type(y))

RS = 20171214
songs_proj = TSNE(random_state=RS).fit_transform(X)
scatter(songs_proj, y,20)
plt.savefig('member_tsne-generated.png', dpi=120)

cluster_member_df = cluster_member_df[['clusterId','msno']].groupby('clusterId').agg(['count']).reset_index()
cluster_member_df.columns = ['clusterId', 'count']

g = sns.barplot(x='clusterId', y='count', data=cluster_member_df, order=cluster_member_df.sort_values(by=['count'], ascending=False)['clusterId'])
g.set_yscale('log', nonposy='clip')
# plt.show()
plt.savefig('members_clusterId.png', dpi=120)