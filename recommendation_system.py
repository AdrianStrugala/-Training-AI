#System rekomendacyjny, ktory podaje uczelnie z najbardziej podobnym poziomem nauczania informatyki do podanego

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import re
from sklearn.neighbors import NearestNeighbors as nn

file = "uczelnie.json"

df = pd.read_json(file)

features = ['nazwa', 'opka', 'ela', 'op']

df = df[features]

#dla czytelnosci nazwy - zamien taki html na pusta linie
df['nazwa'] = [re.sub("<.*?>", "", text) for text in df['nazwa']]

print(df)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel('Ocena przez kadre akademicka')
ax.set_ylabel('Losy absolwentow')
ax.set_zlabel('Oceny parametryczna')

ax.scatter(df['opka'], df['ela'], df['op'])

for index, row in df.iterrows():
    ax.text(row['opka'], row['ela'], row['op'], row['nazwa'])

#plt.show()


X = df.iloc[:, 1:4] #iloc - integer location
y = df.iloc[:, 0]

nbrs = nn(n_neighbors= 2)
model = nbrs.fit(X)

#dist, idx = model.kneighbors(X)

recommendations = model.kneighbors([[100, 100, 100]], 5, return_distance=False)

for idx in recommendations:
    print(y[idx])