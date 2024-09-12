import pandas as pd, numpy as np
import torch, os, scipy
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import umap

dataset = load_dataset("amazon_polarity",split="train")

limit = 1000
corpus=dataset.shuffle(seed=42)[:limit]['content']

model_path="sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_path)

# Corpus with example sentences
corpus_embeddings = model.encode(corpus)

k = 2
kmeans = KMeans(n_clusters=k,random_state=0).fit(corpus_embeddings)
cls_dist=pd.Series(kmeans.labels_).value_counts()
print(cls_dist)


# Find one real sentence closest to each centroid
distances = scipy.spatial.distance.cdist(kmeans.cluster_centers_,corpus_embeddings)
centers={}
print("Cluster", "Size", "Center-idx","Center-Example", sep="\t\t")
for i,d in enumerate(distances):
    ind = np.argsort(d, axis=0)[0]
    centers[i]=ind
    print(i,cls_dist[i], ind, corpus[ind] ,sep="\t\t")

## Reduce dimensionality and visualize clusters
X = umap.UMAP(n_components=2,min_dist=0.0).fit_transform(corpus_embeddings)
labels= kmeans.labels_
print(labels)
fig, ax = plt.subplots(figsize=(12,12))
print(X[:,0])
plt.scatter(X[:,0], X[:,1], c=labels, s=1, cmap='Paired')
for c in centers:
    plt.text(X[centers[c],0], X[centers[c], 1],"CLS-"+ str(c), fontsize=18)
plt.colorbar()
plt.show()


## WIP

# # Reduce dimensionality to 2D
# pca = PCA(n_components=2)
# corpus_embeddings_2d = pca.fit_transform(corpus_embeddings)

# # Plot the KMeans clusters
# plt.scatter(corpus_embeddings_2d[:, 0], corpus_embeddings_2d[:, 1], c=kmeans.labels_)
# plt.title("KMeans Clusters")
# plt.xlabel("Dimension 1")
# plt.ylabel("Dimension 2")
# plt.show()
