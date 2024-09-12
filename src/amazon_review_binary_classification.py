import pandas as pd, numpy as np
import torch, os, scipy
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

dataset = load_dataset("amazon_polarity",split="train")

limit = 10000
corpus=dataset.shuffle(seed=42)[:limit]['content']

model_path="sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_path)

# Corpus with example sentences
corpus_embeddings = model.encode(corpus)
corpus_embeddings.shape

k = 2
kmeans = KMeans(n_clusters=k,random_state=0).fit(corpus_embeddings)
cls_dist=pd.Series(kmeans.labels_).value_counts()


# Reduce dimensionality to 2D
pca = PCA(n_components=2)
corpus_embeddings_2d = pca.fit_transform(corpus_embeddings)

# Plot the KMeans clusters
plt.scatter(corpus_embeddings_2d[:, 0], corpus_embeddings_2d[:, 1], c=kmeans.labels_)
plt.title("KMeans Clusters")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.show()