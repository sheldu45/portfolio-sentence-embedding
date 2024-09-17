import pandas as pd, numpy as np
import torch, os, scipy
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import umap

class AmazonReviewBinaryClassification:

    def __init__(self, model_path="sentence-transformers/all-MiniLM-L6-v2", dataset_name="amazon_polarity", limit=10000):
        self.model_path = model_path
        self.dataset_name = dataset_name
        self.limit = limit
        self.model = SentenceTransformer(self.model_path)
        self.corpus = None
        self.corpus_embeddings = None
        self.X = None

    def load_data(self, split):
        dataset = load_dataset(self.dataset_name, split=split)
        self.corpus = dataset.shuffle(seed=42)[:self.limit]['content']
        self.corpus_embeddings = self.model.encode(self.corpus)

    def save_data(self, file_path):
        tensor = torch.tensor(self.corpus_embeddings)
        torch.save(tensor, file_path)
        
    def to_torch_tensor(self):
        return torch.tensor(self.corpus_embeddings)

    def cluster_data(self, n_clusters=2, random_state=0):
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(self.corpus_embeddings)
        cls_dist = pd.Series(kmeans.labels_).value_counts()
        print(cls_dist)

        distances = scipy.spatial.distance.cdist(kmeans.cluster_centers_, self.corpus_embeddings)
        centers = {}
        print("Cluster", "Size", "Center-idx", "Center-Example", sep="\t\t")
        for i, d in enumerate(distances):
            ind = np.argsort(d, axis=0)[0]
            centers[i] = ind
            print(i, cls_dist[i], ind, self.corpus[ind], sep="\t\t")
            
        return kmeans, centers
            
    def visualize_clusters(self, kmeans, centers, output_file="cluster_visualization.png"):
        X = umap.UMAP(n_components=2, min_dist=0.0).fit_transform(self.corpus_embeddings)
        labels = kmeans.labels_
        print(labels)
        fig, ax = plt.subplots(figsize=(12, 12))
        print(X[:, 0])
        scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, s=1, cmap='Paired')
        for c in centers:
            ax.text(X[centers[c], 0], X[centers[c], 1], "CLS-" + str(c), fontsize=18)
        plt.colorbar(scatter)
        plt.savefig(output_file)
        plt.close()

if __name__ == "__main__":
    classifier = AmazonReviewBinaryClassification(limit=100)
    classifier.load_data(split="train")
    classifier.save_data("corpus_embeddings.pt")
    kmeans, centers = classifier.cluster_data()
    classifier.visualize_clusters(kmeans, centers)



'''
dataset = load_dataset("amazon_polarity",split="train")

limit = 100
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
plt.show()'''