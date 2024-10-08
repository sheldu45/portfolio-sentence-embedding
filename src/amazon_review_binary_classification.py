import pandas as pd, numpy as np
import torch, os, scipy
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import umap

class AmazonReviewBinaryClassification:

    def __init__(self, model_path="sentence-transformers/all-MiniLM-L6-v2", dataset_name="amazon_polarity", n_samples_dataset=10000):
        self.model_path = model_path
        self.dataset_name = dataset_name
        self.n_samples_dataset = n_samples_dataset
        self.model = SentenceTransformer(self.model_path)
        self.corpus = None
        self.corpus_labels = None
        self.corpus_embeddings = None

    def load_data(self, split):
        dataset = load_dataset(self.dataset_name, split=split)
        shuffled_dataset = dataset.shuffle(seed=42)
        print(f"TOTAL DATASET LEN: {len(dataset)}")
        self.corpus = shuffled_dataset[:self.n_samples_dataset]['content']
        self.corpus_labels = shuffled_dataset[:self.n_samples_dataset]['label']
        self.corpus_embeddings = self.model.encode(self.corpus)
        self.embed_size = self.corpus_embeddings.shape[1]

    def save_data(self, file_path):
        tensor = self.to_torch_tensor(self.corpus_embeddings)
        torch.save(tensor, file_path + "_embeds.pt")
        tensor = self.to_torch_tensor(self.corpus_labels)
        torch.save(tensor, file_path + "_labels.pt")
        
    def to_torch_tensor(self, data):
        return torch.tensor(data)

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
    classifier = AmazonReviewBinaryClassification(n_samples_dataset=10000)
    classifier.load_data(split="train")
    classifier.save_data("corpus")

    ## Cluster data using kmeans
    # kmeans, centers = classifier.cluster_data()
    # classifier.visualize_clusters(kmeans, centers)

    ## Create our simple feed forward neural network
    from simple_ffnn import SimpleFFNN

    input_size = classifier.embed_size
    output_size = 1 ## Binary classification
    ffnn = SimpleFFNN(input_size, [10,10], output_size)

    ## Create trainer for the neural network
    from simple_ffnn_trainer import SimpleFFNNTrainer
    import torch.nn as nn
    import torch.optim as optim
    from torch import reshape
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(ffnn.parameters(), lr=0.001)
    
    ffnn_trainer = SimpleFFNNTrainer(ffnn, criterion, optimizer, device='cpu')
    ## Create dataloaders using amazon polarity data
    from torch.utils.data import Dataset, DataLoader

    input_tensor = classifier.to_torch_tensor(classifier.corpus_embeddings)
    output_tensor = classifier.to_torch_tensor(classifier.corpus_labels)
    output_tensor = torch.reshape(output_tensor, (output_tensor.shape[0], 1))

    print(f"input tensor shape: {input_tensor.shape}")
    print(f"output tensor shape: {output_tensor.shape}")

    class CustomDataset(Dataset):
        def __init__(self, inputs, targets):
            self.inputs = inputs
            self.targets = targets

        def __len__(self):
            return len(self.inputs)

        def __getitem__(self, idx):
            input_data = self.inputs[idx]
            target_data = self.targets[idx]
            return input_data, target_data

    output_tensor = output_tensor.float()
    dataset = CustomDataset(input_tensor, output_tensor)
    # tensor_dataset_output = classifier.to_torch_tensor(classifier.corpus_labels)
    # train_loader = DataLoader((input_tensor, output_tensor), batch_size=32, shuffle=True)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        print(f"##### BATCH #{batch_idx} #####")
        print("Input data (sentence embedding):")
        print(inputs)
        print("Target data (label):")
        print(targets)

    ffnn_trainer.train(train_loader)
        
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
