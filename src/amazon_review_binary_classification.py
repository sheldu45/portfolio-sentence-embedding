import pandas as pd
import numpy as np
import torch
import os
import scipy
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import umap
from torch.utils.data import Dataset, DataLoader
from simple_ffnn import SimpleFFNN
from simple_ffnn_trainer import SimpleFFNNTrainer
import torch.nn as nn
import torch.optim as optim
from torch import reshape


class AmazonReviewBinaryClassification:
    """
    A class for binary classification of Amazon reviews using sentence embeddings.

    Attributes:
        model (SentenceTransformer): The sentence transformer model.
        corpus (list): List of review texts.
        corpus_labels (list): List of review labels.
        corpus_embeddings (numpy.ndarray): Embeddings of the review texts.
        embed_size (int): Size of the embeddings.
    """

    def __init__(self, model_path="sentence-transformers/all-MiniLM-L6-v2", dataset_name="amazon_polarity", train_ratio=[0.85,0.05,0.1], n_samples_dataset=None):
        """
        Initializes the classification model and parameters.

        Args:
            model_path (str): Path to the sentence transformer model.
            dataset_name (str): Name of the dataset to load.
            train_ratio (list): List containing the ratio of train, validation, and test samples.
            n_samples_dataset (int, optional): Number of samples to use from the dataset. Defaults to None.
        """
        # Path to the sentence transformer model
        self.model_path = model_path
        
        # Name of the dataset to load
        self.dataset_name = dataset_name
        
        # Number of samples to use from the dataset (None means use all samples)
        self.n_samples_dataset = n_samples_dataset
        
        # Load the sentence transformer model
        self.model = SentenceTransformer(self.model_path)
        
        # Initialize corpus attributes to None; these will be populated later
        self.corpus = None
        self.corpus_labels = None
        self.corpus_embeddings = None
        
        # Ratio of training samples to total samples
        self.train_ratio = train_ratio

    def load_data(self):
        """
        Loads and prepares the data from the specified dataset split.

        Args:
            split (str): The dataset split to load (e.g., 'train', 'test').
        """
        # Load the dataset split (e.g., 'train', 'test')
        # Also limits to n_samples_dataset if provided
        if self.n_samples_dataset:
            dataset = load_dataset(self.dataset_name,
                                   split=f'train[:{self.n_samples_dataset}]')
        else:
            dataset = load_dataset(self.dataset_name,
                                   split=f'train')

        ## TODO: get all splits, not only 'train', merge them into a single
        ## dataset object: for that check all features are identical, and
        ## append to each feature all the elements of each split
        # dataset = load_dataset(self.dataset_name)

        # Shuffle the dataset to ensure randomness
        shuffled_dataset = dataset.shuffle(seed=42)
        ## Determine train, test, val proportions vs dataset len
        train_size = int(self.train_ratio[0]*len(shuffled_dataset))
        val_size = int(self.train_ratio[1]*len(shuffled_dataset))
        test_size = int(self.train_ratio[2]*len(shuffled_dataset))
        
        # Extract the review texts and labels from the dataset
        self.train_corpus = shuffled_dataset['content'][:train_size]
        self.train_corpus_labels = shuffled_dataset['label'][:train_size]

        self.val_corpus = shuffled_dataset['content']\
            [train_size:train_size+val_size]
        self.val_corpus_labels = shuffled_dataset['label']\
            [train_size:train_size+val_size]

        self.test_corpus = shuffled_dataset['content']\
            [train_size+val_size:train_size+val_size+test_size]
        self.test_corpus_labels = shuffled_dataset['label']\
            [train_size+val_size:train_size+val_size+test_size]
        
        # Encode the review texts into embeddings using the sentence transformer model
        self.train_corpus_embeddings = self.model.encode(self.train_corpus)
        self.val_corpus_embeddings = self.model.encode(self.val_corpus)
        self.test_corpus_embeddings = self.model.encode(self.test_corpus)

        # Store the size of the embeddings for later use
        self.embed_size = self.train_corpus_embeddings.shape[1]

    def save_data(self, file_path):
        """
        Saves the corpus embeddings and labels to disk.

        Args:
            file_path (str): Base file path to save embeddings and labels.
        """
        # Convert the corpus embeddings to a PyTorch tensor
        tensor = self.to_torch_tensor(self.train_corpus_embeddings)
        # Save the tensor containing the embeddings to a file
        torch.save(tensor, file_path + "_embeds_train.pt")
        # Convert the corpus labels to a PyTorch tensor
        tensor = self.to_torch_tensor(self.train_corpus_labels)
        # Save the tensor containing the labels to a file
        torch.save(tensor, file_path + "_labels_train.pt")

        # Convert the corpus embeddings to a PyTorch tensor
        tensor = self.to_torch_tensor(self.val_corpus_embeddings)
        # Save the tensor containing the embeddings to a file
        torch.save(tensor, file_path + "_embeds_val.pt")
        # Convert the corpus labels to a PyTorch tensor
        tensor = self.to_torch_tensor(self.val_corpus_labels)
        # Save the tensor containing the labels to a file
        torch.save(tensor, file_path + "_labels_val.pt")

        # Convert the corpus embeddings to a PyTorch tensor
        tensor = self.to_torch_tensor(self.test_corpus_embeddings)
        # Save the tensor containing the embeddings to a file
        torch.save(tensor, file_path + "_embeds_test.pt")
        # Convert the corpus labels to a PyTorch tensor
        tensor = self.to_torch_tensor(self.test_corpus_labels)
        # Save the tensor containing the labels to a file
        torch.save(tensor, file_path + "_labels_test.pt")
        
    @staticmethod
    def to_torch_tensor(data):
        """
        Converts data to a PyTorch tensor.

        Args:
            data: Data to convert.

        Returns:
            torch.Tensor: The converted tensor.
        """
        return torch.tensor(data)

    def cluster_data(self, n_clusters=2, random_state=0):
        """
        Clusters the corpus embeddings using KMeans clustering.

        Args:
            n_clusters (int): Number of clusters to form.
            random_state (int): Random state for reproducibility.

        Returns:
            tuple: KMeans object and a dictionary of cluster centers.
        """
        # Perform KMeans clustering on the corpus embeddings
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)\
            .fit(self.train_corpus_embeddings)
        
        # Get the distribution of cluster labels
        cls_dist = pd.Series(kmeans.labels_).value_counts()
        print(cls_dist)

        # Calculate distances between cluster centers and embeddings
        distances = scipy.spatial.distance.cdist(kmeans.cluster_centers_,
                                                 self.train_corpus_embeddings)
        
        # Dictionary to store the index of the closest example to each cluster center
        centers = {}
        print("Cluster", "Size", "Center-idx", "Center-Example", sep="\t\t")
        
        # Iterate over each cluster center's distances to find the closest example
        for i, d in enumerate(distances):
            # Get the index of the closest example to the cluster center
            ind = np.argsort(d, axis=0)[0]
            centers[i] = ind
            # Print cluster information
            print(i, cls_dist[i], ind, self.train_corpus[ind], sep="\t\t")

        # Return the KMeans object and the dictionary of cluster centers
        return kmeans, centers

    def visualize_clusters(self, kmeans, centers, output_file="cluster_visualization.png"):
        """
        Visualizes the clusters using UMAP and saves the plot to a file.

        Args:
            kmeans: KMeans object containing the cluster information.
            centers (dict): Dictionary containing the center indices for each cluster.
            output_file (str): Path to save the cluster visualization.
        """
        # Reduce the dimensionality of the corpus embeddings to 2D using UMAP
        X = umap.UMAP(n_components=2, min_dist=0.0)\
                .fit_transform(self.train_corpus_embeddings)
        
        # Get the cluster labels from the KMeans object
        labels = kmeans.labels_
        print(labels)
        
        # Create a new figure for the plot
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Scatter plot of the 2D embeddings with colors based on cluster labels
        scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, s=1, cmap='Paired')
        
        # Annotate the plot with the cluster centers
        for c in centers:
            ax.text(X[centers[c], 0], X[centers[c], 1], "CLS-" + str(c), fontsize=18)
        
        # Add a color bar to the plot
        plt.colorbar(scatter)
        
        # Save the plot to the specified output file
        plt.savefig(output_file)
        
        # Close the plot to free up memory
        plt.close()


if __name__ == "__main__":
    # Initialize the classifier with a subset of the dataset
    classifier = AmazonReviewBinaryClassification(n_samples_dataset=10000)
    # classifier = AmazonReviewBinaryClassification()
    
    # Load the training data
    classifier.load_data()
    
    # Save the embeddings and labels to disk
    classifier.save_data("corpus")

    # Cluster the data using KMeans
    kmeans, centers = classifier.cluster_data()
    
    # Visualize the clusters and save the plot
    classifier.visualize_clusters(kmeans, centers)

    # Define the input size based on the embedding size and output size for binary classification
    input_size = classifier.embed_size
    output_size = 1  # Binary classification
    
    # Initialize the simple feed-forward neural network
    ffnn = SimpleFFNN(input_size, [10, 10], output_size)

    # Define the loss function and optimizer for training
    criterion = nn.BCELoss()
    optimizer = optim.Adam(ffnn.parameters(), lr=0.001)

    # Initialize the trainer for the neural network
    ffnn_trainer = SimpleFFNNTrainer(ffnn, criterion, optimizer, device='cpu')

    # Convert the embeddings and labels to PyTorch tensors
    train_input_tensor = classifier.to_torch_tensor(classifier.train_corpus_embeddings)
    train_output_tensor = classifier.to_torch_tensor(classifier.train_corpus_labels)
    val_input_tensor = classifier.to_torch_tensor(classifier.val_corpus_embeddings)
    val_output_tensor = classifier.to_torch_tensor(classifier.val_corpus_labels)
    
    # Reshape the output tensor to match the expected dimensions
    train_output_tensor = torch.reshape(train_output_tensor,
                                        (train_output_tensor.shape[0], 1))
    val_output_tensor = torch.reshape(val_output_tensor,
                                      (val_output_tensor.shape[0], 1))

    # Print the shapes of the input and output tensors for verification
    print(f"Train input tensor shape: {train_input_tensor.shape}")
    print(f"Train output tensor shape: {train_output_tensor.shape}")
    print(f"Val input tensor shape: {val_input_tensor.shape}")
    print(f"Val output tensor shape: {val_output_tensor.shape}") 

    # Convert the output tensor to float for compatibility with the loss function
    train_output_tensor = train_output_tensor.float()
    val_output_tensor = val_output_tensor.float()
    
    # Create a dataset and dataloader for batching the data
    train_dataset = list(zip(train_input_tensor, train_output_tensor))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_dataset = list(zip(val_input_tensor, val_output_tensor))
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    # Iterate over the batches and print the input and target data for each batch
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        print(f"##### BATCH #{batch_idx} #####")
        print("Input data (sentence embedding):")
        print(inputs)
        print("Target data (label):")
        print(targets)

    # Train the neural network using the dataloader
    ffnn_trainer.train(train_loader, val_loader=val_loader)
