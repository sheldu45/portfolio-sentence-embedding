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

    def __init__(self, model_path="sentence-transformers/all-MiniLM-L6-v2", dataset_name="amazon_polarity", batch_size=32, train_ratio=[0.85,0.05,0.1], n_samples_dataset=None):
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

        # Batch size for training
        self.batch_size = batch_size
        
    def load_data(self, verbose=False):
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

        
        # Prepare the Dataloader data for train, validation, and test sets
        self.train_loader = self.create_dataloader(self.train_corpus_embeddings,
                                                   self.train_corpus_labels,
                                                   name_set="train",
                                                   verbose=verbose)
        self.val_loader = self.create_dataloader(self.val_corpus_embeddings,
                                                 self.val_corpus_labels,
                                                 name_set="val",
                                                 verbose=verbose)
        self.test_loader = self.create_dataloader(self.test_corpus_embeddings,
                                                  self.test_corpus_labels,
                                                  name_set="test",
                                                  verbose=verbose)

    def create_dataloader(self, embeddings, labels, name_set="", verbose=False):
        """
        Converts embeddings and labels to PyTorch tensors and reshapes the labels.

        Args:
        embeddings (numpy.ndarray): The embeddings to convert.
        labels (numpy.ndarray): The labels to convert.
        verbose (bool): If True, prints additional information during preparation.

        Returns:
        tuple: A tuple containing the input tensor and reshaped output tensor.
        """
        input_tensor = self.to_torch_tensor(embeddings)
        output_tensor = self.to_torch_tensor(labels)
        output_tensor = torch.reshape(output_tensor, (output_tensor.shape[0], 1))

        if verbose:
            print(f"{name_set} input tensor shape: {input_tensor.shape}")
            print(f"{name_set} output tensor shape: {output_tensor.shape}")

        # Create a dataset and dataloader for batching the data
        dataset = list(zip(input_tensor, output_tensor.float()))
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
    def save_data(self, file_path):
        """
        Saves the corpus embeddings and labels to disk.

        Args:
            file_path (str): Base file path to save embeddings and labels.
        """
        def save_split_data(embeddings, labels, split_name):
            """
            Saves the embeddings and labels for a specific split to disk.

            Args:
            embeddings (numpy.ndarray): The embeddings to save.
            labels (numpy.ndarray): The labels to save.
            split_name (str): The name of the split (e.g., 'train', 'val', 'test').
            """
            # Convert the embeddings to a PyTorch tensor and save to file
            tensor = self.to_torch_tensor(embeddings)
            torch.save(tensor, f"{file_path}_embeds_{split_name}.pt")
            # Convert the labels to a PyTorch tensor and save to file
            tensor = self.to_torch_tensor(labels)
            torch.save(tensor, f"{file_path}_labels_{split_name}.pt")

        # Save the train, validation, and test data
        save_split_data(self.train_corpus_embeddings, self.train_corpus_labels, "train")
        save_split_data(self.val_corpus_embeddings, self.val_corpus_labels, "val")
        save_split_data(self.test_corpus_embeddings, self.test_corpus_labels, "test")
        
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

    def cluster_and_visualize_kmeans(self):
        """
        Clusters and visualizes the data using KMeans.
        """
        # Cluster the data using KMeans
        kmeans, centers = self.cluster_data()
            
        # Visualize the clusters and save the plot
        self.visualize_data(kmeans.labels_, centers=centers, 
                                             output_file="kmeans_classification.png")

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
            .fit(self.test_corpus_embeddings)
        
        # Get the distribution of cluster labels
        cls_dist = pd.Series(kmeans.labels_).value_counts()
        print(cls_dist)

        # Calculate distances between cluster centers and embeddings
        distances = scipy.spatial.distance.cdist(kmeans.cluster_centers_,
                                                 self.test_corpus_embeddings)
        
        # Dictionary to store the index of the closest example to each cluster center
        centers = {}
        print("Cluster", "Size", "Center-idx", "Center-Example", sep="\t\t")
        
        # Iterate over each cluster center's distances to find the closest example
        for i, d in enumerate(distances):
            # Get the index of the closest example to the cluster center
            ind = np.argsort(d, axis=0)[0]
            centers[i] = ind
            # Print cluster information
            print(i, cls_dist[i], ind, self.test_corpus[ind], sep="\t\t")

        # Return the KMeans object and the dictionary of cluster centers
        return kmeans, centers

    def visualize_data(self, labels, centers=[], output_file="cluster_visualization.png"):
        """
        Visualizes the embeddings and labels using UMAP and saves the plot to a file.

        Args:
            kmeans: KMeans object containing the cluster information.
            centers (dict): Dictionary containing the center indices for each cluster.
            output_file (str): Path to save the cluster visualization.
        """
        # Reduce the dimensionality of the corpus embeddings to 2D using UMAP
        X = umap.UMAP(n_components=2, min_dist=0.0, random_state=42)\
                .fit_transform(self.test_corpus_embeddings)
        
        # Get the cluster labels from the KMeans object
        # labels = kmeans.labels_
        # print(labels)
        
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

    def train_ffnn_classifier(self, layers=[10, 10], epochs=10, verbose=False):
        """
        Trains and evaluates the FFNN classifier.

        Args:
            verbose (bool): If True, prints additional information during training.
        """
        # Define the input size based on the embedding size and output size for binary classification
        input_size = self.embed_size
        output_size = 1  # Binary classification
        
        # Initialize the simple feed-forward neural network
        ffnn = SimpleFFNN(input_size, layers, output_size)

        # Define the loss function and optimizer for training
        criterion = nn.BCELoss()
        optimizer = optim.Adam(ffnn.parameters(), lr=0.0001)


        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {device} for training")

        # Initialize the trainer for the neural network
        ffnn_trainer = SimpleFFNNTrainer(ffnn, criterion, optimizer, device=device)

        # Train the neural network using the dataloader
        ffnn_trainer.train(self.train_loader, verbose=verbose,
                           val_loader=self.val_loader, epochs=epochs)

        # If a test set is given, evaluate the model on it
        if self.test_loader is not None:
            test_loss, labels = ffnn_trainer.evaluate(self.test_loader,
                                                      return_outputs=True)
            print(f'Test Loss after Training : {test_loss:.4f}')

            # Convert outputs to binary predictions
            labels = (labels > 0.5).float()
            labels = labels.cpu()

            # Visualize the clusters and save the plot
            self.visualize_data(labels, output_file="ffnn_classification.png")

            # Extract true labels from the data_loader
            true_labels = torch.cat([labels for _, labels in self.test_loader], dim=0)

            self.visualize_data(true_labels, output_file="ground_truth.png")
            
        return ffnn, ffnn_trainer

    def evaluate_ffnn_classifier(self, ffnn_trainer, data_loader):
        """
        Evaluates the FFNN classifier and measures precision, recall, and F1 score.

        Args:
            data_loader (DataLoader): DataLoader for the dataset to evaluate.

        Returns:
            dict: Dictionary containing precision, recall, and F1 score.
        """
        # Define the input size based on the embedding size and output size for binary classification
        input_size = self.embed_size
        output_size = 1  # Binary classification

        # Evaluate the model on the provided data_loader using the FFNNTrainer's evaluate function to get outputs 
        test_loss, outputs = ffnn_trainer.evaluate(data_loader, return_outputs=True)
        
        # Convert outputs to binary predictions
        predictions = (outputs > 0.5).float()
        predictions = predictions.cpu()
        
        # Extract true labels from the data_loader
        true_labels = torch.cat([labels for _, labels in data_loader], dim=0)
        
        # Calculate confusion matrix components
        true_positive = (predictions * true_labels).sum().item()
        true_negative = ((1 - predictions) * (1 - true_labels)).sum().item()
        false_positive = (predictions * (1 - true_labels)).sum().item()
        false_negative = ((1 - predictions) * true_labels).sum().item()
        
        # Calculate precision and recall
        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        accuracy = (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative) if (true_positive + true_negative + false_positive + false_negative) > 0 else 0

        # Calculate F1 score
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Create confusion matrix with precision, recall, and F1 score
        confusion_matrix = {
            "true_positive": true_positive,
            "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            "f1_score": f1_score
        }
        
        return confusion_matrix
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Amazon Review Binary Classification')
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--use-gpu', action='store_true')
    parser.add_argument('--compute-kmeans', action='store_true')
    parser.add_argument('--ffnn-arch', nargs='+', type=int, default=[10,10])
    parser.add_argument('--n-samples-dataset', type=int, default=1000)
    parser.add_argument('--n-epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    
    args = parser.parse_args()

    # Initialize the classifier with a subset of the dataset
    classifier = AmazonReviewBinaryClassification(n_samples_dataset=args.n_samples_dataset,
                                                  batch_size=args.batch_size)
    
    # Load the training data
    classifier.load_data(verbose=args.verbose)
    
    # Save the embeddings and labels to disk
    classifier.save_data("corpus")

    if args.compute_kmeans:
        # Cluster and visualize the data using KMeans
        classifier.cluster_and_visualize_kmeans()

    # Train and evaluate the FFNN classifier
    ffnn, ffnn_trainer = classifier.train_ffnn_classifier(layers=args.ffnn_arch,
                                                          verbose=args.verbose,
                                                          epochs=args.n_epochs)
    
    test_confusion_matrix = classifier.evaluate_ffnn_classifier(ffnn_trainer,
                                                                classifier.test_loader)

    print("#### TEST stats ####")
    print(f"Precision\t= {test_confusion_matrix['precision']:.4f}")
    print(f"Recall   \t= {test_confusion_matrix['recall']:.4f}")
    print(f"Accuracy \t= {test_confusion_matrix['accuracy']:.4f}")
    train_confusion_matrix = classifier.evaluate_ffnn_classifier(ffnn_trainer,
                                                                 classifier.train_loader)
    print("#### TRAIN stats ####")
    print(f"Precision\t= {train_confusion_matrix['precision']:.4f}")
    print(f"Recall   \t= {train_confusion_matrix['recall']:.4f}")
    print(f"Accuracy \t= {train_confusion_matrix['accuracy']:.4f}")
