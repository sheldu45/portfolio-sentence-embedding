import torch
import torch.nn as nn
import torch.optim


class SimpleFFNNTrainer:
    def __init__(self, model, criterion, optimizer, device='cpu'):
        self.model = model ## Model to train
        self.criterion = criterion ## Loss to use for training
        self.optimizer = optimizer ## Optimizer to use for training
        self.device = device ## Device to use for training

        self.model.to(self.device) ## Put model on device to use
        
    def train(self, train_loader, val_loader=None, epochs=10, log_interval=1):

        for epoch in range(epochs):
            self.model.train() ## Put model in training mode
            running_loss = 0.0
            
            ## Iterate over training mini batches
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets, inputs.to(self.device), targets.to(self.device)

                ## Reinit gradients
                self.optimizer.zero_grad()

                ## Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                ## Backward pass and weight updating
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                if batch_idx % log_interval == 0:
                    print(f'Epoch [{epoch+1}/{epochs}], 
                    Step [{batch_idx}/{len(train_loader)}], 
                    Loss: {loss.item():.4f}')

            avg_loss = running_loss / len(train_loader)
            print(f'Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}')

            # If a validation ensemble is given, evaluate the model on it
            if val_loader is not None:
                val_loss = self.evaluate(val_loader)
                print(f'Validation Loss after Epoch {epoch+1}: {val_loss:.4f}')

    def evaluate(self, val_loader):
        self.model.eval()
        val_loss = 0.0

        with torch.no_grad(): ## No gradient calculation during evaluation
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                ## Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item

        avg_val_loss = val_loss / len(val_loader)
        return avg_val_loss

    def save_model(self, filepath):
        torch.save(self.model.state_dict(), filepath)

    def load_model(self, filepath):
        self.model.load_state_dict(torch.load(filepath))
        self.model.to(self.device)
    
