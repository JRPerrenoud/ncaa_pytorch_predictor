"""
NCAA Tournament Prediction Model

This script defines a PyTorch model for predicting NCAA basketball tournament outcomes.
It includes:
1. Model architecture definition
2. Training and evaluation functions
3. Model saving and loading utilities
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from scripts.data_loader import NCAADataset, create_data_loaders

# Set paths
model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
os.makedirs(model_dir, exist_ok=True)

class NCAAPredictor(nn.Module):
    """Neural network model for predicting NCAA tournament outcomes."""
    
    def __init__(self, input_size, hidden_size=64, dropout_rate=0.3):
        """
        Initialize the model.
        
        Args:
            input_size: Number of input features
            hidden_size: Size of hidden layers
            dropout_rate: Dropout rate for regularization
        """
        super(NCAAPredictor, self).__init__()
        
        # Define the network architecture
        self.model = nn.Sequential(
            # First hidden layer
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Second hidden layer
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Output layer
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """Forward pass through the network."""
        return self.model(x).squeeze()

def train_model(train_loader, val_loader, input_size, epochs=50, lr=0.001, weight_decay=1e-4):
    """
    Train the NCAA prediction model.
    
    Args:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        input_size: Number of input features
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: L2 regularization strength
    
    Returns:
        model: Trained model
        history: Training history (loss and metrics)
    """
    # Initialize model, loss function, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NCAAPredictor(input_size).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Initialize history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'val_auc': []
    }
    
    # Training loop
    best_val_acc = 0
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_preds = []
        train_targets = []
        
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track loss and predictions
            train_loss += loss.item()
            train_preds.extend((outputs > 0.5).cpu().numpy())
            train_targets.extend(targets.cpu().numpy())
        
        # Calculate training metrics
        train_loss /= len(train_loader)
        train_acc = accuracy_score(train_targets, train_preds)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_preds = []
        val_probs = []
        val_targets = []
        
        with torch.no_grad():
            for features, targets in val_loader:
                features, targets = features.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(features)
                loss = criterion(outputs, targets)
                
                # Track loss and predictions
                val_loss += loss.item()
                val_probs.extend(outputs.cpu().numpy())
                val_preds.extend((outputs > 0.5).cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
        
        # Calculate validation metrics
        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_targets, val_preds)
        val_auc = roc_auc_score(val_targets, val_probs) if len(set(val_targets)) > 1 else 0.5
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)
        
        # Print progress
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, os.path.join(model_dir, 'best_model.pt'))
    
    # Save final model
    save_model(model, os.path.join(model_dir, 'final_model.pt'))
    
    return model, history

def evaluate_model(model, data_loader):
    """
    Evaluate the model on a dataset.
    
    Args:
        model: Trained model
        data_loader: DataLoader for evaluation data
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    all_probs = []
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for features, targets in data_loader:
            features, targets = features.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(features)
            
            # Track predictions
            all_probs.extend(outputs.cpu().numpy())
            all_preds.extend((outputs > 0.5).cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds)
    auc = roc_auc_score(all_targets, all_probs) if len(set(all_targets)) > 1 else 0.5
    conf_matrix = confusion_matrix(all_targets, all_preds)
    
    # Calculate additional metrics
    tn, fp, fn, tp = conf_matrix.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # True positive rate
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # True negative rate
    
    metrics = {
        'accuracy': accuracy,
        'auc': auc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'confusion_matrix': conf_matrix
    }
    
    return metrics

def save_model(model, path):
    """Save model to disk."""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(path, input_size):
    """Load model from disk."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NCAAPredictor(input_size).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

def plot_training_history(history):
    """Plot training and validation metrics."""
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.plot(history['val_auc'], label='Val AUC')
    plt.xlabel('Epoch')
    plt.ylabel('Metric')
    plt.title('Training and Validation Metrics')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'training_history.png'))
    plt.close()

def predict_matchup(model, team1_stats, team2_stats, feature_cols):
    """
    Predict the outcome of a matchup between two teams.
    
    Args:
        model: Trained model
        team1_stats: Statistics for team 1
        team2_stats: Statistics for team 2
        feature_cols: List of feature columns used by the model
    
    Returns:
        prob: Probability of team 1 winning
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    # Extract features
    team1_features = team1_stats[feature_cols].values
    team2_features = team2_stats[feature_cols].values
    
    # Create feature vector (difference between team1 and team2)
    features = team1_features - team2_features
    
    # Convert to PyTorch tensor
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        prob = model(features_tensor).item()
    
    return prob

if __name__ == "__main__":
    # Create data loaders
    train_loader, val_loader = create_data_loaders(batch_size=16)
    
    # Get input size from dataset
    input_size = len(train_loader.dataset.get_feature_names())
    
    print(f"Training model with {input_size} features...")
    
    # Train model
    model, history = train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        input_size=input_size,
        epochs=50,
        lr=0.001,
        weight_decay=1e-4
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate model on validation set
    print("\nEvaluating model on validation set...")
    metrics = evaluate_model(model, val_loader)
    
    print(f"Validation Accuracy: {metrics['accuracy']:.4f}")
    print(f"Validation AUC: {metrics['auc']:.4f}")
    print(f"Sensitivity (True Positive Rate): {metrics['sensitivity']:.4f}")
    print(f"Specificity (True Negative Rate): {metrics['specificity']:.4f}")
    print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
    
    # Save metrics
    with open(os.path.join(model_dir, 'metrics.json'), 'w') as f:
        json.dump({k: v.tolist() if isinstance(v, np.ndarray) else v 
                  for k, v in metrics.items()}, f, indent=4)
    
    print("\nModel training and evaluation complete!")
