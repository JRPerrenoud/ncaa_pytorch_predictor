"""
Round 2 Specific NCAA Tournament Prediction Model Training with Anti-Overfitting Techniques

This script trains a model specifically for predicting Round 2 matchups in the NCAA tournament
with techniques to prevent overfitting, including:
1. Increased regularization (higher dropout and weight decay)
2. Data augmentation
3. Early stopping
4. Feature selection
5. Simplified model architecture
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import KFold

# Add parent directory to path to handle imports properly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.data_loader import NCAADataset, create_data_loaders
from scripts.model import train_model, evaluate_model, plot_training_history, save_model, load_model

# Set paths
data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
round2_model_path = os.path.join(model_dir, 'round2_model_robust.pt')

def load_tournament_data(year):
    """Load tournament data for a specific year."""
    results_path = os.path.join(data_dir, f'results_{year}.json')
    with open(results_path, 'r') as f:
        return json.load(f)

def prepare_round2_data(years_range):
    """
    Prepare training data specifically for Round 2 matchups.
    
    Args:
        years_range: Range of years to include
        
    Returns:
        round2_data: List of Round 2 matchups
    """
    round2_data = []
    
    for year in years_range:
        # Skip 2020 due to COVID cancellation
        if year == 2020:
            continue
        
        try:
            # Load tournament data for this year
            tournament_data = load_tournament_data(year)
            
            # Filter for Round 2 matchups
            year_round2_data = [match for match in tournament_data if match['round'] == 2]
            
            print(f"Found {len(year_round2_data)} Round 2 matchups for {year}")
            round2_data.extend(year_round2_data)
        except Exception as e:
            print(f"Error loading data for {year}: {e}")
    
    print(f"Total Round 2 matchups: {len(round2_data)}")
    return round2_data

class Round2Dataset(NCAADataset):
    """Dataset specifically for Round 2 matchups."""
    
    def __init__(self, years=None, feature_cols=None, normalize=True):
        """
        Initialize the dataset with only Round 2 matchups.
        
        Args:
            years: List of years to include (None for all years)
            feature_cols: List of columns to use as features (None for all numeric)
            normalize: Whether to normalize features
        """
        # Initialize with parent class
        super().__init__(years=years, feature_cols=feature_cols, normalize=normalize)
        
        # Filter matchups to only include Round 2
        round2_matchups = []
        for matchup in self.matchups:
            if matchup.get('round') == 2:
                round2_matchups.append(matchup)
        
        self.matchups = round2_matchups
        print(f"Filtered to {len(self.matchups)} Round 2 matchups")
        
        # Re-normalize if needed
        if normalize:
            self._normalize_features()

def augment_round2_data(dataset, num_augmentations=2):
    """
    Create augmented versions of Round 2 matchups with small perturbations.
    
    Args:
        dataset: Round2Dataset to augment
        num_augmentations: Number of augmented versions to create per matchup
        
    Returns:
        dataset: Augmented dataset
    """
    original_matchups = dataset.matchups.copy()
    augmented_matchups = []
    
    for matchup in original_matchups:
        for _ in range(num_augmentations):
            # Create a copy of the matchup
            aug_matchup = matchup.copy()
            
            # Add small random noise to team statistics (±5%)
            for stat in ['team1_stats', 'team2_stats']:
                if stat in aug_matchup:
                    aug_matchup[stat] = aug_matchup[stat].copy()
                    for key in aug_matchup[stat]:
                        if isinstance(aug_matchup[stat][key], (int, float)):
                            noise = np.random.uniform(-0.05, 0.05) * aug_matchup[stat][key]
                            aug_matchup[stat][key] += noise
            
            augmented_matchups.append(aug_matchup)
    
    # Add augmented matchups to the dataset
    dataset.matchups.extend(augmented_matchups)
    print(f"Added {len(augmented_matchups)} augmented matchups (total: {len(dataset.matchups)})")
    
    # Re-normalize features
    dataset._normalize_features()
    
    return dataset

def select_top_features(dataset, n_features=20):
    """
    Select top n_features based on correlation with outcome.
    
    Args:
        dataset: Dataset to analyze
        n_features: Number of top features to select
        
    Returns:
        selected_features: List of selected feature names
    """
    # Extract features and labels
    features = []
    labels = []
    
    for i in range(len(dataset)):
        f, l = dataset[i]
        features.append(f.numpy())
        labels.append(l.item())
    
    features = np.array(features)
    labels = np.array(labels)
    
    # Calculate correlation with outcome
    correlations = []
    feature_names = dataset.get_feature_names()
    
    for i in range(features.shape[1]):
        corr = np.corrcoef(features[:, i], labels)[0, 1]
        correlations.append((feature_names[i], i, abs(corr)))
    
    # Sort by absolute correlation
    correlations.sort(key=lambda x: x[2], reverse=True)
    
    # Print top features
    print("\nTop features by correlation with outcome:")
    for name, idx, corr in correlations[:n_features]:
        print(f"{name}: {corr:.4f}")
    
    # Return names of top features
    return [name for name, _, _ in correlations[:n_features]]

class SimpleRound2Predictor(nn.Module):
    """Simplified neural network model for predicting Round 2 outcomes."""
    
    def __init__(self, input_size, hidden_size=32, dropout_rate=0.5):
        """
        Initialize the model with a simpler architecture.
        
        Args:
            input_size: Number of input features
            hidden_size: Size of hidden layer
            dropout_rate: Dropout rate for regularization
        """
        super(SimpleRound2Predictor, self).__init__()
        
        # Define a simpler network architecture
        self.model = nn.Sequential(
            # Single hidden layer
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Output layer
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """Forward pass through the network."""
        return self.model(x).squeeze()

def train_round2_model(train_loader, val_loader, input_size, epochs=100, lr=0.001, weight_decay=1e-3):
    """
    Train the Round 2 prediction model with early stopping.
    
    Args:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        input_size: Number of input features
        epochs: Maximum number of training epochs
        lr: Learning rate
        weight_decay: L2 regularization strength
    
    Returns:
        model: Trained model
        history: Training history (loss and metrics)
    """
    # Initialize model, loss function, and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleRound2Predictor(input_size, dropout_rate=0.5).to(device)
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
    
    # Early stopping parameters
    patience = 15
    best_val_loss = float('inf')
    no_improve_epochs = 0
    best_model_state = None
    
    # Training loop
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
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epochs = 0
            best_model_state = model.state_dict().copy()
            print(f"New best model (val_loss: {val_loss:.4f})")
        else:
            no_improve_epochs += 1
            print(f"No improvement for {no_improve_epochs} epochs")
        
        # Early stopping check
        if no_improve_epochs >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Loaded best model from early stopping")
    
    return model, history

def create_round2_data_loaders(years_train=None, years_val=None, batch_size=16, feature_cols=None, augment=True):
    """
    Create PyTorch DataLoaders for Round 2 matchups.
    
    Args:
        years_train: List of years to use for training
        years_val: List of years to use for validation
        batch_size: Batch size for DataLoader
        feature_cols: List of columns to use as features
        augment: Whether to augment training data
        
    Returns:
        train_loader, val_loader: PyTorch DataLoaders
    """
    # Create datasets
    train_dataset = Round2Dataset(years=years_train, feature_cols=feature_cols)
    val_dataset = Round2Dataset(years=years_val, feature_cols=feature_cols)
    
    # Augment training data if requested
    if augment:
        train_dataset = augment_round2_data(train_dataset, num_augmentations=2)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, val_loader

def evaluate_on_years(model, years, feature_cols=None):
    """
    Evaluate the model on specific years.
    
    Args:
        model: Trained model
        years: List of years to evaluate
        feature_cols: List of feature columns
        
    Returns:
        results: Dictionary with evaluation results
    """
    results = {}
    
    for year in years:
        print(f"Evaluating on {year}...")
        
        # Create dataset for this year
        dataset = Round2Dataset(years=[year], feature_cols=feature_cols)
        
        if len(dataset) == 0:
            print(f"No Round 2 data found for {year}")
            continue
        
        # Create data loader
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=len(dataset),
            shuffle=False
        )
        
        # Evaluate model
        metrics = evaluate_model(model, data_loader)
        
        print(f"{year} Round 2 Accuracy: {metrics['accuracy']:.2%}")
        results[year] = metrics
    
    return results

def train_with_cross_validation(years_train, feature_cols=None, k=5):
    """
    Train Round 2 model using k-fold cross-validation.
    
    Args:
        years_train: List of years to use for training
        feature_cols: List of columns to use as features
        k: Number of folds
        
    Returns:
        best_model: Best model from cross-validation
    """
    print(f"Training with {k}-fold cross-validation...")
    
    # Create dataset with all training years
    full_dataset = Round2Dataset(years=years_train, feature_cols=feature_cols)
    
    # Augment data
    full_dataset = augment_round2_data(full_dataset, num_augmentations=1)
    
    # Get input size
    input_size = len(full_dataset.get_feature_names())
    
    # Initialize k-fold cross-validation
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    # Track best model
    best_val_acc = 0
    best_model = None
    fold_results = []
    
    # Train on each fold
    for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(full_dataset)))):
        print(f"\nTraining fold {fold+1}/{k}")
        
        # Create train/val subsets
        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)
        
        # Create data loaders
        train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=16, shuffle=False)
        
        print(f"Training samples: {len(train_subset)}")
        print(f"Validation samples: {len(val_subset)}")
        
        # Train model
        model, history = train_round2_model(
            train_loader=train_loader,
            val_loader=val_loader,
            input_size=input_size,
            epochs=100,
            lr=0.001,
            weight_decay=1e-3
        )
        
        # Evaluate model
        model.eval()
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for features, targets in val_loader:
                device = next(model.parameters()).device
                features = features.to(device)
                outputs = model(features)
                val_preds.extend((outputs > 0.5).cpu().numpy())
                val_targets.extend(targets.numpy())
        
        # Calculate accuracy
        val_acc = accuracy_score(val_targets, val_preds)
        print(f"Fold {fold+1} validation accuracy: {val_acc:.4f}")
        
        fold_results.append(val_acc)
        
        # Update best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model
    
    # Print cross-validation results
    print("\nCross-validation results:")
    for fold, acc in enumerate(fold_results):
        print(f"Fold {fold+1}: {acc:.4f}")
    print(f"Average accuracy: {np.mean(fold_results):.4f}")
    
    return best_model

def main():
    """Train and evaluate a Round 2 specific model with anti-overfitting techniques."""
    print("Preparing Round 2 training data with anti-overfitting techniques...")
    
    # Use years 2013-2023 for training and 2024 for validation
    years_train = list(range(2013, 2024))  # 2013-2023
    years_val = [2024]  # 2024 for validation
    
    print(f"Training years: {years_train}")
    print(f"Validation year: {years_val}")
    
    # Create data loaders specifically for Round 2 matchups
    train_loader, val_loader = create_round2_data_loaders(
        years_train=years_train,
        years_val=years_val,
        batch_size=16,
        augment=True  # Apply data augmentation
    )
    
    # Get input size from dataset
    input_size = len(train_loader.dataset.get_feature_names())
    
    print(f"Training Round 2 model with {input_size} features...")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Select top features
    top_features = select_top_features(train_loader.dataset, n_features=30)
    
    # Create new data loaders with selected features
    print("\nRetraining with top features...")
    train_loader, val_loader = create_round2_data_loaders(
        years_train=years_train,
        years_val=years_val,
        batch_size=16,
        feature_cols=top_features,
        augment=True
    )
    
    # Get new input size
    input_size = len(train_loader.dataset.get_feature_names())
    print(f"Training with {input_size} selected features")
    
    # Option 1: Train with early stopping
    print("\nTraining model with early stopping...")
    model, history = train_round2_model(
        train_loader=train_loader,
        val_loader=val_loader,
        input_size=input_size,
        epochs=100,
        lr=0.001,
        weight_decay=1e-3  # Increased regularization
    )
    
    # Option 2: Train with cross-validation
    # Uncomment to use cross-validation instead
    # print("\nTraining model with cross-validation...")
    # model = train_with_cross_validation(
    #     years_train=years_train,
    #     feature_cols=top_features,
    #     k=5
    # )
    
    # Plot training history
    plot_training_history(history)
    
    # Save model
    save_model(model, round2_model_path)
    print(f"Model saved to {round2_model_path}")
    
    # Evaluate on recent years
    evaluation_years = [2022, 2023, 2024]
    print(f"Evaluating Round 2 model on years: {evaluation_years}")
    results = evaluate_on_years(model, evaluation_years, feature_cols=top_features)
    
    # Save evaluation results
    results_path = os.path.join(model_dir, 'round2_model_robust_evaluation.json')
    with open(results_path, 'w') as f:
        # Convert numpy values to Python types for JSON serialization
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NpEncoder, self).default(obj)
        
        json.dump(results, f, indent=4, cls=NpEncoder)
    
    print(f"Evaluation results saved to {results_path}")

if __name__ == "__main__":
    main()
