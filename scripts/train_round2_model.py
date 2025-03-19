"""
Round 2 Specific NCAA Tournament Prediction Model Training

This script trains a model specifically for predicting Round 2 matchups in the NCAA tournament.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import torch

# Add parent directory to path to handle imports properly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.data_loader import NCAADataset, create_data_loaders
from scripts.model import train_model, evaluate_model, plot_training_history, save_model, load_model

# Set paths
data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
round2_model_path = os.path.join(model_dir, 'round2_model.pt')

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

def create_round2_data_loaders(years_train=None, years_val=None, batch_size=16, feature_cols=None):
    """
    Create PyTorch DataLoaders for Round 2 matchups.
    
    Args:
        years_train: List of years to use for training
        years_val: List of years to use for validation
        batch_size: Batch size for DataLoader
        feature_cols: List of columns to use as features
        
    Returns:
        train_loader, val_loader: PyTorch DataLoaders
    """
    # Create datasets
    train_dataset = Round2Dataset(years=years_train, feature_cols=feature_cols)
    val_dataset = Round2Dataset(years=years_val, feature_cols=feature_cols)
    
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

def main():
    """Train and evaluate a Round 2 specific model."""
    print("Preparing Round 2 training data...")
    
    # Use years 2013-2023 for training and 2024 for validation
    years_train = list(range(2013, 2024))  # 2013-2023
    years_val = [2024]  # 2024 for validation
    
    print(f"Training years: {years_train}")
    print(f"Validation year: {years_val}")
    
    # Create data loaders specifically for Round 2 matchups
    train_loader, val_loader = create_round2_data_loaders(
        years_train=years_train,
        years_val=years_val,
        batch_size=16
    )
    
    # Get input size from dataset
    input_size = len(train_loader.dataset.get_feature_names())
    
    print(f"Training Round 2 model with {input_size} features...")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
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
    
    # Save model
    save_model(model, round2_model_path)
    print(f"Model saved to {round2_model_path}")
    
    # Evaluate on recent years
    evaluation_years = [2022, 2023, 2024]
    print(f"Evaluating Round 2 model on years: {evaluation_years}")
    results = evaluate_on_years(model, evaluation_years)
    
    # Save evaluation results
    results_path = os.path.join(model_dir, 'round2_model_evaluation.json')
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
