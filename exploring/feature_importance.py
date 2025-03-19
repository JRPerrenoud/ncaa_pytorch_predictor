"""
Feature Importance Analysis

This script analyzes which features are most important for the NCAA tournament prediction model.
It uses a manual permutation importance approach to measure how much model performance decreases 
when each feature is shuffled.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Add the parent directory to the path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Now we can import from scripts
from scripts.data_loader import NCAADataset
from scripts.model import load_model

# Set paths
data_dir = os.path.join(parent_dir, 'data')
model_dir = os.path.join(parent_dir, 'models')
model_path = os.path.join(model_dir, 'best_model.pt')

def manual_permutation_importance(model, X, y, feature_names, n_repeats=10):
    """
    Calculate permutation importance manually.
    
    Args:
        model: PyTorch model
        X: Features array
        y: Target array
        feature_names: List of feature names
        n_repeats: Number of times to repeat permutation
    
    Returns:
        importance_df: DataFrame with importance scores
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # Calculate baseline accuracy
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        baseline_preds = model(X_tensor).cpu().numpy()
    baseline_preds_binary = (baseline_preds > 0.5).astype(int)
    baseline_accuracy = accuracy_score(y, baseline_preds_binary)
    print(f"Baseline accuracy: {baseline_accuracy:.4f}")
    
    # Calculate importance for each feature
    importances = []
    std_devs = []
    
    for i, feature_name in enumerate(feature_names):
        print(f"Calculating importance for feature {i+1}/{len(feature_names)}: {feature_name}")
        feature_importances = []
        
        for _ in range(n_repeats):
            # Create a copy of the data
            X_permuted = X.copy()
            
            # Shuffle the feature
            np.random.shuffle(X_permuted[:, i])
            
            # Calculate accuracy with shuffled feature
            X_permuted_tensor = torch.tensor(X_permuted, dtype=torch.float32).to(device)
            with torch.no_grad():
                preds = model(X_permuted_tensor).cpu().numpy()
            preds_binary = (preds > 0.5).astype(int)
            permuted_accuracy = accuracy_score(y, preds_binary)
            
            # Calculate importance (decrease in accuracy)
            importance = baseline_accuracy - permuted_accuracy
            feature_importances.append(importance)
        
        # Calculate mean and std of importance
        mean_importance = np.mean(feature_importances)
        std_importance = np.std(feature_importances)
        
        importances.append(mean_importance)
        std_devs.append(std_importance)
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances,
        'StdDev': std_devs
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    return importance_df

def calculate_feature_importance():
    """Calculate and visualize feature importance."""
    print("Loading data and model...")
    
    # Load validation dataset (2022 data)
    val_dataset = NCAADataset(years=[2022])
    feature_names = val_dataset.get_feature_names()
    
    # Prepare validation data
    X_val = []
    y_val = []
    for i in range(len(val_dataset)):
        features, label = val_dataset[i]
        X_val.append(features.numpy())
        y_val.append(1 if label > 0.5 else 0)
    
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    
    # Load model
    model = load_model(model_path, len(feature_names))
    model.eval()
    
    print("Calculating permutation importance...")
    # Calculate permutation importance
    importance_df = manual_permutation_importance(
        model, X_val, y_val, feature_names, n_repeats=5
    )
    
    # Print top features
    print("\nTop 10 most important features:")
    print(importance_df.head(10))
    
    # Print bottom features
    print("\nLeast important features:")
    print(importance_df.tail(5))
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    plt.barh(importance_df['Feature'].head(15), importance_df['Importance'].head(15))
    plt.xlabel('Permutation Importance (decrease in accuracy)')
    plt.title('Top 15 Most Important Features')
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'feature_importance.png'))
    print(f"Feature importance plot saved to {os.path.join(model_dir, 'feature_importance.png')}")
    
    # Save full results to CSV
    importance_df.to_csv(os.path.join(model_dir, 'feature_importance.csv'), index=False)
    print(f"Full feature importance results saved to {os.path.join(model_dir, 'feature_importance.csv')}")
    
    return importance_df

def analyze_feature_correlations():
    """Analyze correlations between features and outcomes."""
    print("\nAnalyzing feature correlations with outcomes...")
    
    # Load all data
    dataset = NCAADataset(years=list(range(2019, 2025)))
    feature_names = dataset.get_feature_names()
    
    # Prepare data
    X = []
    y = []
    for i in range(len(dataset)):
        features, label = dataset[i]
        X.append(features.numpy())
        y.append(label.item())
    
    # Create DataFrame
    data_df = pd.DataFrame(X, columns=feature_names)
    data_df['outcome'] = y
    
    # Calculate correlations with outcome
    correlations = data_df.corr()['outcome'].sort_values(ascending=False)
    
    print("\nTop 10 features correlated with winning:")
    print(correlations.head(11).drop('outcome'))
    
    print("\nTop 10 features correlated with losing:")
    print(correlations.tail(10))
    
    # Save correlations to CSV
    correlations.to_csv(os.path.join(model_dir, 'feature_correlations.csv'))
    print(f"Feature correlations saved to {os.path.join(model_dir, 'feature_correlations.csv')}")
    
    return correlations

def main():
    """Main function to run feature importance analysis."""
    importance_df = calculate_feature_importance()
    correlations = analyze_feature_correlations()

if __name__ == "__main__":
    main()
