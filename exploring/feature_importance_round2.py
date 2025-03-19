"""
Round 2 Feature Importance Analysis

This script analyzes which features are most important for the Round 2-specific NCAA tournament prediction model.
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
general_model_path = os.path.join(model_dir, 'best_model.pt')
round2_model_path = os.path.join(model_dir, 'round2_model.pt')

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

def calculate_feature_importance(model_path, model_name):
    """Calculate and visualize feature importance for a specific model."""
    print(f"Loading {model_name} and data...")
    
    # Load validation dataset (2024 data)
    val_dataset = NCAADataset(years=[2024])
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
    
    print(f"Calculating permutation importance for {model_name}...")
    # Calculate permutation importance
    importance_df = manual_permutation_importance(
        model, X_val, y_val, feature_names, n_repeats=5
    )
    
    # Print top features
    print(f"\nTop 10 most important features for {model_name}:")
    print(importance_df.head(10))
    
    # Print bottom features
    print(f"\nLeast important features for {model_name}:")
    print(importance_df.tail(5))
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    plt.barh(importance_df['Feature'].head(15), importance_df['Importance'].head(15))
    plt.xlabel('Permutation Importance (decrease in accuracy)')
    plt.title(f'Top 15 Most Important Features for {model_name}')
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, f'feature_importance_{model_name.lower().replace(" ", "_")}.png'))
    print(f"Feature importance plot saved to {os.path.join(model_dir, f'feature_importance_{model_name.lower().replace(' ', '_')}.png')}")
    
    # Save full results to CSV
    importance_df.to_csv(os.path.join(model_dir, f'feature_importance_{model_name.lower().replace(" ", "_")}.csv'), index=False)
    print(f"Full feature importance results saved to {os.path.join(model_dir, f'feature_importance_{model_name.lower().replace(' ', '_')}.csv')}")
    
    return importance_df

def compare_model_importances(general_importance, round2_importance):
    """Compare feature importance between general and Round 2 models."""
    print("\nComparing feature importance between models...")
    
    # Merge the two importance DataFrames
    comparison = pd.merge(
        general_importance, 
        round2_importance,
        on='Feature', 
        suffixes=('_General', '_Round2')
    )
    
    # Calculate difference in importance
    comparison['Importance_Diff'] = comparison['Importance_Round2'] - comparison['Importance_General']
    
    # Sort by absolute difference
    comparison = comparison.sort_values('Importance_Diff', key=abs, ascending=False)
    
    # Print features with biggest differences
    print("\nFeatures with biggest importance differences (Round 2 - General):")
    print(comparison.head(10))
    
    # Plot comparison of top 15 features
    plt.figure(figsize=(14, 10))
    
    # Get top 15 features by average importance
    comparison['Avg_Importance'] = (comparison['Importance_General'] + comparison['Importance_Round2']) / 2
    top_features = comparison.sort_values('Avg_Importance', ascending=False).head(15)['Feature'].tolist()
    
    # Filter comparison to only include top features
    plot_data = comparison[comparison['Feature'].isin(top_features)].sort_values('Avg_Importance')
    
    # Create bar positions
    y_pos = np.arange(len(plot_data))
    width = 0.35
    
    # Create horizontal bar chart
    plt.barh(y_pos - width/2, plot_data['Importance_General'], width, label='General Model')
    plt.barh(y_pos + width/2, plot_data['Importance_Round2'], width, label='Round 2 Model')
    
    # Add labels and legend
    plt.yticks(y_pos, plot_data['Feature'])
    plt.xlabel('Permutation Importance (decrease in accuracy)')
    plt.title('Feature Importance Comparison: General vs Round 2 Model')
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(model_dir, 'feature_importance_comparison.png'))
    print(f"Feature importance comparison plot saved to {os.path.join(model_dir, 'feature_importance_comparison.png')}")
    
    # Save comparison to CSV
    comparison.to_csv(os.path.join(model_dir, 'feature_importance_comparison.csv'), index=False)
    print(f"Feature importance comparison saved to {os.path.join(model_dir, 'feature_importance_comparison.csv')}")
    
    return comparison

def main():
    """Main function to run Round 2 feature importance analysis."""
    # Calculate feature importance for general model
    general_importance = calculate_feature_importance(general_model_path, "General Model")
    
    # Calculate feature importance for Round 2 model
    round2_importance = calculate_feature_importance(round2_model_path, "Round 2 Model")
    
    # Compare importance between models
    comparison = compare_model_importances(general_importance, round2_importance)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
