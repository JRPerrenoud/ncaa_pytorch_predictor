"""
NCAA Tournament Prediction Model Training

This script trains the NCAA tournament prediction model and evaluates its performance.
"""

import os
import sys
import pandas as pd
import torch

# Add parent directory to path to handle imports properly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.data_loader import NCAADataset, create_data_loaders
from scripts.model import train_model, evaluate_model, plot_training_history, predict_matchup

def main():
    """Train and evaluate the NCAA tournament prediction model."""
    print("Loading data...")
    
    # Use years 2013-2021, 2023-2024 for training and 2022 for validation
    years_train = [y for y in range(2013, 2025) if y != 2022]  # 2013-2021, 2023-2024
    years_val = [2022]  # 2022 for validation
    
    print(f"Training years: {years_train}")
    print(f"Validation year: {years_val}")
    
    train_loader, val_loader = create_data_loaders(
        years_train=years_train,
        years_val=years_val,
        batch_size=16
    )
    
    # Get input size from dataset
    input_size = len(train_loader.dataset.get_feature_names())
    
    print(f"Training model with {input_size} features...")
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
    
    # Evaluate model on validation set
    print("\nEvaluating model on validation set...")
    metrics = evaluate_model(model, val_loader)
    
    print(f"Validation Accuracy: {metrics['accuracy']:.4f}")
    print(f"Validation AUC: {metrics['auc']:.4f}")
    print(f"Sensitivity (True Positive Rate): {metrics['sensitivity']:.4f}")
    print(f"Specificity (True Negative Rate): {metrics['specificity']:.4f}")
    print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
    
    # Print some example predictions
    print("\nExample predictions from validation set:")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for i in range(min(5, len(val_loader.dataset))):
        info = val_loader.dataset.get_matchup_info(i)
        features, label = val_loader.dataset[i]
        
        # Convert features to tensor and make prediction
        features_tensor = features.unsqueeze(0).to(device)
        with torch.no_grad():
            prediction = model(features_tensor).item()
        
        predicted_winner = info['team1'] if prediction > 0.5 else info['team2']
        actual_winner = info['team1'] if label.item() == 1 else info['team2']
        
        print(f"{info['year']} Round {info['round']}: {info['team1']} vs {info['team2']}")
        print(f"  Prediction: {prediction:.4f} ({predicted_winner} wins)")
        print(f"  Actual: {actual_winner} wins")
        print(f"  Correct: {predicted_winner == actual_winner}")
        print()
    
    print("Model training and evaluation complete!")

if __name__ == "__main__":
    main()
