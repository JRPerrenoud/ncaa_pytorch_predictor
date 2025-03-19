"""
NCAA Tournament Simulation with Multiple Round-Specific Models

This script simulates the NCAA tournament using different models for different rounds:
- Round 1: General model
- Round 2: Round 2-specific model
- Rounds 3+: General model

The script will run simulations for multiple years (2017, 2019, 2022, 2023, 2024) to evaluate model performance.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
import argparse

# Add parent directory to path to handle imports properly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.data_loader import NCAADataset, TeamNameMatcher
from scripts.model import load_model, predict_matchup

# Set paths
data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
general_model_path = os.path.join(model_dir, 'best_model.pt')
round2_model_path = os.path.join(model_dir, 'round2_model.pt')

# Define the SimpleRound2Predictor class to match the architecture used in training
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

def load_round2_robust_model(model_path, input_size):
    """
    Load the robust Round 2 model which has a different architecture.
    
    Args:
        model_path: Path to the model file
        input_size: Number of input features
        
    Returns:
        model: Loaded model
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleRound2Predictor(input_size).to(device)
    
    # Load model state
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model

def load_tournament_data(year=2024):
    """Load tournament data for a specific year."""
    results_path = os.path.join(data_dir, f'results_{year}.json')
    with open(results_path, 'r') as f:
        return json.load(f)

def load_team_stats(year=2024):
    """Load team statistics for a specific year."""
    stats_path = os.path.join(data_dir, 'cleaned_data.csv')
    stats_df = pd.read_csv(stats_path)
    return stats_df[stats_df['YEAR'] == year]

def simulate_tournament_multi_models(models, team_stats, tournament_data, feature_cols):
    """
    Simulate the entire tournament using different models for different rounds.
    
    Args:
        models: Dictionary of models for each round
        team_stats: DataFrame with team statistics
        tournament_data: List of tournament matchups
        feature_cols: List of feature columns used by the models
    
    Returns:
        results: Dictionary with simulation results
    """
    # Initialize team name matcher
    name_matcher = TeamNameMatcher()
    
    # Get list of available teams
    available_teams = team_stats['TEAM'].unique().tolist()
    
    # Group matchups by round
    rounds = defaultdict(list)
    for matchup in tournament_data:
        rounds[matchup['round']].append(matchup)
    
    # Initialize results
    results = {
        'correct_predictions': 0,
        'total_predictions': 0,
        'accuracy_by_round': {},
        'predictions': []
    }
    
    # Process each round
    max_round = max(rounds.keys())
    for round_num in range(1, max_round + 1):
        round_matchups = rounds[round_num]
        correct = 0
        total = 0
        
        # Select the appropriate model for this round
        if round_num in models:
            model = models[round_num]
        else:
            # Default to the general model if no specific model exists
            model = models['general']
        
        model_name = "Round 2 Model" if round_num == 2 else "General Model"
        print(f"Using {model_name} for Round {round_num}")
        
        for matchup in round_matchups:
            team1_name = matchup['team1']
            team2_name = matchup['team2']
            actual_winner = matchup['winner']
            
            # Match team names to available teams
            team1_match = name_matcher.find_best_match(team1_name, available_teams)
            team2_match = name_matcher.find_best_match(team2_name, available_teams)
            
            # Skip if team stats not found
            if team1_match is None or team2_match is None:
                print(f"Warning: Stats not found for {team1_name} or {team2_name}")
                continue
            
            # Get team statistics
            team1_stats = team_stats[team_stats['TEAM'] == team1_match]
            team2_stats = team_stats[team_stats['TEAM'] == team2_match]
            
            # Skip if team stats not found
            if len(team1_stats) == 0 or len(team2_stats) == 0:
                print(f"Warning: Stats not found for {team1_match} or {team2_match}")
                continue
            
            try:
                # Extract features
                team1_features = team1_stats.iloc[0][feature_cols].values.astype(np.float32)
                team2_features = team2_stats.iloc[0][feature_cols].values.astype(np.float32)
                
                # Create feature vector (difference between team1 and team2)
                features = team1_features - team2_features
                
                # Convert to PyTorch tensor
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
                
                # Make prediction
                with torch.no_grad():
                    win_prob = model(features_tensor).item()
                
                predicted_winner = team1_name if win_prob > 0.5 else team2_name
                
                # Check if prediction is correct
                is_correct = (predicted_winner == actual_winner or 
                             name_matcher.similarity_score(predicted_winner, actual_winner) > 0.8)
                if is_correct:
                    correct += 1
                total += 1
                
                # Store prediction details
                prediction = {
                    'round': round_num,
                    'team1': team1_name,
                    'team2': team2_name,
                    'team1_match': team1_match,
                    'team2_match': team2_match,
                    'win_probability': float(win_prob),
                    'predicted_winner': predicted_winner,
                    'actual_winner': actual_winner,
                    'correct': is_correct,
                    'model_used': model_name
                }
                results['predictions'].append(prediction)
            except Exception as e:
                print(f"Error processing matchup {team1_name} vs {team2_name}: {str(e)}")
        
        # Calculate accuracy for this round
        if total > 0:
            round_accuracy = correct / total
            results['accuracy_by_round'][round_num] = round_accuracy
            results['correct_predictions'] += correct
            results['total_predictions'] += total
            print(f"Round {round_num}: {correct}/{total} correct ({round_accuracy:.2%})")
    
    # Calculate overall accuracy
    if results['total_predictions'] > 0:
        results['overall_accuracy'] = results['correct_predictions'] / results['total_predictions']
    
    return results

def print_interesting_matchups(results, n=5):
    """Print the most interesting matchups (closest predictions and biggest upsets)."""
    predictions = results['predictions']
    
    # Sort by closeness to 50% probability
    closest_predictions = sorted(predictions, key=lambda x: abs(x['win_probability'] - 0.5))
    
    print("\nClosest Predictions (Most Uncertain):")
    for i, pred in enumerate(closest_predictions[:n]):
        print(f"{i+1}. {pred['team1']} vs {pred['team2']} (Round {pred['round']})")
        print(f"   Predicted: {pred['predicted_winner']} ({pred['win_probability']:.2%})")
        print(f"   Actual: {pred['actual_winner']}")
        print(f"   Correct: {pred['correct']}")
        print(f"   Model Used: {pred['model_used']}")
        print(f"   Confidence: {abs(pred['win_probability'] - 0.5) + 0.5:.2%}")

def main():
    """Simulate the NCAA tournament using multiple models for different rounds."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Simulate NCAA tournament with multiple models')
    parser.add_argument('--years', nargs='+', type=int, default=[2023], 
                        help='Tournament years to simulate (default: 2023)')
    args = parser.parse_args()
    
    # Load dataset to get feature columns (using the latest year for feature columns)
    latest_year = max(args.years)
    dataset = NCAADataset(years=[latest_year])
    feature_cols = dataset.get_feature_names()
    
    # Load models
    print("Loading models...")
    general_model = load_model(general_model_path, len(feature_cols))
    round2_model = load_model(round2_model_path, len(feature_cols))
    
    # Create model dictionary
    models = {
        'general': general_model,  # Default model
        1: general_model,          # Round 1: General model
        2: round2_model,           # Round 2: Round 2-specific model
        # Rounds 3+ will use the general model (default)
    }
    
    # Store combined results across all years
    combined_results = {
        'by_year': {},
        'overall': {
            'correct_predictions': 0,
            'total_predictions': 0,
            'accuracy_by_round': defaultdict(lambda: {'correct': 0, 'total': 0})
        }
    }
    
    # Run simulations for each year
    for year in args.years:
        print(f"\n{'='*50}")
        print(f"SIMULATING {year} TOURNAMENT")
        print(f"{'='*50}")
        
        try:
            # Load tournament data and team statistics
            tournament_data = load_tournament_data(year)
            team_stats = load_team_stats(year)
            
            print(f"Simulating {year} tournament with {len(tournament_data)} matchups using multiple models...")
            results = simulate_tournament_multi_models(models, team_stats, tournament_data, feature_cols)
            
            # Print overall results
            print("\nOverall Results:")
            print(f"Total Accuracy: {results['correct_predictions']}/{results['total_predictions']} ({results['overall_accuracy']:.2%})")
            
            # Print accuracy by round
            print("\nAccuracy by Round:")
            for round_num in sorted(results['accuracy_by_round'].keys()):
                accuracy = results['accuracy_by_round'][round_num]
                model_used = "Round 2 Model" if round_num == 2 else "General Model"
                print(f"Round {round_num} ({model_used}): {accuracy:.2%}")
            
            # Print interesting matchups
            print_interesting_matchups(results)
            
            # Save results with year in filename
            results_path = os.path.join(model_dir, f'tournament_simulation_multi_models_{year}.json')
            with open(results_path, 'w') as f:
                json.dump({k: v if not isinstance(v, defaultdict) else dict(v) 
                          for k, v in results.items()}, f, indent=4)
            
            print(f"Simulation results for {year} saved to {results_path}")
            
            # Add to combined results
            combined_results['by_year'][year] = {
                'correct_predictions': results['correct_predictions'],
                'total_predictions': results['total_predictions'],
                'overall_accuracy': results['overall_accuracy'],
                'accuracy_by_round': results['accuracy_by_round']
            }
            
            # Update overall statistics
            combined_results['overall']['correct_predictions'] += results['correct_predictions']
            combined_results['overall']['total_predictions'] += results['total_predictions']
            
            # Update round-specific statistics
            for round_num, accuracy in results['accuracy_by_round'].items():
                correct = int(accuracy * results['total_predictions'] * len(results['accuracy_by_round'].keys()) ** -1)
                total = int(results['total_predictions'] * len(results['accuracy_by_round'].keys()) ** -1)
                combined_results['overall']['accuracy_by_round'][round_num]['correct'] += correct
                combined_results['overall']['accuracy_by_round'][round_num]['total'] += total
                
        except Exception as e:
            print(f"Error simulating {year} tournament: {str(e)}")
    
    # Calculate overall accuracy across all years
    if combined_results['overall']['total_predictions'] > 0:
        combined_results['overall']['overall_accuracy'] = (
            combined_results['overall']['correct_predictions'] / 
            combined_results['overall']['total_predictions']
        )
    
    # Calculate round-specific accuracies across all years
    for round_num, stats in combined_results['overall']['accuracy_by_round'].items():
        if stats['total'] > 0:
            stats['accuracy'] = stats['correct'] / stats['total']
    
    # Convert defaultdict to regular dict for JSON serialization
    combined_results['overall']['accuracy_by_round'] = dict(combined_results['overall']['accuracy_by_round'])
    
    # Print combined results
    print(f"\n{'='*50}")
    print(f"COMBINED RESULTS ACROSS ALL YEARS")
    print(f"{'='*50}")
    print(f"Total Accuracy: {combined_results['overall']['correct_predictions']}/{combined_results['overall']['total_predictions']} ({combined_results['overall']['overall_accuracy']:.2%})")
    
    print("\nAccuracy by Round:")
    for round_num, stats in sorted(combined_results['overall']['accuracy_by_round'].items()):
        model_used = "Round 2 Model" if round_num == 2 else "General Model"
        print(f"Round {round_num} ({model_used}): {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
    
    # Save combined results
    combined_results_path = os.path.join(model_dir, 'tournament_simulation_multi_models_combined.json')
    with open(combined_results_path, 'w') as f:
        json.dump(combined_results, f, indent=4)
    
    print(f"Combined simulation results saved to {combined_results_path}")

if __name__ == "__main__":
    main()
