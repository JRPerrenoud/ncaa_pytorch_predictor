"""
NCAA Tournament Simulation

This script simulates the entire NCAA tournament using our trained model.
It loads the best model and makes predictions for each matchup.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import torch
from collections import defaultdict

# Add parent directory to path to handle imports properly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.data_loader import NCAADataset, TeamNameMatcher
from scripts.model import load_model, predict_matchup

# Set paths
data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
model_path = os.path.join(model_dir, 'best_model.pt')

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

def simulate_tournament(model, team_stats, tournament_data, feature_cols):
    """
    Simulate the entire tournament and compare with actual results.
    
    Args:
        model: Trained model
        team_stats: DataFrame with team statistics
        tournament_data: List of tournament matchups
        feature_cols: List of feature columns used by the model
    
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
                    'correct': is_correct
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
        print(f"Overall accuracy: {results['overall_accuracy']:.2%}")
    
    return results

def print_interesting_matchups(results, n=5):
    """Print the most interesting matchups (closest predictions and biggest upsets)."""
    predictions = results['predictions']
    
    # Sort by closeness to 0.5 probability (most uncertain predictions)
    uncertain_preds = sorted(predictions, key=lambda x: abs(x['win_probability'] - 0.5))
    
    # Sort by prediction confidence for incorrect predictions (biggest upsets/misses)
    upsets = [p for p in predictions if not p['correct']]
    confident_misses = sorted(upsets, 
                             key=lambda x: abs(x['win_probability'] - 0.5) if x['win_probability'] < 0.5 else abs(1 - x['win_probability']),
                             reverse=True)
    
    print("\nMost uncertain predictions:")
    for i, pred in enumerate(uncertain_preds[:n]):
        print(f"{i+1}. Round {pred['round']}: {pred['team1']} vs {pred['team2']}")
        print(f"   Prediction: {pred['win_probability']:.2%} chance for {pred['team1']}")
        print(f"   Actual winner: {pred['actual_winner']}")
        print(f"   Correct: {pred['correct']}")
    
    print("\nBiggest upsets/misses:")
    for i, pred in enumerate(confident_misses[:n]):
        print(f"{i+1}. Round {pred['round']}: {pred['team1']} vs {pred['team2']}")
        print(f"   Prediction: {pred['win_probability']:.2%} chance for {pred['team1']}")
        print(f"   Actual winner: {pred['actual_winner']}")
        print(f"   Confidence: {abs(pred['win_probability'] - 0.5) + 0.5:.2%}")

def main():
    """Simulate the NCAA tournament using our trained model."""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Simulate NCAA tournament')
    parser.add_argument('--year', type=int, default=2024, help='Tournament year to simulate')
    args = parser.parse_args()
    
    print(f"Loading model and data for {args.year} tournament...")
    
    # Load dataset to get feature columns
    dataset = NCAADataset(years=[args.year])
    feature_cols = dataset.get_feature_names()
    
    # Load model
    model = load_model(model_path, len(feature_cols))
    
    # Load tournament data and team statistics
    tournament_data = load_tournament_data(args.year)
    team_stats = load_team_stats(args.year)
    
    print(f"Simulating {args.year} tournament with {len(tournament_data)} matchups...")
    results = simulate_tournament(model, team_stats, tournament_data, feature_cols)
    
    # Print interesting matchups
    print_interesting_matchups(results)
    
    # Save results with year in filename
    results_path = os.path.join(model_dir, f'tournament_simulation_{args.year}.json')
    with open(results_path, 'w') as f:
        json.dump({k: v if not isinstance(v, defaultdict) else dict(v) 
                  for k, v in results.items()}, f, indent=4)
    
    print(f"Simulation results for {args.year} saved to {results_path}")

if __name__ == "__main__":
    main()
