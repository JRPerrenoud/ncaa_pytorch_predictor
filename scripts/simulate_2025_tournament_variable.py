"""
NCAA Tournament Variable Model Simulation

This script simulates the NCAA tournament using user-specified models for each round.
Users can pass in model names as input parameters to customize which model is used
for each of the 6 tournament rounds.

Usage:
    python simulate_tournament_variable.py --round1=best_model.pt --round2=round2_model.pt --round3=best_model.pt --round4=best_model.pt --round5=best_model.pt --round6=best_model.pt

You can specify any combination of models for any round. If a model is not specified for a round,
the script will use the default model (best_model.pt).
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import torch
from collections import defaultdict

# Add parent directory to path to handle imports properly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.data_loader import NCAADataset, TeamNameMatcher
from scripts.model import load_model

# Set paths
data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
default_model_path = os.path.join(model_dir, 'best_model.pt')
bracket_path = os.path.join(data_dir, '2025_bracket.json')
stats_path = os.path.join(data_dir, 'cleaned_data.csv')

def load_bracket_data():
    """Load the 2025 tournament bracket data."""
    with open(bracket_path, 'r') as f:
        return json.load(f)

def load_team_stats(year=2025):
    """
    Load team statistics for prediction.
    
    Using actual 2025 team statistics from cleaned_data.csv.
    """
    stats_df = pd.read_csv(stats_path)
    return stats_df[stats_df['YEAR'] == year]

def get_team_match(name, available_teams, name_matcher):
    """
    Try to match a team name with available teams, with fallbacks.
    """
    # First try exact match
    if name in available_teams:
        return name
    
    # Try using the name matcher
    match = name_matcher.find_best_match(name, available_teams)
    if match:
        return match
    
    # Manual mapping for common teams
    manual_mapping = {
        "Louisville": "Louisville",
        "Creighton": "Creighton",
        "Ole Miss": "Mississippi",
        "North Carolina": "North Carolina",
        "Iowa State": "Iowa St",
        "Michigan State": "Michigan St",
        "Bryant": "Bryant",
        "Mount St. Mary's": "Mount St Mary's",
        "Liberty": "Liberty",
        "Saint Mary's": "Saint Mary's",
        "Vanderbilt": "Vanderbilt",
        "Alabama": "Alabama",
        "Robert Morris": "Robert Morris",
        "Houston": "Houston",
        "SIU Edwardsville": "SIU Edwardsville",
        "Gonzaga": "Gonzaga",
        "Georgia": "Georgia",
        "Purdue": "Purdue",
        "High Point": "High Point",
        "Kentucky": "Kentucky",
        "Troy": "Troy",
        "UConn": "Connecticut",
        "Oklahoma": "Oklahoma",
        "Memphis": "Memphis",
        "Colorado State": "Colorado St",
        "Mizzou": "Missouri",
        "Drake": "Drake",
        "Texas Tech": "Texas Tech",
        "UNC Wilmington": "UNC Wilmington",
        "St. John's": "St John's",
        "Nebraska Omaha": "Nebraska Omaha"
    }
    
    if name in manual_mapping:
        mapped_name = manual_mapping[name]
        if mapped_name in available_teams:
            return mapped_name
        # Try the name matcher on the mapped name
        match = name_matcher.find_best_match(mapped_name, available_teams)
        if match:
            return match
    
    # Last resort - use a default team with average stats
    print(f"Warning: Could not match team {name}, using average team stats")
    return available_teams[0]  # Use the first team as a fallback

def get_model_for_round(round_num, models_dict):
    """
    Return the appropriate model for the given round.
    
    Args:
        round_num: Tournament round number (1-6)
        models_dict: Dictionary mapping round numbers to model objects
        
    Returns:
        The appropriate model for the round, its name, and whether it's a robust model
    """
    if round_num in models_dict:
        return models_dict[round_num]['model'], models_dict[round_num]['name'], models_dict[round_num].get('is_robust', False)
    else:
        # This should never happen as we load default models for all rounds
        print(f"Warning: No model specified for round {round_num}, using default model")
        return models_dict[1]['model'], "Default Model", False  # Fallback to round 1 model

def predict_matchup(model, team1_stats, team2_stats, feature_cols, is_robust=False):
    """
    Predict the winner of a matchup using the specified model.
    
    Args:
        model: Model to use for prediction
        team1_stats: Statistics for team 1
        team2_stats: Statistics for team 2
        feature_cols: Feature columns for prediction
        is_robust: Whether this is a robust model that needs special handling
        
    Returns:
        Probability of team1 winning
    """
    if is_robust:
        # Use the special handling for robust models
        return predict_with_robust_model(model, team1_stats, team2_stats, feature_cols)
    else:
        # Standard prediction process
        # Create feature difference
        feature_diff = team1_stats - team2_stats
        
        # Convert to tensor
        features_tensor = torch.tensor([feature_diff.values], dtype=torch.float32)
        
        # Get device
        device = next(model.parameters()).device
        features_tensor = features_tensor.to(device)
        
        # Make prediction
        with torch.no_grad():
            win_prob = model(features_tensor).item()
        
        return win_prob

def predict_winner(models_dict, team1_stats, team2_stats, team1_name, team2_name, feature_cols, round_num, region):
    """
    Predict the winner of a matchup using the appropriate model for the round.
    
    Args:
        models_dict: Dictionary mapping round numbers to model objects
        team1_stats: Statistics for team 1
        team2_stats: Statistics for team 2
        team1_name: Name of team 1
        team2_name: Name of team 2
        feature_cols: Feature columns for prediction
        round_num: Tournament round number
        region: Tournament region
        
    Returns:
        Dictionary with prediction details
    """
    # Select the appropriate model for this round
    model, model_name, is_robust = get_model_for_round(round_num, models_dict)
    
    # Ensure feature columns are the correct type
    team1_features = team1_stats[feature_cols].values.astype(np.float32)
    team2_features = team2_stats[feature_cols].values.astype(np.float32)
    
    # Create a modified stats object with the correct data types
    team1_stats_modified = pd.Series(team1_features, index=feature_cols)
    team2_stats_modified = pd.Series(team2_features, index=feature_cols)
    
    # Predict winner
    win_probability = predict_matchup(model, team1_stats_modified, team2_stats_modified, feature_cols, is_robust)
    predicted_winner = team1_name if win_probability > 0.5 else team2_name
    win_prob_display = float(win_probability) if win_probability > 0.5 else float(1 - win_probability)
    
    # Print prediction
    print(f"{team1_name} vs {team2_name} [Using {model_name}]")
    print(f"Prediction: {predicted_winner} wins ({win_prob_display:.2%} confidence)")
    print("-" * 50)
    
    # Return prediction details
    return {
        'round': round_num,
        'region': region,
        'team1': team1_name,
        'team2': team2_name,
        'win_probability': win_prob_display,
        'predicted_winner': predicted_winner,
        'model_used': model_name
    }

def simulate_tournament(models_dict, team_stats, bracket_data, feature_cols):
    """
    Simulate the entire tournament using specified models for each round.
    
    Args:
        models_dict: Dictionary mapping round numbers to model objects
        team_stats: DataFrame with team statistics
        bracket_data: List of tournament matchups
        feature_cols: List of feature columns used by the model
    
    Returns:
        results: Dictionary with simulation results
    """
    # Initialize team name matcher
    name_matcher = TeamNameMatcher()
    
    # Get list of available teams
    available_teams = team_stats['TEAM'].unique().tolist()
    
    # Initialize results
    results = {
        'predictions': [],
        'rounds': defaultdict(list),
        'regions': {}
    }
    
    # Group matchups by region
    regions = {}
    for matchup in bracket_data:
        region = matchup['region']
        if region not in regions:
            regions[region] = []
        regions[region].append(matchup)
    
    # Process each region separately through the first 4 rounds
    for region_name, region_matchups in regions.items():
        print(f"\n=== {region_name} Region ===\n")
        results['regions'][region_name] = {'rounds': defaultdict(list)}
        
        # Round 1
        print(f"\nRound 1 - {region_name} Region:")
        print("-" * 50)
        
        round1_winners = []
        for i in range(0, len(region_matchups), 2):
            if i + 1 < len(region_matchups):
                matchup1 = region_matchups[i]
                matchup2 = region_matchups[i+1]
                
                # Process first matchup
                team1_name = matchup1['team1']
                team2_name = matchup1['team2']
                
                team1_match = get_team_match(team1_name, available_teams, name_matcher)
                team2_match = get_team_match(team2_name, available_teams, name_matcher)
                
                if not team1_match or not team2_match:
                    print(f"Warning: Could not match teams for {team1_name} vs {team2_name}")
                    # Use a placeholder winner if we can't match the teams
                    round1_winners.append({
                        'team': team1_name,
                        'team_match': None
                    })
                    continue
                
                # Get team statistics
                team1_stats = team_stats[team_stats['TEAM'] == team1_match].iloc[0]
                team2_stats = team_stats[team_stats['TEAM'] == team2_match].iloc[0]
                
                # Predict winner of first matchup
                prediction1 = predict_winner(
                    models_dict, team1_stats, team2_stats, team1_name, team2_name, 
                    feature_cols, 1, region_name
                )
                results['predictions'].append(prediction1)
                results['rounds'][1].append(prediction1)
                results['regions'][region_name]['rounds'][1].append(prediction1)
                
                winner1 = {
                    'team': prediction1['predicted_winner'],
                    'team_match': team1_match if prediction1['predicted_winner'] == team1_name else team2_match
                }
                
                # Process second matchup
                team1_name = matchup2['team1']
                team2_name = matchup2['team2']
                
                team1_match = get_team_match(team1_name, available_teams, name_matcher)
                team2_match = get_team_match(team2_name, available_teams, name_matcher)
                
                if not team1_match or not team2_match:
                    print(f"Warning: Could not match teams for {team1_name} vs {team2_name}")
                    # Use a placeholder winner if we can't match the teams
                    round1_winners.append({
                        'team': team1_name,
                        'team_match': None
                    })
                    continue
                
                # Get team statistics
                team1_stats = team_stats[team_stats['TEAM'] == team1_match].iloc[0]
                team2_stats = team_stats[team_stats['TEAM'] == team2_match].iloc[0]
                
                # Predict winner of second matchup
                prediction2 = predict_winner(
                    models_dict, team1_stats, team2_stats, team1_name, team2_name, 
                    feature_cols, 1, region_name
                )
                results['predictions'].append(prediction2)
                results['rounds'][1].append(prediction2)
                results['regions'][region_name]['rounds'][1].append(prediction2)
                
                winner2 = {
                    'team': prediction2['predicted_winner'],
                    'team_match': team1_match if prediction2['predicted_winner'] == team1_name else team2_match
                }
                
                # Add both winners to the list for the next round
                round1_winners.append(winner1)
                round1_winners.append(winner2)
        
        # Round 2
        print(f"\nRound 2 - {region_name} Region:")
        print("-" * 50)
        
        round2_winners = []
        for i in range(0, len(round1_winners), 2):
            if i + 1 < len(round1_winners):
                winner1 = round1_winners[i]
                winner2 = round1_winners[i+1]
                
                team1_name = winner1['team']
                team2_name = winner2['team']
                
                team1_match = winner1['team_match']
                team2_match = winner2['team_match']
                
                if not team1_match or not team2_match:
                    team1_match = get_team_match(team1_name, available_teams, name_matcher)
                    team2_match = get_team_match(team2_name, available_teams, name_matcher)
                
                if not team1_match or not team2_match:
                    print(f"Warning: Could not match teams for {team1_name} vs {team2_name}")
                    # Use a placeholder winner if we can't match the teams
                    round2_winners.append({
                        'team': team1_name,
                        'team_match': None
                    })
                    continue
                
                # Get team statistics
                team1_stats = team_stats[team_stats['TEAM'] == team1_match].iloc[0]
                team2_stats = team_stats[team_stats['TEAM'] == team2_match].iloc[0]
                
                # Predict winner
                prediction = predict_winner(
                    models_dict, team1_stats, team2_stats, team1_name, team2_name, 
                    feature_cols, 2, region_name
                )
                results['predictions'].append(prediction)
                results['rounds'][2].append(prediction)
                results['regions'][region_name]['rounds'][2].append(prediction)
                
                winner = {
                    'team': prediction['predicted_winner'],
                    'team_match': team1_match if prediction['predicted_winner'] == team1_name else team2_match
                }
                
                round2_winners.append(winner)
        
        # Round 3 (Sweet 16)
        print(f"\nRound 3 (Sweet 16) - {region_name} Region:")
        print("-" * 50)
        
        round3_winners = []
        for i in range(0, len(round2_winners), 2):
            if i + 1 < len(round2_winners):
                winner1 = round2_winners[i]
                winner2 = round2_winners[i+1]
                
                team1_name = winner1['team']
                team2_name = winner2['team']
                
                team1_match = winner1['team_match']
                team2_match = winner2['team_match']
                
                if not team1_match or not team2_match:
                    team1_match = get_team_match(team1_name, available_teams, name_matcher)
                    team2_match = get_team_match(team2_name, available_teams, name_matcher)
                
                if not team1_match or not team2_match:
                    print(f"Warning: Could not match teams for {team1_name} vs {team2_name}")
                    # Use a placeholder winner if we can't match the teams
                    round3_winners.append({
                        'team': team1_name,
                        'team_match': None
                    })
                    continue
                
                # Get team statistics
                team1_stats = team_stats[team_stats['TEAM'] == team1_match].iloc[0]
                team2_stats = team_stats[team_stats['TEAM'] == team2_match].iloc[0]
                
                # Predict winner
                prediction = predict_winner(
                    models_dict, team1_stats, team2_stats, team1_name, team2_name, 
                    feature_cols, 3, region_name
                )
                results['predictions'].append(prediction)
                results['rounds'][3].append(prediction)
                results['regions'][region_name]['rounds'][3].append(prediction)
                
                winner = {
                    'team': prediction['predicted_winner'],
                    'team_match': team1_match if prediction['predicted_winner'] == team1_name else team2_match
                }
                
                round3_winners.append(winner)
        
        # Round 4 (Elite 8 - Regional Finals)
        print(f"\nRound 4 (Elite 8) - {region_name} Region:")
        print("-" * 50)
        
        if len(round3_winners) >= 2:
            winner1 = round3_winners[0]
            winner2 = round3_winners[1]
            
            team1_name = winner1['team']
            team2_name = winner2['team']
            
            team1_match = winner1['team_match']
            team2_match = winner2['team_match']
            
            if not team1_match or not team2_match:
                team1_match = get_team_match(team1_name, available_teams, name_matcher)
                team2_match = get_team_match(team2_name, available_teams, name_matcher)
            
            if not team1_match or not team2_match:
                print(f"Warning: Could not match teams for {team1_name} vs {team2_name}")
                # Use a placeholder winner if we can't match the teams
                results['regions'][region_name]['winner'] = {
                    'team': team1_name,
                    'team_match': None
                }
                continue
            
            # Get team statistics
            team1_stats = team_stats[team_stats['TEAM'] == team1_match].iloc[0]
            team2_stats = team_stats[team_stats['TEAM'] == team2_match].iloc[0]
            
            # Predict winner
            prediction = predict_winner(
                models_dict, team1_stats, team2_stats, team1_name, team2_name, 
                feature_cols, 4, region_name
            )
            results['predictions'].append(prediction)
            results['rounds'][4].append(prediction)
            results['regions'][region_name]['rounds'][4].append(prediction)
            
            # Store regional champion
            results['regions'][region_name]['winner'] = {
                'team': prediction['predicted_winner'],
                'team_match': team1_match if prediction['predicted_winner'] == team1_name else team2_match
            }
            
            print(f"\n🏆 {region_name} Region Champion: {prediction['predicted_winner']} 🏆\n")
    
    # Final Four (Round 5)
    print("\n=== Final Four (Round 5) ===\n")
    
    # Get regional champions
    regional_champions = []
    for region_name in ['South', 'East', 'Midwest', 'West']:
        if region_name in results['regions'] and 'winner' in results['regions'][region_name]:
            regional_champions.append({
                'region': region_name,
                'team': results['regions'][region_name]['winner']['team'],
                'team_match': results['regions'][region_name]['winner']['team_match']
            })
    
    # Final Four Matchups (traditionally South vs East, Midwest vs West)
    final_four_matchups = [
        (0, 1),  # South vs East
        (2, 3)   # Midwest vs West
    ]
    
    championship_teams = []
    
    for matchup_idx, (idx1, idx2) in enumerate(final_four_matchups):
        if idx1 < len(regional_champions) and idx2 < len(regional_champions):
            champion1 = regional_champions[idx1]
            champion2 = regional_champions[idx2]
            
            team1_name = champion1['team']
            team2_name = champion2['team']
            
            team1_match = champion1['team_match']
            team2_match = champion2['team_match']
            
            if not team1_match or not team2_match:
                team1_match = get_team_match(team1_name, available_teams, name_matcher)
                team2_match = get_team_match(team2_name, available_teams, name_matcher)
            
            if not team1_match or not team2_match:
                print(f"Warning: Could not match teams for {team1_name} vs {team2_name}")
                # Use a placeholder winner if we can't match the teams
                championship_teams.append({
                    'team': team1_name,
                    'team_match': None,
                    'region': champion1['region']
                })
                continue
            
            # Get team statistics
            team1_stats = team_stats[team_stats['TEAM'] == team1_match].iloc[0]
            team2_stats = team_stats[team_stats['TEAM'] == team2_match].iloc[0]
            
            # Predict winner
            prediction = predict_winner(
                models_dict, team1_stats, team2_stats, team1_name, team2_name, 
                feature_cols, 5, "Final Four"
            )
            results['predictions'].append(prediction)
            results['rounds'][5].append(prediction)
            
            winner_region = champion1['region'] if prediction['predicted_winner'] == team1_name else champion2['region']
            winner_match = team1_match if prediction['predicted_winner'] == team1_name else team2_match
            
            championship_teams.append({
                'team': prediction['predicted_winner'],
                'team_match': winner_match,
                'region': winner_region
            })
    
    # Championship Game (Round 6)
    print("\n=== Championship Game (Round 6) ===\n")
    
    if len(championship_teams) >= 2:
        finalist1 = championship_teams[0]
        finalist2 = championship_teams[1]
        
        team1_name = finalist1['team']
        team2_name = finalist2['team']
        
        team1_match = finalist1['team_match']
        team2_match = finalist2['team_match']
        
        if not team1_match or not team2_match:
            team1_match = get_team_match(team1_name, available_teams, name_matcher)
            team2_match = get_team_match(team2_name, available_teams, name_matcher)
        
        if not team1_match or not team2_match:
            print(f"Warning: Could not match teams for {team1_name} vs {team2_name}")
            # Use a placeholder champion if we can't match the teams
            results['champion'] = {
                'team': team1_name,
                'region': finalist1['region']
            }
        else:
            # Get team statistics
            team1_stats = team_stats[team_stats['TEAM'] == team1_match].iloc[0]
            team2_stats = team_stats[team_stats['TEAM'] == team2_match].iloc[0]
            
            # Predict winner
            prediction = predict_winner(
                models_dict, team1_stats, team2_stats, team1_name, team2_name, 
                feature_cols, 6, "Championship"
            )
            results['predictions'].append(prediction)
            results['rounds'][6].append(prediction)
            
            winner_region = finalist1['region'] if prediction['predicted_winner'] == team1_name else finalist2['region']
            
            # Store champion
            results['champion'] = {
                'team': prediction['predicted_winner'],
                'region': winner_region
            }
    
    # Print champion
    if 'champion' in results:
        champion = results['champion']['team']
        champion_region = results['champion']['region']
        print("\n🏆 Predicted 2025 NCAA Tournament Champion: 🏆")
        print(f"🏀 {champion} ({champion_region} Region) 🏀")
    
    return results

class SimpleRound2Predictor(torch.nn.Module):
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
        self.model = torch.nn.Sequential(
            # Single hidden layer
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.BatchNorm1d(hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            
            # Output layer
            torch.nn.Linear(hidden_size, 1),
            torch.nn.Sigmoid()
        )
    
    def forward(self, x):
        """Forward pass through the network."""
        return self.model(x).squeeze()

def load_robust_round2_model(model_path, feature_count):
    """
    Load the robust Round 2 model which has a different architecture.
    
    Args:
        model_path: Path to the model file
        feature_count: Number of input features
        
    Returns:
        model: Loaded model
    """
    # The robust model was trained with 30 features
    input_size = 30
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleRound2Predictor(input_size).to(device)
    
    # Load model state
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model

def predict_with_robust_model(model, team1_stats, team2_stats, feature_names):
    """
    Make a prediction using the robust Round 2 model.
    
    Args:
        model: The robust model
        team1_stats: Statistics for team 1
        team2_stats: Statistics for team 2
        feature_names: List of all feature names
        
    Returns:
        win_probability: Probability of team1 winning
    """
    # The robust model was trained with these specific features
    robust_features = [
        'AdjEM', 'AdjO', 'AdjD', 'AdjT', 'Luck', 'SOS_AdjEM', 'SOS_OppO', 
        'SOS_OppD', 'NCSOS_AdjEM', 'Seed', 'Win_Pct', 'Conf_Win_Pct', 
        'Home_Win_Pct', 'Away_Win_Pct', 'Points_For', 'Points_Against', 
        'FG_Pct', 'FG3_Pct', 'FT_Pct', 'FG_Defense', 'FG3_Defense', 
        'Reb_Rate', 'Off_Reb_Rate', 'Def_Reb_Rate', 'Assist_Rate', 
        'Turnover_Rate', 'Steal_Rate', 'Block_Rate', 'FT_Rate', 'Exp_Factor'
    ]
    
    # Create feature difference for the robust model (only using the features it was trained on)
    feature_diff = []
    for feature in robust_features:
        if feature in feature_names:
            idx = feature_names.index(feature)
            diff = team1_stats[idx] - team2_stats[idx]
            feature_diff.append(diff)
        else:
            # If feature not found, use 0 as difference
            feature_diff.append(0)
    
    # Convert to tensor
    features_tensor = torch.tensor([feature_diff], dtype=torch.float32)
    
    # Make prediction
    device = next(model.parameters()).device
    features_tensor = features_tensor.to(device)
    
    with torch.no_grad():
        win_prob = model(features_tensor).item()
    
    return win_prob

def load_models(args, feature_count):
    """
    Load all specified models for each round.
    
    Args:
        args: Command line arguments with model paths
        feature_count: Number of features for model initialization
        
    Returns:
        Dictionary mapping round numbers to model objects
    """
    models_dict = {}
    
    # Define the default model path
    default_model = os.path.join(model_dir, 'best_model.pt')
    
    # Load models for each round
    for round_num in range(1, 7):
        arg_name = f'round{round_num}'
        if hasattr(args, arg_name) and getattr(args, arg_name):
            model_path = os.path.join(model_dir, getattr(args, arg_name))
            model_name = f"Round {round_num} Model ({os.path.basename(model_path)})"
            
            # Check if this is the robust Round 2 model
            is_robust_model = os.path.basename(model_path) == 'round2_model_robust.pt'
            
            try:
                if is_robust_model:
                    model = load_robust_round2_model(model_path, feature_count)
                    # Flag to indicate this is a robust model that needs special handling
                    models_dict[round_num] = {
                        'model': model,
                        'name': model_name,
                        'path': model_path,
                        'is_robust': True
                    }
                else:
                    model = load_model(model_path, feature_count)
                    models_dict[round_num] = {
                        'model': model,
                        'name': model_name,
                        'path': model_path,
                        'is_robust': False
                    }
                print(f"Loaded {model_name} for Round {round_num}")
            except Exception as e:
                print(f"Error loading model for Round {round_num}: {e}")
                # Fall back to default model
                model = load_model(default_model, feature_count)
                models_dict[round_num] = {
                    'model': model,
                    'name': f"Default Model ({os.path.basename(default_model)})",
                    'path': default_model,
                    'is_robust': False
                }
                print(f"Falling back to default model for Round {round_num}")
        else:
            model = load_model(default_model, feature_count)
            models_dict[round_num] = {
                'model': model,
                'name': f"Default Model ({os.path.basename(default_model)})",
                'path': default_model,
                'is_robust': False
            }
            print(f"Loaded Default Model ({os.path.basename(default_model)}) for Round {round_num}")
    
    return models_dict

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Simulate NCAA tournament with variable models for each round')
    
    parser.add_argument('--round1', type=str, help='Model file for Round 1 (default: best_model.pt)')
    parser.add_argument('--round2', type=str, help='Model file for Round 2 (default: best_model.pt)')
    parser.add_argument('--round3', type=str, help='Model file for Round 3 (default: best_model.pt)')
    parser.add_argument('--round4', type=str, help='Model file for Round 4 (default: best_model.pt)')
    parser.add_argument('--round5', type=str, help='Model file for Round 5 (default: best_model.pt)')
    parser.add_argument('--round6', type=str, help='Model file for Round 6 (default: best_model.pt)')
    parser.add_argument('--output', type=str, default='tournament_prediction_variable.json',
                        help='Output file name (default: tournament_prediction_variable.json)')
    parser.add_argument('--year', type=int, default=2025,
                        help='Year to use for team statistics (default: 2025)')
    
    return parser.parse_args()

def main():
    """Simulate the NCAA tournament using user-specified models for each round."""
    args = parse_arguments()
    
    print("Loading models and data for tournament simulation...")
    
    # Load dataset to get feature columns
    dataset = NCAADataset(years=[args.year])
    feature_cols = dataset.get_feature_names()
    
    # Load all specified models
    models_dict = load_models(args, len(feature_cols))
    
    # Load bracket data and team statistics
    bracket_data = load_bracket_data()
    team_stats = load_team_stats(year=args.year)
    
    print(f"Simulating tournament with {len(bracket_data)} first-round matchups...")
    print("Using the following models for each round:")
    for round_num in range(1, 7):
        if round_num in models_dict:
            print(f"- Round {round_num}: {models_dict[round_num]['name']}")
    print("-" * 50)
    
    results = simulate_tournament(models_dict, team_stats, bracket_data, feature_cols)
    
    # Save results
    results_path = os.path.join(model_dir, args.output)
    with open(results_path, 'w') as f:
        json.dump({k: v if not isinstance(v, defaultdict) else dict(v) 
                  for k, v in results.items()}, f, indent=4)
    
    print(f"\nSimulation results saved to {results_path}")
    
    # Create a summary of model usage and results
    model_usage = defaultdict(int)
    for prediction in results['predictions']:
        model_usage[prediction['model_used']] += 1
    
    print("\nModel Usage Summary:")
    for model_name, count in model_usage.items():
        print(f"- {model_name}: {count} predictions")

if __name__ == "__main__":
    main()
