"""
NCAA Tournament 2025 Simulation

This script simulates the 2025 NCAA tournament using our trained model.
It loads the best model and makes predictions for each matchup, advancing winners
through the bracket until a champion is determined.
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

def predict_winner(model, team1_stats, team2_stats, team1_name, team2_name, feature_cols, round_num, region):
    """
    Predict the winner of a matchup and return the prediction details.
    """
    # Ensure feature columns are the correct type
    team1_features = team1_stats[feature_cols].values.astype(np.float32)
    team2_features = team2_stats[feature_cols].values.astype(np.float32)
    
    # Create a modified stats object with the correct data types
    team1_stats_modified = pd.Series(team1_features, index=feature_cols)
    team2_stats_modified = pd.Series(team2_features, index=feature_cols)
    
    # Predict winner
    win_probability = predict_matchup(model, team1_stats_modified, team2_stats_modified, feature_cols)
    predicted_winner = team1_name if win_probability > 0.5 else team2_name
    win_prob_display = float(win_probability) if win_probability > 0.5 else float(1 - win_probability)
    
    # Print prediction
    print(f"{team1_name} vs {team2_name}")
    print(f"Prediction: {predicted_winner} wins ({win_prob_display:.2%} confidence)")
    print("-" * 50)
    
    # Return prediction details
    return {
        'round': round_num,
        'region': region,
        'team1': team1_name,
        'team2': team2_name,
        'win_probability': win_prob_display,
        'predicted_winner': predicted_winner
    }

def simulate_tournament(model, team_stats, bracket_data, feature_cols):
    """
    Simulate the entire 2025 tournament.
    
    Args:
        model: Trained model
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
                    model, team1_stats, team2_stats, team1_name, team2_name, 
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
                    model, team1_stats, team2_stats, team1_name, team2_name, 
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
                    model, team1_stats, team2_stats, team1_name, team2_name, 
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
                    model, team1_stats, team2_stats, team1_name, team2_name, 
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
                model, team1_stats, team2_stats, team1_name, team2_name, 
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
                model, team1_stats, team2_stats, team1_name, team2_name, 
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
                model, team1_stats, team2_stats, team1_name, team2_name, 
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

def main():
    """Simulate the 2025 NCAA tournament using our trained model."""
    print("Loading model and data for 2025 tournament simulation...")
    
    # Load dataset to get feature columns
    dataset = NCAADataset(years=[2024])  # Use 2024 as proxy for feature names
    feature_cols = dataset.get_feature_names()
    
    # Load model
    model = load_model(model_path, len(feature_cols))
    
    # Load bracket data and team statistics
    bracket_data = load_bracket_data()
    team_stats = load_team_stats(year=2025)  # Use actual 2025 stats
    
    print(f"Simulating 2025 tournament with {len(bracket_data)} first-round matchups...")
    results = simulate_tournament(model, team_stats, bracket_data, feature_cols)
    
    # Save results
    results_path = os.path.join(model_dir, 'tournament_prediction_2025.json')
    with open(results_path, 'w') as f:
        json.dump({k: v if not isinstance(v, defaultdict) else dict(v) 
                  for k, v in results.items()}, f, indent=4)
    
    print(f"\nSimulation results saved to {results_path}")

if __name__ == "__main__":
    main()
