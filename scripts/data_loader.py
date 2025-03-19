"""
NCAA Tournament Data Loader

This script loads and processes NCAA basketball tournament data for model training.
It handles:
1. Loading team statistics from cleaned_data.csv
2. Loading tournament results from JSON files
3. Matching teams across datasets (handling name variations)
4. Creating feature vectors for each matchup
5. Preparing data for PyTorch training
"""

import os
import json
import pandas as pd
import numpy as np
import re
from difflib import SequenceMatcher
import torch
from torch.utils.data import Dataset, DataLoader

# Set paths
data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
stats_path = os.path.join(data_dir, 'cleaned_data.csv')

class TeamNameMatcher:
    """Handles matching team names between different datasets with variations."""
    
    def __init__(self):
        # Common name variations to standardize
        self.name_variations = {
            'st': ['state', 'st.', 'st'],
            'saint': ['st.', 'st', 'saint'],
            'nc': ['north carolina', 'nc'],
            'uc': ['california', 'uc'],
            'a&m': ['a & m', 'a&m'],
            'fl': ['florida', 'fl'],
            'usc': ['southern california', 'usc'],
            'pitt': ['pittsburgh', 'pitt'],
            'lsu': ['louisiana state', 'lsu'],
            'smu': ['southern methodist', 'smu'],
            'byu': ['brigham young', 'byu'],
            'ucf': ['central florida', 'ucf'],
            'vcu': ['virginia commonwealth', 'vcu'],
            'ole miss': ['mississippi', 'ole miss']
        }
        
        # Cache for previously matched names
        self.match_cache = {}
    
    def standardize_name(self, name):
        """Standardize team name by removing common elements and lowercasing."""
        # Convert to lowercase and remove 'university', 'college', etc.
        name = name.lower()
        name = re.sub(r'university|college|the|of|at|\(.*?\)', '', name)
        # Remove punctuation and extra spaces
        name = re.sub(r'[^\w\s]', '', name)
        name = re.sub(r'\s+', ' ', name).strip()
        return name
    
    def similarity_score(self, name1, name2):
        """Calculate similarity between two team names."""
        # Use SequenceMatcher for string similarity
        return SequenceMatcher(None, name1, name2).ratio()
    
    def find_best_match(self, team_name, available_teams):
        """Find the best matching team name from available teams."""
        # Check cache first
        cache_key = (team_name, tuple(available_teams))
        if cache_key in self.match_cache:
            return self.match_cache[cache_key]
        
        std_name = self.standardize_name(team_name)
        best_match = None
        best_score = 0
        
        for team in available_teams:
            std_team = self.standardize_name(team)
            score = self.similarity_score(std_name, std_team)
            
            # Check for common variations
            for key, variations in self.name_variations.items():
                if key in std_name and any(var in std_team for var in variations):
                    score += 0.2  # Boost score for known variations
                if key in std_team and any(var in std_name for var in variations):
                    score += 0.2
            
            if score > best_score:
                best_score = score
                best_match = team
        
        # Only accept matches above a threshold
        if best_score > 0.6:
            self.match_cache[cache_key] = best_match
            return best_match
        return None

class NCAADataset(Dataset):
    """PyTorch Dataset for NCAA tournament data."""
    
    def __init__(self, years=None, feature_cols=None, normalize=True):
        """
        Initialize the dataset.
        
        Args:
            years: List of years to include (None for all years)
            feature_cols: List of columns to use as features (None for all numeric)
            normalize: Whether to normalize features
        """
        self.years = years if years is not None else list(range(2013, 2025))
        self.normalize = normalize
        
        # Load team statistics
        self.stats_df = pd.read_csv(stats_path)
        
        # Select feature columns (exclude non-numeric and identifier columns)
        if feature_cols is None:
            self.feature_cols = self.stats_df.select_dtypes(include=[np.number]).columns.tolist()
            # Remove YEAR, SEED, ROUND, GAMES, W, L columns if present
            exclude_cols = ['YEAR', 'SEED', 'ROUND', 'GAMES', 'W', 'L']
            self.feature_cols = [col for col in self.feature_cols if col not in exclude_cols]
        else:
            self.feature_cols = feature_cols
        
        # Initialize team name matcher
        self.name_matcher = TeamNameMatcher()
        
        # Load and process tournament data
        self.matchups = self._load_tournament_data()
        
        # Normalize features if requested
        if normalize:
            self._normalize_features()
    
    def _load_tournament_data(self):
        """Load tournament data from JSON files and match with team statistics."""
        all_matchups = []
        
        for year in self.years:
            # Skip years without data
            results_path = os.path.join(data_dir, f'results_{year}.json')
            if not os.path.exists(results_path):
                print(f"Warning: No tournament data found for {year}")
                continue
            
            try:
                # Load tournament results
                with open(results_path, 'r') as f:
                    results = json.load(f)
                
                # Get teams available in stats for this year
                year_stats = self.stats_df[self.stats_df['YEAR'] == year]
                if len(year_stats) == 0:
                    print(f"Warning: No team statistics found for {year}")
                    continue
                
                available_teams = year_stats['TEAM'].unique().tolist()
                
                # Process each matchup
                for matchup in results:
                    if matchup['year'] != year:
                        print(f"Warning: Year mismatch in {results_path}: {matchup['year']} != {year}")
                        continue
                    
                    # Match team names to statistics
                    team1_name = matchup['team1']
                    team2_name = matchup['team2']
                    winner_name = matchup['winner']
                    
                    team1_match = self.name_matcher.find_best_match(team1_name, available_teams)
                    team2_match = self.name_matcher.find_best_match(team2_name, available_teams)
                    
                    if team1_match is None or team2_match is None:
                        print(f"Warning: Could not match teams for {team1_name} vs {team2_name} in {year}")
                        continue
                    
                    # Determine winner (1 for team1, 0 for team2)
                    if self.name_matcher.similarity_score(winner_name, team1_name) > self.name_matcher.similarity_score(winner_name, team2_name):
                        winner = 1
                    else:
                        winner = 0
                    
                    # Get team statistics
                    team1_stats = year_stats[year_stats['TEAM'] == team1_match].iloc[0]
                    team2_stats = year_stats[year_stats['TEAM'] == team2_match].iloc[0]
                    
                    # Create matchup record
                    matchup_data = {
                        'year': year,
                        'round': matchup['round'],
                        'team1': team1_match,
                        'team2': team2_match,
                        'team1_features': team1_stats[self.feature_cols].values,
                        'team2_features': team2_stats[self.feature_cols].values,
                        'winner': winner
                    }
                    
                    all_matchups.append(matchup_data)
            except Exception as e:
                print(f"Error processing tournament data for {year}: {str(e)}")
        
        if not all_matchups:
            raise ValueError("No valid matchups found. Check your data files and team name matching.")
            
        return all_matchups
    
    def _normalize_features(self):
        """Normalize features to have zero mean and unit variance."""
        # Calculate mean and std for each feature across all teams
        all_features = np.vstack([m['team1_features'] for m in self.matchups] + 
                                [m['team2_features'] for m in self.matchups])
        
        self.feature_mean = np.mean(all_features, axis=0)
        self.feature_std = np.std(all_features, axis=0, dtype=np.float64)
        
        # Replace zero std with 1 to avoid division by zero
        self.feature_std = np.where(self.feature_std < 0.0001, 1.0, self.feature_std)
        
        # Normalize features
        for matchup in self.matchups:
            matchup['team1_features'] = (matchup['team1_features'] - self.feature_mean) / self.feature_std
            matchup['team2_features'] = (matchup['team2_features'] - self.feature_mean) / self.feature_std
    
    def __len__(self):
        """Return the number of matchups."""
        return len(self.matchups)
    
    def __getitem__(self, idx):
        """Get a matchup by index."""
        matchup = self.matchups[idx]
        
        # Create feature vector (difference between team1 and team2)
        features = matchup['team1_features'] - matchup['team2_features']
        
        # Convert to PyTorch tensors - ensure features are float32
        features_tensor = torch.tensor(features.astype(np.float32), dtype=torch.float32)
        label_tensor = torch.tensor(matchup['winner'], dtype=torch.float32)
        
        return features_tensor, label_tensor
    
    def get_feature_names(self):
        """Return the names of features used."""
        return self.feature_cols
    
    def get_matchup_info(self, idx):
        """Return human-readable information about a matchup."""
        matchup = self.matchups[idx]
        return {
            'year': matchup['year'],
            'round': matchup['round'],
            'team1': matchup['team1'],
            'team2': matchup['team2'],
            'winner': matchup['team1'] if matchup['winner'] == 1 else matchup['team2']
        }

def create_data_loaders(years_train=None, years_val=None, batch_size=32, feature_cols=None):
    """
    Create PyTorch DataLoaders for training and validation.
    
    Args:
        years_train: List of years to use for training
        years_val: List of years to use for validation
        batch_size: Batch size for DataLoader
        feature_cols: List of columns to use as features
    
    Returns:
        train_loader, val_loader: PyTorch DataLoaders
    """
    # Default to using 2013-2023 for training, 2024 for validation
    if years_train is None:
        years_train = list(range(2013, 2024))
    if years_val is None:
        years_val = [2024]
    
    # Create datasets
    train_dataset = NCAADataset(years=years_train, feature_cols=feature_cols)
    val_dataset = NCAADataset(years=years_val, feature_cols=feature_cols)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

if __name__ == "__main__":
    # Example usage
    print("Loading NCAA tournament data...")
    
    # Create dataset for all years
    dataset = NCAADataset(years=list(range(2013, 2025)))
    
    # Print dataset statistics
    print(f"Loaded {len(dataset)} matchups")
    print(f"Using {len(dataset.get_feature_names())} features: {dataset.get_feature_names()}")
    
    # Print a few example matchups
    print("\nExample matchups:")
    for i in range(min(5, len(dataset))):
        info = dataset.get_matchup_info(i)
        print(f"{info['year']} Round {info['round']}: {info['team1']} vs {info['team2']} - Winner: {info['winner']}")
    
    # Create train/val split
    train_loader, val_loader = create_data_loaders()
    print(f"\nCreated training loader with {len(train_loader.dataset)} matchups")
    print(f"Created validation loader with {len(val_loader.dataset)} matchups")
    
    # Debug team name matching
    print("\nTesting team name matching:")
    matcher = TeamNameMatcher()
    test_cases = [
        ("UConn", ["Connecticut", "UConn", "UC"]),
        ("North Carolina", ["NC", "North Carolina", "UNC"]),
        ("Michigan St.", ["Michigan", "Michigan State", "MSU"]),
        ("Saint Mary's", ["St. Mary's", "Saint Mary's", "SMC"]),
        ("Texas A&M", ["Texas A&M", "Texas A & M", "TAMU"])
    ]
    
    for test_name, available_teams in test_cases:
        match = matcher.find_best_match(test_name, available_teams)
        print(f"'{test_name}' matched to '{match}'")
