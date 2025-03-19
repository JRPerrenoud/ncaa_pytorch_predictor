"""
Explore and preprocess the NCAA basketball data.
Focus on specific columns as identified in the original R package.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set paths
data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
raw_data_path = os.path.join(data_dir, 'rawdata.csv')

# Load the raw data
print("Loading raw data...")
raw_data = pd.read_csv(raw_data_path)
print(f"Raw data shape: {raw_data.shape}")

# Select only the columns we're interested in
# Columns 1, 2, 8, 9, 10, 13, 17, 21, 25, 26, 27, and 29-64
# Note: Python is 0-indexed, so we subtract 1 from the column numbers
selected_columns = [0, 1, 7, 8, 9, 12, 16, 20, 24, 25, 26] + list(range(28, 64))

# Get the column names for better readability
selected_column_names = raw_data.columns[selected_columns].tolist()
print("\nSelected columns:")
for i, col in enumerate(selected_column_names):
    print(f"{i+1}. {col}")

# Create a cleaned dataset with only the selected columns
cleaned_data = raw_data.iloc[:, selected_columns].copy()
print(f"\nCleaned data shape: {cleaned_data.shape}")

# Display basic statistics for the cleaned data
print("\nBasic statistics for numeric columns:")
numeric_stats = cleaned_data.describe().T
print(numeric_stats[['count', 'mean', 'std', 'min', 'max']])

# Save the cleaned data
cleaned_data_path = os.path.join(data_dir, 'cleaned_data.csv')
cleaned_data.to_csv(cleaned_data_path, index=False)
print(f"\nCleaned data saved to {cleaned_data_path}")

# Display the first few rows of the cleaned data
print("\nFirst 5 rows of cleaned data:")
print(cleaned_data.head())

if __name__ == "__main__":
    print("\nData exploration complete!")
