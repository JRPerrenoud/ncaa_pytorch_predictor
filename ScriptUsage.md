# Model Training Functions

## model.py

### How to run it:
Not meant to be run directly, but can be imported by other scripts
If run directly: python scripts/model.py

### Parameters:
No command-line parameters
When imported, provides classes and functions with their own parameters:
NCAAPredictor class: input_size, hidden_size, dropout_rate
train_model function: train_loader, val_loader, input_size, epochs, lr, weight_decay
evaluate_model function: model, data_loader
predict_matchup function: model, team1_stats, team2_stats, feature_cols

### What it does:
- Defines the neural network model architecture for predicting NCAA tournament outcomes
- Provides functions for training, evaluating, and using the model
- Includes model saving and loading utilities
- The model architecture uses two hidden layers with batch normalization and dropout
- Implements functions to evaluate model performance with metrics like accuracy, AUC, sensitivity, and specificity

### Output:
- When imported: Returns model objects and functions
- When run directly: Creates and trains a model, then evaluates it on example data

## train_model.py

### How to run it:
python scripts/train_model.py

### Parameters:
No command-line parameters

### What it does:
- Trains the general NCAA tournament prediction model
- Uses years 2013-2024 for training and 2022 for validation
- Creates data loaders for training and validation data
- Trains the model for 50 epochs with Adam optimizer
- Evaluates the model on the validation set
- Saves the best model during training and the final model

### Output:
- Prints training progress (loss, accuracy, AUC) for each epoch
- Shows validation metrics (accuracy, AUC, sensitivity, specificity)
- Displays the confusion matrix
- Prints example predictions from the validation set
- Saves the trained model to models/best_model.pt and models/final_model.pt
- Generates a plot of training history


## train_round2_model.py

### How to run it:
python scripts/train_round2_model.py

### Parameters:
No command-line parameters

### What it does:
- Trains a model specifically for predicting Round 2 matchups
- Creates a specialized dataset containing only Round 2 matchups
- Uses years 2013-2023 for training and 2024 for validation
- Trains the model for 50 epochs with Adam optimizer
- Evaluates the model on the validation set
- Evaluates the model on specific years (2022, 2023, 2024)
- Saves the trained model

### Output:
- Prints training progress (loss, accuracy, AUC) for each epoch
- Shows validation metrics (accuracy, AUC, sensitivity, specificity)
- Displays the confusion matrix
- Prints Round 2 accuracy for specific years
- Saves the trained model to models/round2_model.pt
- Generates a plot of training history


## train_round2_alt1_model.py

### How to run it:
python scripts/train_round2_alt1_model.py

### Parameters:
No command-line parameters

### What it does:
- Trains a simplified model specifically for Round 2 matchups
- Uses a different architecture with only one hidden layer
- Creates a specialized dataset containing only Round 2 matchups
- Uses years 2013-2023 for training and 2024 for validation
- Trains the model for 100 epochs with Adam optimizer
- Evaluates the model on the validation set
- Evaluates the model on specific years (2022, 2023, 2024)
- Saves the trained model

### Output:
- Prints training progress (loss, accuracy, AUC) for each epoch
- Shows validation metrics (accuracy, AUC, sensitivity, specificity)
- Displays the confusion matrix
- Prints Round 2 accuracy for specific years
- Saves the trained model to models/round2_robust_model.pt
- Generates a plot of training history
- Saves evaluation metrics to a JSON file



# Data Management Functions

## data_loader.py

### How to run it:
Not meant to be run directly, but can be imported by other scripts
If run directly: python scripts/data_loader.py

### Parameters:
No command-line parameters
When imported, provides classes and functions with their own parameters:
NCAADataset class: years, feature_cols, normalize
create_data_loaders function: years_train, years_val, batch_size, feature_cols

### What it does:
- Loads and processes NCAA basketball tournament data for model training
- Handles team name matching between different datasets with variations
- Creates feature vectors for each matchup by calculating differences between team statistics
- Normalizes features for better model training
- Prepares data in PyTorch Dataset format for training
- Contains a TeamNameMatcher class that handles matching team names with variations
- Provides a create_data_loaders function to create PyTorch DataLoaders

### Output:
- When imported: Returns dataset objects and data loaders
- When run directly: Prints example dataset information and sample matchups


## explore_data.py

### How to run it:
python scripts/explore_data.py

### Parameters:
No command-line parameters

### What it does:
- Loads raw NCAA basketball data from rawdata.csv
- Selects specific columns of interest for the model
- Creates a cleaned dataset with only the selected columns
- Calculates basic statistics for the cleaned data
- Saves the cleaned data to cleaned_data.csv

### Output:
- Prints information about the data (shape, selected columns, basic statistics)
- Saves cleaned data to cleaned_data.csv
- Displays the first 5 rows of the cleaned data



# Simulation Functions

## simulate_2025_multi_model_tournament.py

### How to run it:
python scripts/simulate_2025_multi_model_tournament.py

### Parameters:
No command-line parameters

### What it does:
- Simulates the 2025 NCAA tournament using multiple specialized models:
  - Round 1: General model (best_model.pt)
  - Round 2: Round 2-specific model (round2_model.pt)
  - Rounds 3-6: General model (best_model.pt)
- Loads the 2025 tournament bracket data
- Predicts winners for each matchup using the appropriate model for each round
- Advances winners through the bracket until a champion is determined
- Handles team name matching between the bracket and statistics

### Output:
- Prints predictions for each matchup with confidence percentages
- Shows which model is being used for each prediction
- Outputs the predicted tournament results, including Elite Eight, Final Four, and champion
- Saves the simulation results to a JSON file

#### TODO
This looks very close to a chalk bracket - need to do some fine-tuning to try to predict some more upsets


## simulate_2025_tournament.py

### How to run it:
python scripts/simulate_2025_tournament.py

### Parameters:
No command-line parameters

### What it does:
- Simulates the 2025 NCAA tournament using a single model (best_model.pt)
- Loads the 2025 tournament bracket data
- Predicts winners for each matchup
- Advances winners through the bracket until a champion is determined
- Handles team name matching between the bracket and statistics

### Output:
- Prints predictions for each matchup with confidence percentages
- Outputs the predicted tournament results, including Elite Eight, Final Four, and champion
- Saves the simulation results to a JSON file

#### TODO
This looks very close to a chalk bracket - need to do some fine-tuning to try to predict some more upsets



## simulate_tournament.py

### How to run it:
python scripts/simulate_tournament.py --year YEAR

### Parameters:
--year: Tournament year to simulate (default: 2024)

### What it does:
- Simulates a past NCAA tournament using the trained model
- Loads actual tournament results for the specified year
- Compares model predictions with actual outcomes
- Calculates accuracy metrics overall and by round
- Identifies interesting matchups (closest predictions and biggest upsets)
- This script can be used to test the model's performance on past tournaments where
we have known results so that we can see how well the model performs.

### Output:
- Prints accuracy by round and overall accuracy
- Lists the most uncertain predictions (closest to 50% probability)
- Shows the biggest upsets/misses (where the model was confident but wrong)
- Saves the simulation results to a JSON file



## simulate_tournament_multi_models.py

### How to run it:
python scripts/simulate_tournament_multi_models.py --years YEAR1 YEAR2 ... --round2_model PATH

### Parameters:
--years: Tournament years to simulate (default: 2017 2019 2022 2023 2024)
--round2_model: Path to the Round 2 specific model (default: models/round2_model.pt)

### What it does:
- Simulates multiple past NCAA tournaments using different models for different rounds:
    - Round 1: General model
    - Round 2: Round 2-specific model
    - Rounds 3+: General model
- Compares model predictions with actual outcomes
- Calculates accuracy metrics overall and by round for each year
- Identifies interesting matchups (closest predictions and biggest upsets)
- This script can be used to test the model's performance on past tournaments where
we have known results so that we can see how well the model performs.

### Output:
- Prints accuracy by round and overall accuracy for each year
- Lists the most uncertain predictions (closest to 50% probability)
- Shows the biggest upsets/misses (where the model was confident but wrong)
- Saves the simulation results to a JSON file for each year


## simulate_2025_tournament_variable.py

### How to run it:
python scripts/simulate_2025_tournament_variable.py --round1 MODEL1 --round2 MODEL2 ... --round6 MODEL6

### Parameters:
--round1: Model to use for Round 1 (default: best_model.pt)
--round2: Model to use for Round 2 (default: best_model.pt)
--round3: Model to use for Round 3 (default: best_model.pt)
--round4: Model to use for Round 4 (default: best_model.pt)
--round5: Model to use for Round 5 (default: best_model.pt)
--round6: Model to use for Round 6 (default: best_model.pt)

### What it does:
- Simulates the 2025 NCAA tournament using user-specified models for each round
- Allows complete customization of which model to use for each of the 6 tournament rounds
- Loads the 2025 tournament bracket data
- Predicts winners for each matchup using the specified model for each round
- Advances winners through the bracket until a champion is determined
- Handles team name matching between the bracket and statistics
- Includes special handling for "robust" models with different architectures

### Output:
- Prints predictions for each matchup with confidence percentages
- Shows which model is being used for each prediction
- Outputs the predicted tournament results, including Elite Eight, Final Four, and champion
- Saves the simulation results to a JSON file



# Analysis Functions

## feature_importance.py

### How to run it:
python exploring/feature_importance.py

### Parameters:
This script doesn't accept any command-line parameters.

### What the script does:
The script analyzes which features are most important for the general NCAA tournament prediction model using a permutation importance approach. 
It:
- Loads the best_model.pt from the models directory
- Loads the 2022 validation dataset
- Calculates permutation importance by:
  - Measuring baseline accuracy of the model
  - For each feature, shuffling its values and measuring the decrease in accuracy
  - Repeating this process 5 times per feature to get a reliable estimate
- Sorts features by importance (those that cause the largest accuracy drop when shuffled)
- Analyzes feature correlations with outcomes across tournaments from 2019-2024
- Identifies which features are most positively and negatively correlated with winning

### What is output by the script:
- Console output:
  - Baseline accuracy of the model
  - Progress updates during calculation
  - Top 10 most important features
  - 5 least important features
  - Top 10 features correlated with winning
  - Top 10 features correlated with losing
- Files:
  - models/feature_importance.png: Bar chart of the top 15 most important features
  - models/feature_importance.csv: CSV file with importance scores for all features
  - models/feature_correlations.csv: CSV file with correlation values between features and outcomes


## feature_importance_round2.py

### How to run it:
python exploring/feature_importance_round2.py

### Parameters:
This script doesn't accept any command-line parameters.

### What the script does:
The script compares feature importance between the general model and the Round 2-specific model. 
It:
- Loads both the general model (best_model.pt) and the Round 2 model (round2_model.pt)
- Loads the 2024 validation dataset
- Calculates permutation importance for both models using the same approach as feature_importance.py
- Compares the importance of features between the two models
- Identifies which features have the biggest differences in importance between models
- This analysis helps understand why the Round 2 model performs differently from the general model, which is particularly relevant given your interest in improving Round 2 predictions where the general model shows a significant performance drop-off (from 96.54% in Round 1 to 67.42% in Round 2).

### What is output by the script:
- Console output:
  - Baseline accuracy for both models
  - Progress updates during calculation
  - Top 10 most important features for each model
  - 5 least important features for each model
  - Features with the biggest importance differences between models

### Output:
- Files:
  - models/feature_importance_general_model.png: Bar chart of top features for general model
  - models/feature_importance_general_model.csv: CSV with importance scores for general model
  - models/feature_importance_round_2_model.png: Bar chart of top features for Round 2 model
  - models/feature_importance_round_2_model.csv: CSV with importance scores for Round 2 model
  - models/feature_importance_comparison.png: Comparative bar chart showing differences between models
models/feature_importance_comparison.csv: CSV with detailed comparison data