import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import mlflow
import mlflow.sklearn# Placeholder for train_model.py

# Set the URI to your desired MLflow tracking server with host and port
mlflow.set_tracking_uri("http://127.0.0.1:8080")

def train_random_forest(file_path, random_state=42):
    """
    Train a Random Forest classifier on the given dataset.

    Args:
        file_path (str): The path to the CSV file containing the dataset.
        test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.2.
        random_state (int, optional): The seed used by the random number generator. Defaults to 42.
    """
    # Load and prepare the data
    pred_df = pd.read_csv(file_path, index_col=0)

    pred_df['Home'] = pred_df['Home'].astype('category')
    pred_df['Away'] = pred_df['Away'].astype('category')
    pred_df['FTR'] = pred_df['FTR'].astype('category')

    # Ensure 'Date' is in datetime format
    pred_df['Date'] = pd.to_datetime(pred_df['Date'])

    # Split data into train and test sets based on date
    train = pred_df[pred_df['Date'] < '2024-01-01']
    test = pred_df[pred_df['Date'] >= '2024-01-01']

    # Define predictors and target for train and test sets
    X_train = train[['Home', 'Away', 'Attendance', 'B365H', 'B365D', 'B365A',
                     'AttackStrengthHome', 'AttackStrengthAway', 'DefenseWeaknessHome',
                     'DefenseWeaknessAway', 'AvgHomePoints', 'AvgAwayPoints',
                     'AvgLosingHomePoints', 'AvgLosingAwayPoints', 'AvgGoalDiffHome',
                     'AvgGoalDiffAway', 'HomeWinsRatio', 'HomeDrawsRatio', 'AwayWinsRatio',
                     'AwayDrawsRatio', 'AvgHomeCornersLast5', 'AvgAwayCornersLast5',
                     'AvgHomeShotsLast5', 'AvgHomeShotsOnTargetLast5', 'AvgAwayShotsLast5',
                     'AvgAwayShotsOnTargetLast5', 'elo', 'elo_away', 'FormHomeTeam',
                     'FormAwayTeam', 'ProbabilityDraw', 'ProbabilityHomeWin',
                     'ProbabilityAwayWin']]
    y_train = train['FTR']
    X_test = test[X_train.columns]
    y_test = test['FTR']

    # Standardize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define the parameter grid with stricter regularization
    param_grid = {
        'n_estimators': [50, 500],  # Use fewer estimators to reduce complexity
        'max_depth': [5, 10, 15],
        'max_features': ['sqrt', 'log2']  # Limit the number of features considered for splitting at each node
    }

    # Initialize TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)

    # Initialize RandomForestClassifier
    rf_model = RandomForestClassifier(random_state=random_state)

    # Initialize GridSearchCV with TimeSeriesSplit on training set only
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=tscv, n_jobs=-1, scoring='accuracy')
    grid_search.fit(X_train_scaled, y_train)

    # Get the best parameters
    best_params = grid_search.best_params_
    print("Best Parameters:", best_params)

    # Get the best model
    best_rf_model = grid_search.best_estimator_

    # Train the best model on the entire training set
    best_rf_model.fit(X_train_scaled, y_train)

    # Make predictions on training set
    y_train_pred = best_rf_model.predict(X_train_scaled)

    # Make predictions on test set
    y_test_pred = best_rf_model.predict(X_test_scaled)

    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')

    # Print the metrics
    print(f"Training Accuracy with Best Random Forest: {train_accuracy:.4f}")
    print(f"Test Accuracy with Best Random Forest: {test_accuracy:.4f}")
    print(f"Training F1 Score with Best Random Forest: {train_f1:.4f}")
    print(f"Test F1 Score with Best Random Forest: {test_f1:.4f}")

    # Print the confusion matrix
    print("Confusion Matrix - Training Set with Best Random Forest")
    print(confusion_matrix(y_train, y_train_pred))

    print("Confusion Matrix - Test Set with Best Random Forest")
    print(confusion_matrix(y_test, y_test_pred))

    # Print classification report
    print("Classification Report - Training Set with Best Random Forest")
    print(classification_report(y_train, y_train_pred))

    print("Classification Report - Test Set with Best Random Forest")
    print(classification_report(y_test, y_test_pred))

    # Set our tracking server uri for logging
    #mlflow server --host 127.0.0.1 --port 8080
    
    # Set experiment name
    mlflow.set_experiment("Random Forest Model Training")

    # Start an MLflow run
    with mlflow.start_run():

        # Log the accuracy and F1 score metrics
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("train_f1", train_f1)
        mlflow.log_metric("test_f1", test_f1)

        # Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tag("Training Info", "Random Forest model training with accuracy and F1 score metrics.")

        # Log the model
        mlflow.sklearn.log_model(
            sk_model=best_rf_model,
            artifact_path="random_forest_model",
            registered_model_name="RandomForestClassifier"
        )

if __name__ == "__main__":
    train_random_forest()
