"""
Logistic Regression Implementation: Ultramarathon Completion Prediction
=====================================================================

This script demonstrates a complete logistic regression implementation using scikit-learn
to predict whether ultramarathon runners will complete a 50-mile race based on their
weekly training mileage.

Author: Muatasim Ahmed
Date: July 2025
Dataset: Mock ultramarathon training data
"""

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def create_mock_data():
    """
    Creates mock ultramarathon training data.
    
    Returns:
        dict: Dictionary containing training miles and completion status
    """
    # Generate mock data for 100 ultramarathon participants
    n_participants = 100
    
    # Create realistic training data with some correlation to completion
    # Lower mileage runners have lower completion probability
    miles_per_week = np.random.normal(45, 15, n_participants)
    miles_per_week = np.clip(miles_per_week, 15, 80)  # Realistic range
    
    # Create completion status based on training miles with some randomness
    completion_prob = 1 / (1 + np.exp(-(miles_per_week - 40) / 8))
    completed = np.random.binomial(1, completion_prob, n_participants)
    completed_text = ['yes' if x == 1 else 'no' for x in completed]
    
    return {
        'miles_per_week': miles_per_week,
        'completed_50_mile_ultra': completed_text
    }

def load_and_prepare_data():
    """
    Loads and prepares the dataset for analysis.
    
    Returns:
        pd.DataFrame: Prepared DataFrame with encoded target variable
    """
    print("=" * 60)
    print("STEP 1: DATA LOADING AND PREPARATION")
    print("=" * 60)
    
    # Create mock data
    d = create_mock_data()
    
    # Create DataFrame
    df = pd.DataFrame(d)
    print(f"Dataset created with {len(df)} participants")
    print(f"Dataset shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    
    # Display basic statistics
    print(f"\nDataset Info:")
    print(f"- Average training miles per week: {df['miles_per_week'].mean():.2f}")
    print(f"- Min training miles: {df['miles_per_week'].min():.2f}")
    print(f"- Max training miles: {df['miles_per_week'].max():.2f}")
    print(f"- Completion rate: {(df['completed_50_mile_ultra'] == 'yes').mean():.2%}")
    
    return df

def encode_target_variable(df):
    """
    Encodes the categorical target variable to numerical format.
    
    Args:
        df (pd.DataFrame): DataFrame with categorical target
        
    Returns:
        pd.DataFrame: DataFrame with encoded target variable
    """
    print("\n" + "=" * 60)
    print("STEP 2: ENCODING CATEGORICAL TARGET VARIABLE")
    print("=" * 60)
    
    # Define the order for ordinal encoding
    finished_race = ['no', 'yes']  # 'no' will be 0, 'yes' will be 1
    
    # Create and fit the ordinal encoder
    enc = OrdinalEncoder(categories=[finished_race])
    
    # Transform the target column
    df['completed_50_mile_ultra'] = enc.fit_transform(df[['completed_50_mile_ultra']])
    
    print("Target variable encoded:")
    print("- 'no' (did not complete) → 0")
    print("- 'yes' (completed) → 1")
    print(f"\nEncoded target distribution:")
    print(df['completed_50_mile_ultra'].value_counts().sort_index())
    
    return df

def perform_eda(df):
    """
    Performs exploratory data analysis on the dataset.
    
    Args:
        df (pd.DataFrame): Prepared DataFrame
    """
    print("\n" + "=" * 60)
    print("STEP 3: EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Scatter plot showing the relationship
    axes[0].scatter(df['miles_per_week'], df['completed_50_mile_ultra'], 
                   alpha=0.6, color='blue', edgecolors='black', linewidth=0.5)
    axes[0].set_xlabel('Miles per Week (Training)')
    axes[0].set_ylabel('Completed 50-Mile Ultra (0=No, 1=Yes)')
    axes[0].set_title('Training Miles vs. Race Completion')
    axes[0].grid(True, alpha=0.3)
    
    # Count plot showing distribution of outcomes
    completion_counts = df['completed_50_mile_ultra'].value_counts().sort_index()
    axes[1].bar(['Did Not Complete (0)', 'Completed (1)'], completion_counts.values, 
               color=['red', 'green'], alpha=0.7, edgecolor='black')
    axes[1].set_ylabel('Number of Participants')
    axes[1].set_title('Distribution of Race Completion')
    axes[1].grid(True, alpha=0.3)
    
    # Add count labels on bars
    for i, v in enumerate(completion_counts.values):
        axes[1].text(i, v + 1, str(v), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print("EDA Insights:")
    print("- The scatter plot shows the characteristic S-curve pattern suitable for logistic regression")
    print("- Higher training mileage generally correlates with higher completion probability")
    print("- Some outliers exist (high mileage non-completers), which is realistic")

def split_data(df):
    """
    Splits the data into training and testing sets.
    
    Args:
        df (pd.DataFrame): Prepared DataFrame
        
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    print("\n" + "=" * 60)
    print("STEP 4: DATA SPLITTING")
    print("=" * 60)
    
    # Separate features and target
    # Use iloc to keep X as 2D array (required by sklearn)
    X = df.iloc[:, 0:1]  # miles_per_week (as DataFrame)
    y = df.iloc[:, 1]    # completed_50_mile_ultra (as Series)
    
    print(f"Features (X) shape: {X.shape}")
    print(f"Target (y) shape: {y.shape}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        train_size=0.8, 
        random_state=42,  # For reproducibility
        stratify=y        # Maintain class distribution in both sets
    )
    
    print(f"\nData split completed:")
    print(f"- Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(df)*100:.1f}%)")
    print(f"- Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(df)*100:.1f}%)")
    print(f"- Training completion rate: {y_train.mean():.2%}")
    print(f"- Test completion rate: {y_test.mean():.2%}")
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """
    Trains the logistic regression model.
    
    Args:
        X_train: Training features
        y_train: Training target
        
    Returns:
        LogisticRegression: Trained model
    """
    print("\n" + "=" * 60)
    print("STEP 5: MODEL TRAINING")
    print("=" * 60)
    
    # Create and train the logistic regression model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    print("Logistic Regression model trained successfully!")
    print(f"Model coefficients:")
    print(f"- Slope (coefficient): {model.coef_[0][0]:.4f}")
    print(f"- Intercept: {model.intercept_[0]:.4f}")
    
    # Display the logistic regression equation
    print(f"\nLogistic Regression Equation:")
    print(f"p = 1 / (1 + e^(-({model.coef_[0][0]:.4f} * miles_per_week + {model.intercept_[0]:.4f})))")
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model on test data.
    
    Args:
        model: Trained logistic regression model
        X_test: Test features
        y_test: Test target
        
    Returns:
        np.array: Predictions on test set
    """
    print("\n" + "=" * 60)
    print("STEP 6: MODEL EVALUATION")
    print("=" * 60)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of completion
    
    # Calculate accuracy
    accuracy = model.score(X_test, y_test)
    print(f"Model Accuracy: {accuracy:.4f} ({accuracy:.2%})")
    
    # Confusion Matrix
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Create a more readable confusion matrix
    print(f"\nConfusion Matrix Breakdown:")
    tn, fp, fn, tp = cm.ravel()
    print(f"- True Negatives (correctly predicted non-completion): {tn}")
    print(f"- False Positives (incorrectly predicted completion): {fp}")
    print(f"- False Negatives (incorrectly predicted non-completion): {fn}")
    print(f"- True Positives (correctly predicted completion): {tp}")
    
    # Classification Report
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Did Not Complete', 'Completed']))
    
    # Additional metrics
    print(f"\nAdditional Insights:")
    print(f"- Sensitivity (Recall for Completed): {tp/(tp+fn):.3f}")
    print(f"- Specificity (Recall for Not Completed): {tn/(tn+fp):.3f}")
    print(f"- Average prediction probability for completers: {y_pred_proba[y_test==1].mean():.3f}")
    print(f"- Average prediction probability for non-completers: {y_pred_proba[y_test==0].mean():.3f}")
    
    return y_pred

def visualize_results(model, df, X_test, y_test, y_pred):
    """
    Creates visualizations of the model results.
    
    Args:
        model: Trained model
        df: Original DataFrame
        X_test: Test features
        y_test: Test target
        y_pred: Predictions
    """
    print("\n" + "=" * 60)
    print("STEP 7: RESULTS VISUALIZATION")
    print("=" * 60)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Logistic regression curve with data points
    miles_range = np.linspace(df['miles_per_week'].min(), df['miles_per_week'].max(), 100)
    probabilities = model.predict_proba(miles_range.reshape(-1, 1))[:, 1]
    
    axes[0].scatter(df['miles_per_week'], df['completed_50_mile_ultra'], 
                   alpha=0.6, color='lightblue', label='Training Data')
    axes[0].plot(miles_range, probabilities, 'r-', linewidth=2, label='Logistic Curve')
    axes[0].set_xlabel('Miles per Week (Training)')
    axes[0].set_ylabel('Probability of Completion')
    axes[0].set_title('Logistic Regression Fit')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Prediction accuracy visualization
    correct_predictions = (y_test == y_pred)
    axes[1].scatter(X_test[correct_predictions], y_test[correct_predictions], 
                   color='green', alpha=0.7, label='Correct Predictions', s=60)
    axes[1].scatter(X_test[~correct_predictions], y_test[~correct_predictions], 
                   color='red', alpha=0.7, label='Incorrect Predictions', s=60, marker='x')
    axes[1].set_xlabel('Miles per Week (Training)')
    axes[1].set_ylabel('Actual Completion (0=No, 1=Yes)')
    axes[1].set_title('Prediction Accuracy on Test Set')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def make_sample_predictions(model):
    """
    Makes predictions for sample training scenarios.
    
    Args:
        model: Trained logistic regression model
    """
    print("\n" + "=" * 60)
    print("STEP 8: SAMPLE PREDICTIONS")
    print("=" * 60)
    
    # Sample training miles for prediction
    sample_miles = [25, 35, 45, 55, 65]
    
    print("Predictions for different training scenarios:")
    print("-" * 50)
    
    for miles in sample_miles:
        prob = model.predict_proba([[miles]])[0][1]
        prediction = model.predict([[miles]])[0]
        result = "Complete" if prediction == 1 else "Not Complete"
        
        print(f"Training {miles} miles/week:")
        print(f"  → Probability of completion: {prob:.3f} ({prob:.1%})")
        print(f"  → Prediction: {result}")
        print()

def main():
    """
    Main function that orchestrates the entire logistic regression workflow.
    """
    print("LOGISTIC REGRESSION: ULTRAMARATHON COMPLETION PREDICTION")
    print("=" * 80)
    print("This script demonstrates predicting ultramarathon completion based on training miles.")
    print("=" * 80)
    
    # Execute the complete workflow
    df = load_and_prepare_data()
    df = encode_target_variable(df)
    perform_eda(df)
    X_train, X_test, y_train, y_test = split_data(df)
    model = train_model(X_train, y_train)
    y_pred = evaluate_model(model, X_test, y_test)
    visualize_results(model, df, X_test, y_test, y_pred)
    make_sample_predictions(model)
    
    print("\n" + "=" * 80)
    print("WORKFLOW SUMMARY")
    print("=" * 80)
    print("✓ Step 1: Loaded and prepared mock ultramarathon data")
    print("✓ Step 2: Encoded categorical target variable (yes/no → 1/0)")
    print("✓ Step 3: Performed exploratory data analysis with visualizations")
    print("✓ Step 4: Split data into 80% training, 20% testing")
    print("✓ Step 5: Trained logistic regression model")
    print("✓ Step 6: Evaluated model performance with multiple metrics")
    print("✓ Step 7: Visualized results and model fit")
    print("✓ Step 8: Made sample predictions for different scenarios")
    print("\nLogistic Regression implementation completed successfully! 🎉")
    
    return model, df

if __name__ == "__main__":
    # Run the complete logistic regression workflow
    trained_model, dataset = main()
    
    # Optional: Save the model and data for future use
    # import joblib
    # joblib.dump(trained_model, 'ultramarathon_logistic_model.pkl')
    # dataset.to_csv('ultramarathon_data.csv', index=False)
    # print("\nModel and data saved to files!")
