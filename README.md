# Logistic Regression- Ultramarathon-completion-predictor-classifier

A comprehensive implementation of logistic regression using scikit-learn to predict whether ultramarathon runners will complete a 50-mile race based on their weekly training mileage. Features complete ML workflow from data preprocessing to model evaluation with visualizations.

## 📊 Project Overview

This project demonstrates the complete machine learning workflow for binary classification using logistic regression. The model predicts race completion outcomes based on training data, showcasing the characteristic S-curve relationship between training volume and success probability.

## 🎯 Objective

Predict whether an ultramarathon participant will complete a 50-mile race based on their average weekly training mileage using logistic regression classification.

## 📁 Dataset

The project uses mock ultramarathon data with the following features:
- **Feature**: `miles_per_week` - Average weekly training mileage
- **Target**: `completed_50_mile_ultra` - Binary outcome (yes/no → 1/0)
- **Size**: 100 participants
- **Distribution**: Realistic training ranges (15-80 miles/week) with correlated completion rates

## 🛠️ Technologies Used

- **Python 3.x**
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computations
- **scikit-learn** - Machine learning algorithms and tools
- **matplotlib** - Data visualization
- **seaborn** - Statistical data visualization

## 📋 Requirements

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

## 🚀 Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/Logistic-Regression-Ultramarathon-completion-predictor-classifier.git
cd Logistic-Regression-Ultramarathon-completion-predictor-classifier
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the main script:
```bash
python logistic_regression_ultramarathon.py
```

## 📈 Workflow

The implementation follows these key steps:

### 1. **Data Loading & Preparation**
- Generate realistic mock ultramarathon training data
- Create DataFrame with training miles and completion status
- Display dataset statistics and basic information

### 2. **Data Preprocessing**
- Convert categorical target ('yes'/'no') to numerical (1/0) using OrdinalEncoder
- Maintain proper ordering for binary classification

### 3. **Exploratory Data Analysis**
- Scatter plot visualization showing S-curve relationship
- Count plot showing distribution of completion outcomes
- Statistical insights and data quality assessment

### 4. **Data Splitting**
- Split data into 80% training and 20% testing sets
- Stratified sampling to maintain class distribution
- Ensure reproducible results with random seed

### 5. **Model Training**
- Initialize and train LogisticRegression model
- Display model coefficients and equation
- Fit the characteristic sigmoid curve to data

### 6. **Model Evaluation**
- Calculate accuracy score on test data
- Generate confusion matrix with detailed breakdown
- Comprehensive classification report with precision, recall, F1-score
- Additional metrics (sensitivity, specificity)

### 7. **Results Visualization**
- Plot logistic regression curve with training data
- Visualize prediction accuracy on test set
- Color-coded correct/incorrect predictions

### 8. **Sample Predictions**
- Demonstrate model usage with various training scenarios
- Show probability estimates and binary predictions
- Real-world application examples

## 📊 Model Performance

The logistic regression model typically achieves:
- **Accuracy**: ~90%+ on test data
- **Precision**: High precision for both classes
- **Recall**: Balanced sensitivity and specificity
- **F1-Score**: Strong overall performance metric

## 🔍 Key Insights

- **S-Curve Pattern**: Clear logistic relationship between training volume and completion probability
- **Training Effect**: Higher weekly mileage strongly correlates with race completion success
- **Realistic Outliers**: Some high-mileage runners still don't complete (injuries, weather, etc.)
- **Probability Threshold**: ~40-45 miles/week appears to be the inflection point

## 📁 File Structure

```
Logistic-Regression-Ultramarathon-completion-predictor-classifier/
│
├── logistic_regression_ultramarathon.py    # Main implementation
├── README.md                               # Project documentation
├── requirements.txt                        # Dependencies
└── results/                               # Generated plots and outputs
    ├── eda_plots.png
    ├── model_results.png
    └── confusion_matrix.png
```

## 🔧 Usage Examples

### Basic Usage
```python
# Run the complete workflow
python logistic_regression_ultramarathon.py
```

### Custom Predictions
```python
from logistic_regression_ultramarathon import main

# Train the model
model, data = main()

# Make custom predictions
miles_to_predict = [[30], [50], [70]]
probabilities = model.predict_proba(miles_to_predict)[:, 1]
predictions = model.predict(miles_to_predict)

print("Training Miles | Completion Probability | Prediction")
for i, miles in enumerate([30, 50, 70]):
    print(f"{miles:12} | {probabilities[i]:18.3f} | {'Complete' if predictions[i] else 'Incomplete'}")
```

## 📚 Mathematical Foundation

### Logistic Regression Equation
```
p = 1 / (1 + e^-(mx + b))
```

Where:
- `p` = Probability of completion
- `m` = Slope coefficient (training effect)
- `x` = Miles per week
- `b` = Intercept
- `e` = Euler's number

### Key Concepts
- **Sigmoid Function**: Maps any real number to (0,1) probability range
- **Linear Decision Boundary**: Separates classes in feature space
- **Maximum Likelihood Estimation**: Optimization method for parameter fitting

## 🎓 Learning Objectives

This project demonstrates:
- Binary classification with logistic regression
- Data preprocessing for categorical variables
- Train/test split methodology
- Model evaluation techniques
- Visualization of ML results
- Professional code structure and documentation

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, fork the repository, and create pull requests for any improvements.

## 👨‍💻 Author

**Muatasim Ahmed**
- Demonstrating machine learning concepts through practical implementation
- Focus on clear documentation and educational value

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🔗 Related Projects

- [Linear Regression Implementation])


---

⭐ **If you found this project helpful, please consider giving it a star!** ⭐
