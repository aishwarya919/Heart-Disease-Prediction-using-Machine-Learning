# Predictive Modeling for Heart Disease

## Overview
This project leverages machine learning techniques to predict the likelihood of heart disease in individuals based on their medical and demographic attributes. Using the heart disease dataset, the analysis involves data cleaning, visualization, feature engineering, model training, and evaluation. The project provides insights into significant predictors of heart disease and demonstrates the application of machine learning in healthcare.

## Dataset
The dataset used in this project is `heart.csv`, which includes the following attributes:
- **age**: Age of the patient.
- **sex**: Gender (1 = Male, 0 = Female).
- **cp**: Chest pain type (0–3 categories).
- **trestbps**: Resting blood pressure.
- **chol**: Serum cholesterol in mg/dl.
- **fbs**: Fasting blood sugar > 120 mg/dl (1 = True, 0 = False).
- **restecg**: Resting electrocardiographic results (0–2 categories).
- **thalach**: Maximum heart rate achieved.
- **exang**: Exercise-induced angina (1 = Yes, 0 = No).
- **oldpeak**: ST depression induced by exercise relative to rest.
- **slope**: The slope of the peak exercise ST segment (0–2 categories).
- **ca**: Number of major vessels (0–3) colored by fluoroscopy.
- **thal**: Thalassemia (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect).
- **target**: Presence of heart disease (1 = Yes, 0 = No).

## Key Steps
### 1. Data Loading and Exploration
- Loaded the dataset using Pandas.
- Explored the data structure, statistics, and checked for missing or inconsistent values.

### 2. Data Visualization
- Created plots using Matplotlib and Seaborn to visualize distributions and relationships between features.
- Identified patterns and correlations that could be useful for modeling.

### 3. Data Preprocessing
- Handled missing values and outliers.
- Encoded categorical variables where necessary.
- Scaled numerical features for better model performance.

### 4. Model Building
- Split the data into training and testing sets using `train_test_split`.
- Applied machine learning models like Logistic Regression, Decision Trees, Random Forest, and more.
- Optimized models using techniques like GridSearchCV for hyperparameter tuning.

### 5. Model Evaluation
- Evaluated models using metrics such as accuracy, precision, recall, F1 score, and ROC-AUC.
- Compared model performance and identified the best-performing approach.

### 6. Insights and Interpretations
- Highlighted the most significant features influencing heart disease predictions.
- Provided actionable insights based on model results.

## Dependencies
To run this project, the following Python libraries are required:
- **NumPy**: For numerical computations.
- **Pandas**: For data manipulation and analysis.
- **Matplotlib** and **Seaborn**: For data visualization.
- **scikit-learn**: For machine learning model development and evaluation.

Install the dependencies using:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## How to Run
1. Clone this repository or download the notebook file.
2. Ensure the dataset (`heart.csv`) is placed in the appropriate directory.
3. Open the Jupyter Notebook and execute the cells sequentially.
4. Review the outputs, visualizations, and model evaluations.

## Results
The best-performing model achieved an accuracy of 90% on the test dataset, demonstrating its ability to predict heart disease effectively. Significant predictors included features like age, chest pain type, and maximum heart rate.

## Future Work
- Explore advanced models such as Gradient Boosting or Neural Networks.
- Incorporate additional datasets for improved generalization.
- Deploy the model using Flask or Streamlit for real-time predictions.
