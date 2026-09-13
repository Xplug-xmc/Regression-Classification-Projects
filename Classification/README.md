# Customer Purchase Classification Using Decision Tree

## Overview

This project demonstrates a **Supervised Machine Learning Classification** workflow using a Decision Tree Classifier.

The model predicts whether a customer will make a purchase based on their age, income, and number of site visits.

## Dataset

A customer dataset was created with the following variables:

- `Age` — Customer age
- `Income` — Customer income
- `Site_Visits` — Number of visits to the site
- `Bought` — Purchase outcome

The target variable is:

- `0` = No Purchase
- `1` = Purchase

## Project Workflow

1. Created the customer dataset using Pandas
2. Inspected the dataset
3. Checked for missing values
4. Separated features and target
5. Split the data into training and testing sets
6. Built a Decision Tree Classifier
7. Trained the model
8. Generated predictions
9. Evaluated the model using Accuracy
10. Created a Confusion Matrix
11. Visualized customer purchase behavior
12. Predicted the purchase outcome for a new customer

## Model

**Algorithm:** Decision Tree Classifier

**Features:**
- Age
- Income
- Site Visits

**Target:** Bought

## Evaluation

The model was evaluated using:

- Accuracy Score
- Confusion Matrix

## Prediction

The trained model was also used to predict whether a new customer with specified age, income, and site visits would make a purchase.

## Tools Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Author

**Xplug**