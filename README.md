# Diamond Price Prediction using Machine Learning

Statistical modeling project comparing multiple machine learning approaches to predict diamond prices based on physical characteristics and quality grades.

## Project Overview
This project explores various regression and classification models to accurately predict diamond prices in the luxury goods market. Using a dataset of 53,940 diamonds with 11 variables, we compared traditional linear methods against advanced ensemble techniques to find the optimal prediction model.

## Dataset
- **Size:** 53,940 diamonds
- **Split:** 80% training, 20% testing
- **Key Features:**
  - **Carat:** Diamond weight (1 carat = 200 milligrams)
  - **Cut:** Quality grade (Fair, Good, Very Good, Premium, Ideal)
  - **Color:** Color grade (D-J scale, D is best)
  - **Clarity:** Clarity grade (I1, SI1, SI2, VS1, VS2, VVS1, VVS2, IF)
  - **Dimensions:** x, y, z (length, width, depth in millimeters)
  - **Table:** Width of top facet relative to widest point
  - **Depth:** Total depth percentage

## Technologies Used
- **Language:** R
- **Libraries:** tidyverse, caret, randomForest, xgboost, rpart, bestglm, pROC
- **IDE:** RStudio

## Models Implemented

### Regression Models
1. **Linear Regression** - Baseline model with log transformation
2. **Decision Tree (CART)** - Rule-based regression approach
3. **Random Forest** - Ensemble of 100 decision trees
4. **XGBoost** - Gradient boosting with 100 rounds
5. **Best Subset Selection** - Feature selection using AIC/BIC
6. **Stepwise Selection** - Forward/backward variable selection

### Classification Models
Binary classification (high vs. low price based on median) using:
- Random Forest, Decision Tree, XGBoost
- Logistic regression with best subset selection
- ROC curve analysis for model comparison

## Results

| Model | Test RMSE |
|-------|-----------|
| Linear Regression | 0.1670 |
| Decision Tree | 0.3080 |
| Random Forest | 0.0923 |
| **XGBoost** | **0.0893** |

**Best Model:** XGBoost achieved the lowest RMSE (0.0893) with an R² of 0.992

### Feature Importance
Top predictors identified by XGBoost:
1. y (width)
2. carat
3. clarity
4. x (length)

## Key Findings
- XGBoost outperformed all other models for diamond price prediction
- Carat and dimensions (x, y) were the most influential predictors
- Log transformation improved linear model performance
- Ensemble methods significantly outperformed single decision trees

## Applications
This model can be applied to:
- Online diamond marketplaces for automated pricing
- Valuation tools for jewelry retailers
- Investment analysis in the luxury goods market
- Price verification and fraud detection

## Course Information
**Institution:** Carnegie Mellon University  
**Project Type:** Final Project
