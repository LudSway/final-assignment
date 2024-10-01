#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, RegressorMixin
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load the synthetic dataset
df = pd.read_excel(r'C:\Users\dell\Downloads\synthetic_pricing_data_updated.xlsx')

# Function to check and handle missing values
def handle_missing_values(df):
    df.fillna(df.median(numeric_only=True), inplace=True)
    for col in df.select_dtypes(include='object').columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
    return df

# Feature Engineering: Add Interaction Terms and Polynomial Features
def feature_engineering(df):
    df['Price_Marketing_Spend'] = df['Price'] * df['Marketing_Spend']  # Interaction term
    df['Price_squared'] = df['Price'] ** 2  # Polynomial term
    return df

# Preprocess data and split features/target
def preprocess_data(df, target='Demand_Units_Sold'):
    df = handle_missing_values(df)
    df = feature_engineering(df)
    
    X = df[['Price', 'Competitor_Price', 'Customer_Ratings', 'Marketing_Spend',
            'Price_Marketing_Spend', 'Price_squared', 'Promotion_Type', 
            'Product_Category', 'Customer_Segment', 'Region']]
    y = df[target]
    
    return X, y

# Plot correlation heatmap
def plot_corr_heatmap(df):
    numeric_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()

# Build preprocessing pipeline
def build_preprocessor():
    categorical_features = ['Promotion_Type', 'Product_Category', 'Customer_Segment', 'Region']
    numerical_features = ['Price', 'Competitor_Price', 'Customer_Ratings', 'Marketing_Spend', 
                          'Price_Marketing_Spend', 'Price_squared']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )
    
    return preprocessor

# Evaluate model performance
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"{model_name} R²: {r2:.4f}")
    print(f"{model_name} RMSE: {rmse:.4f}")
    return r2, rmse

# Plot actual vs predicted values
def plot_actual_vs_predicted(y_test, y_pred, model_name):
    plt.scatter(y_test, y_pred)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Actual vs Predicted - {model_name}')
    plt.show()

# Hyperparameter tuning using RandomizedSearchCV
def hyperparameter_tuning(model, param_grid, X_train, y_train, model_name):
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"Best parameters for {model_name}: {grid_search.best_params_}")
    return grid_search.best_estimator_

# Optimize price based on the model's predictions
def optimize_price(model, base_price, competitor_price, customer_ratings, marketing_spend, preprocessor):
    prices = np.linspace(base_price * 0.8, base_price * 1.2, 100)
    predictions = []

    for price in prices:
        input_data = pd.DataFrame({
            'Price': [price],
            'Competitor_Price': [competitor_price],
            'Customer_Ratings': [customer_ratings],
            'Marketing_Spend': [marketing_spend],
            'Price_Marketing_Spend': [price * marketing_spend],
            'Price_squared': [price ** 2],
            'Promotion_Type': ['Discount'],
            'Product_Category': ['Category_A'],
            'Customer_Segment': ['Segment_1'],
            'Region': ['Region_1']
        })
        input_data = preprocessor.transform(input_data)
        demand_prediction = model.predict(input_data)
        predictions.append((price, demand_prediction[0]))

    # Calculate revenues and find the optimal price
    revenues = [(price, price * demand) for price, demand in predictions]
    return max(revenues, key=lambda x: x[1])[0]

# Main training and evaluation pipeline
def main():
    # Preprocess data
    X, y = preprocess_data(df)
    preprocessor = build_preprocessor()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    # ElasticNet model with hyperparameter tuning
    elastic_net = ElasticNet(random_state=42)
    param_grid_enet = {'alpha': [0.1, 1.0, 10], 'l1_ratio': [0.2, 0.5, 0.8]}
    best_enet = hyperparameter_tuning(elastic_net, param_grid_enet, X_train, y_train, 'ElasticNet')

    # Random Forest model with hyperparameter tuning
    rf = RandomForestRegressor(random_state=42)
    param_grid_rf = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]}
    best_rf = hyperparameter_tuning(rf, param_grid_rf, X_train, y_train, 'Random Forest')

    # XGBoost model
    xgboost_model = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1,
                                     max_depth=5, alpha=10, n_estimators=10)
    xgboost_model.fit(X_train, y_train)

    # Stacking Model: ElasticNet + RandomForest + XGBoost
    stack_model = StackingRegressor(
        estimators=[('elastic_net', best_enet), ('random_forest', best_rf), ('xgboost', xgboost_model)],
        final_estimator=ElasticNet()
    )
    stack_model.fit(X_train, y_train)

    # Evaluate models
    evaluate_model(best_enet, X_test, y_test, "ElasticNet")
    evaluate_model(best_rf, X_test, y_test, "Random Forest")
    evaluate_model(xgboost_model, X_test, y_test, "XGBoost")
    evaluate_model(stack_model, X_test, y_test, "Stacking Model")

    # Plot Actual vs Predicted for Stacking Model
    y_pred_stack = stack_model.predict(X_test)
    plot_actual_vs_predicted(y_test, y_pred_stack, "Stacking Model")

    # Optimize price using Stacking Model
    optimal_price = optimize_price(stack_model, base_price=150, competitor_price=140, customer_ratings=4.2, 
                                   marketing_spend=5000, preprocessor=preprocessor)
    print(f"Optimal Price: {optimal_price:.2f}")

    # Save the stacking model
    joblib.dump(stack_model, 'stacking_pricing_model.pkl')

# Run the main pipeline
if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:




