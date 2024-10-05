#!/usr/bin/env python
# coding: utf-8

# In[5]:


from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import joblib

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained model and preprocessor with absolute paths
model = joblib.load(r'C:\Users\dell\stacking_pricing_model.pkl')  # Update the path
preprocessor = joblib.load(r'C:\Users\dell\stacking_pricing_model.pkl')  # Update the path

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the route to get predictions and optimize the price
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data from the user input
    base_price = float(request.form['price'])
    competitor_price = float(request.form['competitor_price'])
    customer_ratings = float(request.form['customer_ratings'])
    marketing_spend = float(request.form['marketing_spend'])

    # Create input data in the format the model expects
    input_data = pd.DataFrame({
        'Price': [base_price],
        'Competitor_Price': [competitor_price],
        'Customer_Ratings': [customer_ratings],
        'Marketing_Spend': [marketing_spend],
        'Price_Marketing_Spend': [base_price * marketing_spend],
        'Price_squared': [base_price ** 2],
        'Promotion_Type': ['Discount'],
        'Product_Category': ['Category_A'],
        'Customer_Segment': ['Segment_1'],
        'Region': ['Region_1']
    })

    # Preprocess the input data
    input_data_transformed = preprocessor.transform(input_data)

    # Predict the demand using the model
    demand_prediction = model.predict(input_data_transformed)

    # Optimize the price by checking different price points
    optimal_price = optimize_price(model, base_price, competitor_price, customer_ratings, marketing_spend, preprocessor)

    # Return the result to the user
    return render_template('index.html', prediction=demand_prediction[0], optimal_price=optimal_price)

# Function to optimize price based on the model's predictions
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
        input_data_transformed = preprocessor.transform(input_data)
        demand_prediction = model.predict(input_data_transformed)
        predictions.append((price, demand_prediction[0]))

    # Calculate revenues and find the optimal price
    revenues = [(price, price * demand) for price, demand in predictions]
    return max(revenues, key=lambda x: x[1])[0]

if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:





# In[ ]:




