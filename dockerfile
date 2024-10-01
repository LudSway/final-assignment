# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir pandas numpy scikit-learn xgboost matplotlib seaborn joblib

# Copy your Excel file into the container (update this path if needed)
COPY synthetic_pricing_data_updated.xlsx /app/

# Make port 80 available for the app
EXPOSE 80

# Define environment variable (optional)
ENV NAME World

# Run the application
CMD ["python", "business analytics capstone final assignment.py"]  # Replace lambda_function.py with the actual name of your Python file
