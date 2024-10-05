FROM python:3.8-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir pandas numpy scikit-learn xgboost matplotlib seaborn joblib
COPY synthetic_pricing_data_updated.xlsx /app/
EXPOSE 80
ENV NAME World
CMD ["python", "business analytics capstone final assignment.py"]
