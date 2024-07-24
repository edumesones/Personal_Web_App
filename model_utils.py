# model_utils.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import os

def read_data(file_path):
    pd.read_csv(file_path)
    """Read the dataset from the given file path."""
    return pd.read_csv(file_path)

def preprocess_data(data):
    """Preprocess the data by encoding categorical variables and splitting into features and target."""
    label_encoders = {}
    categorical_columns = data.select_dtypes(include=['object']).columns
    data_copy=data.copy()
    for column in categorical_columns:
        le = LabelEncoder()
        data_copy[column] = le.fit_transform(data_copy[column])
        label_encoders[column] = le

    X = data_copy.drop(columns=['deposit'])
    y = data_copy['deposit']
    return X, y, label_encoders

def train_model(X, y):
    """Train a RandomForestClassifier model on the given features and target."""
    model = RandomForestClassifier()
    model.fit(X, y)
    return model
