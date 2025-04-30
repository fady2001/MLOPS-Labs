import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def handle_missing_values(df:pd.DataFrame, strategy='mean'):
    """Fill missing values for numeric and categorical columns"""
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            if strategy == 'mean':
                df[col].fillna(df[col].mean(), inplace=True)
            elif strategy == 'median':
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(0, inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
    return df

def encode_categorical(df:pd.DataFrame):
    """Label encode categorical variables"""
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    return df, label_encoders

def scale_features(df:pd.DataFrame):
    """Standard scale numerical features"""
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df, scaler

def preprocessor(df:pd.DataFrame,output_path='processed.csv'):
    """Main preprocessing pipeline"""
    df = handle_missing_values(df)
    df, encoders = encode_categorical(df)
    df, scaler = scale_features(df)
    return df, encoders, scaler