from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def handle_missing_values(df:pd.DataFrame, strategy='mean') -> pd.DataFrame:
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

def encode_categorical(df:pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Label encode categorical variables"""
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    return df, label_encoders

def scale_features(df:pd.DataFrame)-> Tuple[pd.DataFrame, StandardScaler]:
    """Standard scale numerical features"""
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df, scaler

def before_split_preprocess(df:pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Preprocess the dataset before splitting"""
    df, label_encoders = encode_categorical(df)
    df.drop_duplicates(inplace=True)
    return df, label_encoders

def after_split_preprocess(df:pd.DataFrame, scaler:StandardScaler) -> Tuple[pd.DataFrame, StandardScaler]:
    """Preprocess the dataset after splitting"""
    df = handle_missing_values(df)
    df,scaler = scale_features(df)
    return df, scaler

def process_test(df:pd.DataFrame, label_encoders:dict, scaler:StandardScaler) -> pd.DataFrame:
    """Process the test dataset"""
    df = handle_missing_values(df)
    for col in df.select_dtypes(include=['object']).columns:
        if col in label_encoders:
            df[col] = label_encoders[col].transform(df[col])
        else:
            raise ValueError(f"Column {col} not found in label encoders.")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    return df