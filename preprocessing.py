from scipy import stats
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def handle_missing_values(df):
    df.fillna(df.mean(), inplace=True)

def remove_outliers(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df = df[(np.abs(stats.zscore(df[numeric_cols])) < 3).all(axis=1)]
    return df

def encode_categorical(df, categorical_columns):
    df = pd.get_dummies(df, columns=categorical_columns)
    return df

def normalize_numeric(df):
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df
