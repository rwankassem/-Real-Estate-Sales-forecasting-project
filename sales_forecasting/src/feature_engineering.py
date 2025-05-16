# src/feature_engineering.py
from datetime import date
import pandas as pd

def feature_engineer(df):

    q1 = df['Square Footage'].quantile(0.25)
    q3 = df['Square Footage'].quantile(0.75)
    bins = [df['Square Footage'].min(), q1, q3, df['Square Footage'].max()]
    df['House Size'] = pd.cut(df['Square Footage'], bins=bins, labels=['Small', 'Medium', 'Large'])
    df['House Size'].fillna(df['House Size'].mode()[0], inplace=True)

    df['City_freq'] = df['City'].map(df['City'].value_counts())
    df['City_freq'] = df['City_freq'].fillna(0).astype(int)
    return df