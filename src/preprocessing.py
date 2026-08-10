"""
Data Preprocessing Utilities

Functions for cleaning and preparing data for analysis.
"""

import pandas as pd
import numpy as np


def handle_missing_values(df, strategy='mean', columns=None):
    """
    Handle missing values in a DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame with missing values
    strategy : str, default='mean'
        Strategy for handling missing values: 'mean', 'median', 'mode', or 'drop'
    columns : list, optional
        Specific columns to apply the strategy to. If None, applies to all numeric columns.
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with missing values handled
    """
    df_clean = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    if strategy == 'mean':
        df_clean[columns] = df_clean[columns].fillna(df_clean[columns].mean())
    elif strategy == 'median':
        df_clean[columns] = df_clean[columns].fillna(df_clean[columns].median())
    elif strategy == 'mode':
        df_clean[columns] = df_clean[columns].fillna(df_clean[columns].mode().iloc[0])
    elif strategy == 'drop':
        df_clean = df_clean.dropna(subset=columns)
    
    return df_clean


def remove_outliers_iqr(df, column, multiplier=1.5):
    """
    Remove outliers using the IQR method.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    column : str
        Column name to check for outliers
    multiplier : float, default=1.5
        IQR multiplier for outlier detection
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with outliers removed
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    df_clean = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return df_clean


def normalize_data(df, columns=None, method='minmax'):
    """
    Normalize numerical columns in a DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    columns : list, optional
        Columns to normalize. If None, normalizes all numeric columns.
    method : str, default='minmax'
        Normalization method: 'minmax' or 'zscore'
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with normalized columns
    """
    df_normalized = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if method == 'minmax':
            min_val = df_normalized[col].min()
            max_val = df_normalized[col].max()
            df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
        elif method == 'zscore':
            mean_val = df_normalized[col].mean()
            std_val = df_normalized[col].std()
            df_normalized[col] = (df_normalized[col] - mean_val) / std_val
    
    return df_normalized
