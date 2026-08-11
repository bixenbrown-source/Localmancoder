"""
Exploratory Data Analysis Utilities

Functions for performing EDA and generating statistical summaries.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def generate_summary(df):
    """
    Generate a comprehensive summary of the DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    
    Returns:
    --------
    dict
        Dictionary containing summary statistics
    """
    summary = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
        'unique_values': df.nunique().to_dict(),
        'statistical_summary': df.describe().to_dict()
    }
    
    return summary


def plot_distribution(df, column, bins=30, figsize=(10, 6)):
    """
    Plot the distribution of a numerical column.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    column : str
        Column name to plot
    bins : int, default=30
        Number of bins for histogram
    figsize : tuple, default=(10, 6)
        Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.histplot(data=df, x=column, bins=bins, kde=True, ax=ax)
    ax.set_title(f'Distribution of {column}')
    ax.set_xlabel(column)
    ax.set_ylabel('Frequency')
    
    plt.tight_layout()
    return fig, ax


def plot_correlation_heatmap(df, figsize=(10, 8), annot=True):
    """
    Plot a correlation heatmap for numerical columns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    figsize : tuple, default=(10, 8)
        Figure size
    annot : bool, default=True
        Whether to annotate cells with correlation values
    
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    numeric_df = df.select_dtypes(include=[np.number])
    
    fig, ax = plt.subplots(figsize=figsize)
    correlation_matrix = numeric_df.corr()
    
    sns.heatmap(correlation_matrix, annot=annot, cmap='coolwarm', 
                center=0, fmt='.2f', ax=ax, linewidths=0.5)
    ax.set_title('Correlation Heatmap')
    
    plt.tight_layout()
    return fig, ax


def plot_boxplot_by_category(df, numeric_col, category_col, figsize=(10, 6)):
    """
    Create a boxplot showing distribution of a numeric column by category.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    numeric_col : str
        Numeric column name
    category_col : str
        Category column name
    figsize : tuple, default=(10, 6)
        Figure size
    
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.boxplot(data=df, x=category_col, y=numeric_col, ax=ax)
    ax.set_title(f'{numeric_col} by {category_col}')
    ax.set_xlabel(category_col)
    ax.set_ylabel(numeric_col)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    return fig, ax


def missing_values_plot(df, figsize=(10, 6)):
    """
    Plot missing values per column.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame
    figsize : tuple, default=(10, 6)
        Figure size
    
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    missing = df.isnull().sum().sort_values(ascending=False)
    missing_percentage = (missing / len(df) * 100).round(2)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.barplot(x=missing_percentage.index, y=missing_percentage.values, ax=ax)
    ax.set_title('Missing Values Percentage by Column')
    ax.set_xlabel('Column')
    ax.set_ylabel('Missing Percentage (%)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    return fig, ax
