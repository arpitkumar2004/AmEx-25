import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import os


# Function to read data from a folder containing CSV and Parquet files and return a dictionary of pandas dataframes
def read_data(folder_path):
    """
    Reads all CSV and parquet files in a given folder and returns a dictionary of pandas dataframes

    Parameters
    ----------
    folder_path : str
        The path of the folder to read from

    Returns
    -------
    dict
        A dictionary of pandas dataframes where the key is the filename and the value is the dataframe
    """
    files = [f for f in os.listdir(folder_path) if f.endswith('.parquet') or f.endswith('.csv')]
    dataframes = {}
    for file in files:
        file_path = os.path.join(folder_path, file)
        if file.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        dataframes[file] = df
    return dataframes


def print_feature_details(features_to_drop, data_dictionary):
    """
    Prints the details of features to be dropped.

    Parameters
    ----------
    features_to_drop : list
        List of features to be dropped
    data_dictionary : pd.DataFrame
        Dataframe containing information about the features

    Returns
    -------
    None

    """
    
    for feature in features_to_drop:
        feature_info = data_dictionary[data_dictionary['masked_column'] == feature]
        if not feature_info.empty:
            for _, row in feature_info.iterrows():
                print(f"ID: {row['masked_column']}\nDescription: {row['Description']}\nType: {row['Type']}\nMissing Values: {missing_value_count[feature]}\n{'-'*60}")
        else:
            print(f"ID: {feature} - No information available in data dictionary.\n{'-'*60}")
            


            
def impute_missing_values(df, strategy='median'):
    df_imputed = df.copy()
    for column in df_imputed.columns:
        if df_imputed[column].isnull().any():
            if df_imputed[column].dtype == 'float64':
                if strategy == 'mean':
                    df_imputed[column].fillna(df_imputed[column].mean(), inplace=True)
                elif strategy == 'median':
                    df_imputed[column].fillna(df_imputed[column].median(), inplace=True)
            # elif str(df_imputed[column].dtype) in ['category', 'object']:
            #     df_imputed[column].fillna(df_imputed[column].mode()[0], inplace=True)
    return df_imputed
            
            
missing_value_count = df.isnull().sum().sort_values( ascending=False)

missing_value_count = missing_value_count.sort_values(ascending=False)
