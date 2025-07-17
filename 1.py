import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import os

def read_data(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith('.parquet') or f.endswith('.csv')]
    dataframes = {}
    for file in files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        dataframes[file] = df
    return dataframes



missing_value_count = df.isnull().sum().sort_values( ascending=False)

missing_value_count = missing_value_count.sort_values(ascending=False)


def print_feature_details(features_to_drop, data_dictionary):
    for feature in features_to_drop:
        feature_info = data_dictionary[data_dictionary['masked_column'] == feature]
        if not feature_info.empty:
            for _, row in feature_info.iterrows():
                print(f"ID: {row['masked_column']}\nDescription: {row['Description']}\nType: {row['Type']}\nMissing Values: {missing_value_count[feature]}\n{'-'*60}")
        else:
            print(f"ID: {feature} - No information available in data dictionary.\n{'-'*60}")