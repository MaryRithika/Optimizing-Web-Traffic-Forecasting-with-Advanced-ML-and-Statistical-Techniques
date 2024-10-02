import pandas as pd
import random

# Load the dataset
file_path = 'C:/Users/manas/OneDrive/Desktop/GMU/Fall 2023 subs/584-Data mining/Project/584- DATAMINING-FINAL PROJECT/CODE/CoromonDataset.csv'
coromon_data = pd.read_csv(file_path)

# Organize the data into a data structure
# Group by type and calculate the average for each property
grouped_data = coromon_data.groupby('Type').mean()

# Tasks to perform:
# 1. Count the total number of Coromons
total_coromons = coromon_data.shape[0]

# 2. Select a random Coromon and display its information
random_coromon = coromon_data.sample(1).iloc[0]

# 3. Display the different types of Coromons
coromon_types = coromon_data['Type'].unique()

# 4. For each Common type, display the average value for each of its properties
common_type_avg = grouped_data.loc[grouped_data.index == 'Common']

# 5. Find Coromon type(s) with the highest and lowest average points for each property
highest_lowest_avg = {
    'Highest Average Health Points': grouped_data['Health Points'].idxmax(),
    'Lowest Average Health Points': grouped_data['Health Points'].idxmin(),
    'Highest Average Attack Points': grouped_data['Attack'].idxmax(),
    'Lowest Average Attack Points': grouped_data['Attack'].idxmin(),
    'Highest Average Special Attack Points': grouped_data['Special Attack'].idxmax(),
    'Lowest Average Special Attack Points': grouped_data['Special Attack'].idxmin(),
    'Highest Average Defense Points': grouped_data['Defense'].idxmax(),
    'Lowest Average Defense Points': grouped_data['Defense'].idxmin(),
    'Highest Average Special Defense Points': grouped_data['Special Defense'].idxmax(),
    'Lowest Average Special Defense Points': grouped_data['Special Defense'].idxmin(),
    'Highest Average Speed Points': grouped_data['Speed'].idxmax(),
    'Lowest Average Speed Points': grouped_data['Speed'].idxmin()
}

total_coromons, random_coromon, coromon_types, common_type_avg, highest_lowest_avg

