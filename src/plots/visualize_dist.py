# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

order = "se"
donor = 9861

filepath=f'../../data/abagendata/train/{order}_{donor}_merged.csv'
meta_df = pd.read_csv(filepath)

# Set the aesthetic style of the plots
sns.set(style="whitegrid")

# Plot histograms for continuous data
for column in ['mni_x', 'mni_y', 'mni_z', 'value', 'se']:
    plt.figure(figsize=(10, 6))
    sns.histplot(meta_df[column], kde=False)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

# Plot bar plots for categorical data
for column in ['gene_symbol', 'well_id', 'classification']:
    plt.figure(figsize=(10, 6))
    if meta_df[column].nunique() <= 30:  # Only plot if there aren't too many unique values
        sns.countplot(y=column, data=meta_df)
        plt.title(f'Count of {column}')
        plt.xlabel('Count')
        plt.ylabel(column)
        plt.show()
    else:
        print(f"{column} has too many unique values to display effectively as a bar plot.")

# %%
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

# Standard scaling
scaler = StandardScaler()
meta_df['se_norm'] = scaler.fit_transform(meta_df[['se']])

min_max_scaler = MinMaxScaler()
meta_df['se_min_max'] = min_max_scaler.fit_transform(meta_df[['se']])


meta_df['log_normalized_value'] = np.log(meta_df['se'] + 1 - meta_df['se'].min())

# Plot the normalized data
sns.histplot(meta_df['se_norm'], kde=True)
plt.xlabel('Normalized Value')
plt.ylabel('Frequency')
plt.show()


# %%
