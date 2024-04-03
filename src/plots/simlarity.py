import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("./data/se_merged.csv")

grouped_df = df.groupby(['gene_symbol', 'se']).agg(list).reset_index()
grouped_df = grouped_df[['gene_symbol', 'value', 'se']]
grouped_df = grouped_df.sort_values(by='se', ascending=True)

data_for_matrix = {}

for index, row in grouped_df.iterrows():
    data_for_matrix[row['gene_symbol']] = row['value']

df_for_covariance = pd.DataFrame(data_for_matrix)

def draw(covariance_matrix, name):
    plt.figure(figsize=(10, 8))  # You can adjust the figure size as needed
    ax = sns.heatmap(covariance_matrix, 
                annot=False,     # Set to True if you want to see the numbers
                cmap='coolwarm', # Color map
                linewidths=.5)   # Line widths between cells

    plt.title(f'{name} Matrix Heatmap')

    ax.xaxis.tick_top()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.xaxis.set_label_position('top')

    plt.savefig(f'{name}_matrix_heatmap.png', dpi=300, bbox_inches='tight')

draw(df_for_covariance.corr(), "Correlation")
draw(df_for_covariance.cov(), "Covariance")

