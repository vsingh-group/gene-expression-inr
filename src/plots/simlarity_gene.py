import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

donor = '10021'
df = pd.read_csv(f"data/abagendata/train_83_new/se_{donor}_merged.csv")

grouped_df = df.groupby(['gene_symbol', 'se']).agg(list).reset_index()
grouped_df = grouped_df[['gene_symbol', 'value', 'se']]
grouped_df = grouped_df.sort_values(by='se', ascending=True)

print(grouped_df.head())
print(grouped_df.shape)

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

    plt.title(f'{name} Gene Matrix Heatmap, Donor {donor}')

    ax.xaxis.tick_top()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.xaxis.set_label_position('top')

    plt.savefig(f'{name}_gene_heatmap_{donor}.png', dpi=300, bbox_inches='tight')

draw(df_for_covariance.corr(), "Correlation")
# draw(df_for_covariance.cov(), "Covariance")

