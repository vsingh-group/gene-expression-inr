import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def get_matrix(donor, sort=True):
    df = pd.read_csv(f"data/abagendata/train_83_new/se_{donor}_merged.csv")

    grouped_df = df.groupby(['gene_symbol', 'se']).agg(list).reset_index()
    grouped_df = grouped_df[['gene_symbol', 'value', 'se']]
    if sort:
        grouped_df = grouped_df.sort_values(by='se', ascending=True)

    print(grouped_df.head())
    print(grouped_df.shape)

    data_for_matrix = {}

    for index, row in grouped_df.iterrows():
        data_for_matrix[row['gene_symbol']] = row['value']

    df_for_covariance = pd.DataFrame(data_for_matrix)

    # Create the correlation matrix
    correlation_matrix = df_for_covariance.corr()

    # Set diagonal values to 0
    np.fill_diagonal(correlation_matrix.values, 0)

    return correlation_matrix


def draw(unsorted_matrix_1, sorted_matrix_1, unsorted_matrix_2, sorted_matrix_2, donor_1, donor_2, name):
    # Create a 2x2 grid of heatmaps
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 20), gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.05, 'hspace': 0.2})
    
    # Set common tick label size
    tick_label_size = 10

    # Unsorted heatmap for donor 1 (top left)
    sns.heatmap(unsorted_matrix_1, 
                ax=axes[0, 0],
                annot=False,
                cmap='coolwarm',
                linewidths=.5,
                cbar=False,
                square=True)

    axes[0, 0].set_title(f'(a) Unsorted {name} Gene Matrix Heatmap Donor {donor_1}')
    axes[0, 0].xaxis.tick_top()
    axes[0, 0].set_xticklabels(axes[0, 0].get_xticklabels(), rotation=90, size=tick_label_size)
    axes[0, 0].xaxis.set_label_position('top')
    axes[0, 0].tick_params(left=False, bottom=False)  # Remove ticks
    axes[0, 0].set_ylabel('')  # Remove y-axis label

    # Sorted heatmap for donor 1 (top right)
    sns.heatmap(sorted_matrix_1, 
                ax=axes[0, 1],
                annot=False,
                cmap='coolwarm',
                linewidths=.5,
                cbar=False,
                square=True)

    axes[0, 1].set_title(f'(b) Spectrum Embedding Sorted {name} Gene Matrix Heatmap Donor {donor_1}')
    axes[0, 1].xaxis.tick_top()
    axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=90, size=tick_label_size)
    axes[0, 1].xaxis.set_label_position('top')
    axes[0, 1].tick_params(left=False, bottom=False)  # Remove ticks
    axes[0, 1].set_ylabel('')

    # Unsorted heatmap for donor 2 (bottom left)
    sns.heatmap(unsorted_matrix_2, 
                ax=axes[1, 0],
                annot=False,
                cmap='coolwarm',
                linewidths=.5,
                cbar=False,
                square=True)

    axes[1, 0].set_title(f'(c) Unsorted {name} Gene Matrix Heatmap Donor {donor_2}')
    axes[1, 0].xaxis.tick_top()
    axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=90, size=tick_label_size)
    axes[1, 0].xaxis.set_label_position('top')
    axes[1, 0].tick_params(left=False, bottom=False)  # Remove ticks
    axes[1, 0].set_ylabel('')  # Remove y-axis label

    # Sorted heatmap for donor 2 (bottom right)
    sns.heatmap(sorted_matrix_2, 
                ax=axes[1, 1],
                annot=False,
                cmap='coolwarm',
                linewidths=.5,
                cbar=False,
                square=True)

    axes[1, 1].set_title(f'(d) Spectrum Embedding Sorted {name} Gene Matrix Heatmap Donor {donor_2}')
    axes[1, 1].xaxis.tick_top()
    axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=90, size=tick_label_size)
    axes[1, 1].xaxis.set_label_position('top')
    axes[1, 1].tick_params(left=False, bottom=False)  # Remove ticks
    axes[1, 1].set_ylabel('')

    # Add a common color bar for all heatmaps below the 2x2 grid
    cbar_ax = fig.add_axes([0.14, 0.08, 0.745, 0.02])  # Reduced gap by lowering the 'y' value to 0.01
    plt.colorbar(axes[1, 1].collections[0], cax=cbar_ax, orientation='horizontal')

    plt.tight_layout(rect=[0, 0.01, 1, 1])  # Adjust layout to fit colorbar
    plt.savefig(f'./manuscript_imgs/appendix/{name}_gene_heatmap_comparison_{donor_1}_vs_{donor_2}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'./manuscript_imgs/appendix/{name}_gene_heatmap_comparison_{donor_1}_vs_{donor_2}.svg', format='svg', bbox_inches='tight')

    plt.close()


# Usage example:
unsorted_matrix_9861 = get_matrix("9861", sort=False)
sorted_matrix_9861 = get_matrix("9861", sort=True)
unsorted_matrix_10021 = get_matrix("10021", sort=False)
sorted_matrix_10021 = get_matrix("10021", sort=True)

draw(unsorted_matrix_9861, sorted_matrix_9861,
     unsorted_matrix_10021, sorted_matrix_10021, "9861", "10021", "Correlation")


