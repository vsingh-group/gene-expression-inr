import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

# Load data
donor_list = ['9861', '10021']

raw_tau = pd.read_csv('./ADNI_tau.csv')
region_meta = pd.read_csv('./atlas-desikankilliany-meta.csv')

result_inr = pd.read_csv('./data/83_new_interpolation_inrs.csv', index_col=0)
result_abagen = pd.read_csv('./data/83_new_interpolation_abagen.csv')
result_abagen.index = result_inr.index
result_abagen = result_abagen[result_inr.columns.to_list()]

# Filter and preprocess raw_tau
raw_tau = raw_tau[raw_tau['merge_DX'].isin(['Dementia', 'MCI'])]
raw_tau = raw_tau[raw_tau['best_DX'].isin(['LMCI', 'AD'])]
raw_tau = raw_tau.iloc[:, 3:]

raw_tau['D'] = (raw_tau['Left-Cerebellum-Cortex'] + raw_tau['Right-Cerebellum-Cortex']) / 2
new_tau = raw_tau.div(raw_tau['D'], axis=0).sub(1, axis=0)
new_tau = new_tau.drop(columns=['D'])

# Merge tau data with region_meta
new_tau = new_tau.mean(axis=0).to_frame().reset_index(level=0)
new_tau.columns = ['tau_id', 'tau']
res = pd.merge(region_meta, new_tau, on='tau_id', how='inner')
res = res[['id', 'tau']]
res.index = res['id']
res = res.drop(columns=['id'])

# Ensure result_inr and result_abagen have the same set of regions
common_regions = res.index.intersection(result_inr.index)
result_inr = result_inr.loc[common_regions]
result_abagen = result_abagen.loc[common_regions]

result_inr['method'] = 'INR'
result_abagen['method'] = 'ABAGEN'

# Melt dataframes to long format
result_inr_long = result_inr.reset_index().melt(id_vars=['index', 'method'], var_name='gene', value_name='expression')
result_abagen_long = result_abagen.reset_index().melt(id_vars=['index', 'method'], var_name='gene', value_name='expression')

# Concatenate the dataframes
combined_df = pd.concat([result_inr_long, result_abagen_long])

# Rename the columns for clarity
combined_df.columns = ['region', 'method', 'gene', 'expression']

# Get unique genes
unique_genes = combined_df['gene'].unique()

# Number of rows needed
num_rows = math.ceil(len(unique_genes) / 10)

# Set the width and height of each subplot
height = 1.5
width = 7 * height

# Create the subplots
fig, axes = plt.subplots(num_rows, 1, figsize=(width, height * num_rows), sharey=True)

# Make sure axes is an array even if there's only one subplot
if num_rows == 1:
    axes = [axes]

# Create the violin plots
for i in range(num_rows):
    subset_genes = unique_genes[i*10:(i+1)*10]
    subset_df = combined_df[combined_df['gene'].isin(subset_genes)]
    sns.violinplot(data=subset_df, x="gene", y="expression", hue="method",
                   split=True, inner="quart", fill=True,
                   palette=sns.color_palette(['#55a868', '#4c72b0']),
                   ax=axes[i])
    # make legend smaller
    if i == 0:
        axes[i].legend(loc='upper right', fontsize='small')
    else:
        axes[i].get_legend().remove()

    if i < num_rows - 1:
        axes[i].set_xlabel('')


plt.tight_layout()

plt.savefig("tau_plot_2.png", bbox_inches='tight', dpi=300)
plt.show()
