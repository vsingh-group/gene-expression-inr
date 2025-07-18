import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import SpectralEmbedding  # Import SpectralEmbedding

plt.style.use('seaborn-v0_8-paper')

# Update plotting parameters
plt.rcParams.update({
    # 'font.family': 'science',
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 12,
    'legend.fontsize': 8,
    'figure.titlesize': 16,
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.5
})

# donor_list = ['9861', '10021', '12876', '14380', '15496', '15697']
donor_list = ['9861', '10021']

raw_tau = pd.read_csv('./ADNI_tau.csv')
region_meta = pd.read_csv('./atlas-desikankilliany-meta.csv')

# dfs = []
# for donor in donor_list:
#     dfs.append(pd.read_csv(f'./data/result_83_new_{donor}_inrs_avg.csv', index_col=0))
# # take average of all donors
# result_inr = pd.concat(dfs).groupby(level=0).mean()
# result_inr.to_csv('./data/83_new_interpolation_inrs.csv')
result_inr = pd.read_csv('./data/83_new_interpolation_inrs.csv', index_col=0)

# option 1, get the embedding from result data
gene_df_embedding = result_inr.T
embedding = SpectralEmbedding(n_components=1)
gene_embedding = embedding.fit_transform(gene_df_embedding)
gene_df_embedding["se"] = gene_embedding[:, 0].flatten()
gene_df_embedding = gene_df_embedding.sort_values(by="se", ascending=True)
print(gene_df_embedding.head())
result_inr = gene_df_embedding.drop('se', axis=1).T

# # option 2, use original embedding
# se_9861 = pd.read_csv('./data/abagendata/train_83_new/se_9861.csv')
# se_10021 = pd.read_csv('./data/abagendata/train_83_new/se_10021.csv')
# missing_index = set(se_9861.index) - set(result_inr.index)
# print(result_inr.index)
# print(se_9861.index)
# print(se_10021)

# result_inr = result_inr.reindex(index=se_9861.index)

# make abagen have same label as inr
result_abagen = pd.read_csv('./data/83_new_interpolation_abagen.csv')
result_abagen.index = result_inr.index
result_abagen = result_abagen[result_inr.columns.to_list()]
# take only columns that are in both datasets

# take only AD in raw_tau
raw_tau = raw_tau[raw_tau['merge_DX'].isin(['Dementia', 'MCI'])]
raw_tau = raw_tau[raw_tau['best_DX'].isin(['LMCI', 'AD'])]
raw_tau = raw_tau.iloc[:, 3:] # drop non value columns

raw_tau['D'] = (raw_tau['Left-Cerebellum-Cortex'] + raw_tau['Right-Cerebellum-Cortex']) / 2
new_tau = raw_tau.div(raw_tau['D'], axis=0)
new_tau = new_tau.sub(1, axis=0)
# remove D
new_tau = new_tau.drop(columns=['D'])

tau1 = region_meta['tau_id'].dropna().to_list()
tau2 = new_tau.columns.to_list()
print(set(tau2) - set(tau1))

# pick a gene that is known to have similar expression to a disease pattern.
# e.g. APOE, BIN1, MAPT and APP in Alzheimer disease.
# Then show the regional correlation between each such gene and the regional
# pattern of atrophy and/or tau of AD subjects.
# We have the latter and can share easily.

new_tau = new_tau.mean(axis=0).to_frame() # take mean for regions
new_tau.reset_index(level=0, inplace=True) # set index to column
new_tau.columns = ['tau_id', 'tau'] # rename columns
res = pd.merge(region_meta, new_tau, on='tau_id', how='inner')
res = res[['id', 'tau']]
res.index = res['id']
res = res.drop(columns=['id'])

print(res.head())
print(result_abagen.head())
print(res.shape, result_abagen.shape, result_inr.shape)

def get_corr(result):
    #merge on index
    merged_data = pd.merge(res, result, left_index=True, right_index=True)
    print(merged_data.shape)
    gene_columns = result.columns
    tau_values = merged_data['tau']

    # Calculate correlation between each gene and tau values.
    correlations = {}
    for gene in gene_columns:
        correlations[gene] = merged_data[gene].corr(tau_values)
        
    correlation_df = pd.DataFrame(list(correlations.items()),
                                  columns=['Gene', 'Correlation'])
    return correlation_df

correlation_abagen = get_corr(result_abagen)
correlation_inr = get_corr(result_inr)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Merge the correlation dataframes on the 'Gene' column
merged_correlations = pd.merge(correlation_abagen, correlation_inr, on='Gene', suffixes=(' Abagen', ' INR'))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 16))

# Plot 1: Bar plot
melted_correlations = pd.melt(merged_correlations, id_vars=['Gene'], 
                              value_vars=['Correlation Abagen', 'Correlation INR'],
                              var_name='Method', value_name='Correlation')

palette = sns.color_palette(['#4c72b0', '#55a868'])  # Blue and green shades


sns.barplot(x='Gene', y='Correlation', hue='Method', data=melted_correlations, palette=palette, ax=ax1)

ax1.set_xlabel('Genes (Ordered by Spectral Embedding)')
ax1.set_ylabel('Correlation with Tau')
ax1.set_title('Comparison of Gene Correlations with Tau between Abagen and INR Methods')
ax1.tick_params(axis='x', rotation=90)
ax1.legend(title='Method', loc='upper right')
ax1.grid(axis='y', linestyle='--', alpha=0.7)


# Plot 2: Scatter plot
# R and p values on the scatterplot
from scipy.stats import pearsonr

r, p = pearsonr(merged_correlations['Correlation Abagen'], merged_correlations['Correlation INR'])

sns.scatterplot(x='Correlation Abagen', y='Correlation INR', data=merged_correlations, s=100, color=palette[0], ax=ax2)

ax2.set_xlabel('Correlation with Tau (Abagen)')
ax2.set_ylabel('Correlation with Tau (INR)')
ax2.set_title('Scatterplot of Gene Correlations with Tau between Abagen and INR Methods')

ax2.grid(axis='both', linestyle='--', alpha=0.7)

ax2.annotate(f'R = {r:.2f}, p = {p:.2e}', xy=(0.5, 0.9), xycoords='axes fraction', ha='center')

z = np.polyfit(merged_correlations['Correlation Abagen'], merged_correlations['Correlation INR'], 1)
p = np.poly1d(z)
xfit = np.linspace(merged_correlations['Correlation Abagen'].min(), merged_correlations['Correlation Abagen'].max(), 50)
yfit = p(xfit)
ax2.plot(xfit, yfit, "r--")

plt.tight_layout()
# plt.savefig("./manuscript_imgs/combined_tau_plots.png", bbox_inches='tight', dpi=300)
plt.savefig("./manuscript_imgs/combined_tau_plots.pdf", bbox_inches='tight', format='pdf')

plt.show()