import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import SpectralEmbedding
from scipy.stats import pearsonr

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

# Load data
genecard = pd.read_csv('GeneCards-SearchResults.csv')
raw_tau = pd.read_csv('./ADNI_tau.csv')
region_meta = pd.read_csv('./atlas-desikankilliany-meta.csv')
result_inr = pd.read_csv('./data/83_new_interpolation_inrs.csv', index_col=0)

# Process the GeneCards data to extract gene symbols and relevance scores
gene_to_score = genecard[['Gene Symbol', 'Relevance score']].copy()
gene_to_score.columns = ['Gene', 'relevanceScore']
# Convert relevance score to numeric if needed
gene_to_score['relevanceScore'] = pd.to_numeric(gene_to_score['relevanceScore'], errors='coerce')
# Drop rows with missing values
gene_to_score = gene_to_score.dropna()

# Option 1, get the embedding from result data
gene_df_embedding = result_inr.T
embedding = SpectralEmbedding(n_components=1)
gene_embedding = embedding.fit_transform(gene_df_embedding)
gene_df_embedding["se"] = gene_embedding[:, 0].flatten()
gene_df_embedding = gene_df_embedding.sort_values(by="se", ascending=True)
print(gene_df_embedding.head())
result_inr = gene_df_embedding.drop('se', axis=1).T

# Make abagen have same label as inr
result_abagen = pd.read_csv('./data/83_new_interpolation_abagen.csv')
result_abagen.index = result_inr.index
result_abagen = result_abagen[result_inr.columns.to_list()]

# Process tau data
raw_tau = raw_tau[raw_tau['merge_DX'].isin(['Dementia', 'MCI'])]
raw_tau = raw_tau[raw_tau['best_DX'].isin(['LMCI', 'AD'])]
raw_tau = raw_tau.iloc[:, 3:]  # drop non value columns

raw_tau['D'] = (raw_tau['Left-Cerebellum-Cortex'] + raw_tau['Right-Cerebellum-Cortex']) / 2
new_tau = raw_tau.div(raw_tau['D'], axis=0)
new_tau = new_tau.sub(1, axis=0)
# Remove D
new_tau = new_tau.drop(columns=['D'])

tau1 = region_meta['tau_id'].dropna().to_list()
tau2 = new_tau.columns.to_list()
print(set(tau2) - set(tau1))

new_tau = new_tau.mean(axis=0).to_frame()  # take mean for regions
new_tau.reset_index(level=0, inplace=True)  # set index to column
new_tau.columns = ['tau_id', 'tau']  # rename columns
res = pd.merge(region_meta, new_tau, on='tau_id', how='inner')
res = res[['id', 'tau']]
res.index = res['id']
res = res.drop(columns=['id'])

print(res.head())
print(result_abagen.head())
print(res.shape, result_abagen.shape, result_inr.shape)

def get_corr(result):
    # Merge on index
    merged_data = pd.merge(res, result, left_index=True, right_index=True)
    print(merged_data.shape)
    gene_columns = result.columns
    tau_values = merged_data['tau']

    # Calculate correlation between each gene and tau values.
    correlations = {}
    for gene in gene_columns:
        correlations[gene] = merged_data[gene].corr(tau_values)
        # get the absolute value of correlation
        correlations[gene] = abs(correlations[gene])
        
    correlation_df = pd.DataFrame(list(correlations.items()),
                                  columns=['Gene', 'Correlation'])
    return correlation_df

# Get correlations
correlation_abagen = get_corr(result_abagen)
correlation_inr = get_corr(result_inr)

# Merge the correlation data with GeneCards data
# Join correlation data with GeneCards relevance scores
abagen_with_relevance = pd.merge(correlation_abagen, gene_to_score, on='Gene', how='inner')
inr_with_relevance = pd.merge(correlation_inr, gene_to_score, on='Gene', how='inner')

# Print the number of genes that were successfully matched
print(f"Genes matched with GeneCards data: Abagen={len(abagen_with_relevance)}, INR={len(inr_with_relevance)}")

# Create the plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# Plot 1: Abagen correlation vs GeneCards relevance score
corr_abagen, p_abagen = pearsonr(abagen_with_relevance['Correlation'], abagen_with_relevance['relevanceScore'])
sns.scatterplot(x='Correlation', y='relevanceScore', data=abagen_with_relevance, 
                s=100, color='#4c72b0', ax=ax1)

ax1.set_xlabel('Gene Correlation with Tau (Abagen)')
ax1.set_ylabel('GeneCards Relevance Score')
ax1.set_title('Abagen Gene-Tau Correlation vs GeneCards Relevance Score')
ax1.grid(axis='both', linestyle='--', alpha=0.7)
ax1.annotate(f'R = {corr_abagen:.2f}, p = {p_abagen:.2e}', 
             xy=(0.5, 0.9), xycoords='axes fraction', ha='center')

# Add trend line
if len(abagen_with_relevance) > 1:  # Ensure we have enough data points
    z = np.polyfit(abagen_with_relevance['Correlation'], abagen_with_relevance['relevanceScore'], 1)
    p = np.poly1d(z)
    xfit = np.linspace(abagen_with_relevance['Correlation'].min(), abagen_with_relevance['Correlation'].max(), 50)
    yfit = p(xfit)
    ax1.plot(xfit, yfit, "r--")

# Plot 2: INR correlation vs GeneCards relevance score
corr_inr, p_inr = pearsonr(inr_with_relevance['Correlation'], inr_with_relevance['relevanceScore'])
sns.scatterplot(x='Correlation', y='relevanceScore', data=inr_with_relevance, 
                s=100, color='#55a868', ax=ax2)

ax2.set_xlabel('Gene Correlation with Tau (INR)')
ax2.set_ylabel('GeneCards Relevance Score')
ax2.set_title('INR Gene-Tau Correlation vs GeneCards Relevance Score')
ax2.grid(axis='both', linestyle='--', alpha=0.7)
ax2.annotate(f'R = {corr_inr:.2f}, p = {p_inr:.2e}', 
             xy=(0.5, 0.9), xycoords='axes fraction', ha='center')

# Add trend line
if len(inr_with_relevance) > 1:  # Ensure we have enough data points
    z = np.polyfit(inr_with_relevance['Correlation'], inr_with_relevance['relevanceScore'], 1)
    p = np.poly1d(z)
    xfit = np.linspace(inr_with_relevance['Correlation'].min(), inr_with_relevance['Correlation'].max(), 50)
    yfit = p(xfit)
    ax2.plot(xfit, yfit, "r--")

plt.tight_layout()
plt.savefig("./manuscript_imgs/genecard_correlation_plots.pdf", bbox_inches='tight', format='pdf')
plt.savefig("./manuscript_imgs/genecard_correlation_plots.png", bbox_inches='tight', dpi=300)
plt.show()