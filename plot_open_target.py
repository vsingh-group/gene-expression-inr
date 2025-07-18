import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import SpectralEmbedding  # Import SpectralEmbedding
from scipy.stats import pearsonr

plt.style.use('seaborn-v0_8-paper')

# Update plotting parameters
plt.rcParams.update({
    # 'font.family': 'science',
    'font.size': 10,
    'axes.labelsize': 20,
    'axes.titlesize': 20,
    'xtick.labelsize': 10,
    'ytick.labelsize': 12,
    'legend.fontsize': 10,
    'figure.titlesize': 20,
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.5
})

# donor_list = ['9861', '10021', '12876', '14380', '15496', '15697']
donor_list = ['9861', '10021']

# Load data
open_target_association = pd.read_csv('OT-MONDO_0004975-associated-targets-2_26_2025-v24_09.tsv', sep='\t')
raw_tau = pd.read_csv('./ADNI_tau.csv')
region_meta = pd.read_csv('./atlas-desikankilliany-meta.csv')

# Process the Open Target data to extract gene symbols and globalScore
# Assuming that 'targetSymbol' (or some similar field) and 'globalScore' columns exist
gene_to_score = open_target_association[['symbol', 'globalScore']].copy()
gene_to_score.columns = ['Gene', 'globalScore']
# Convert globalScore to numeric if needed
gene_to_score['globalScore'] = pd.to_numeric(gene_to_score['globalScore'], errors='coerce')
# Drop rows with missing values
gene_to_score = gene_to_score.dropna()

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
# Then show the regional correlation between each gene and the regional
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

# Get correlations (original way - without absolute value)
correlation_abagen = get_corr(result_abagen)
correlation_inr = get_corr(result_inr)

# Also get absolute correlations for the open target comparison
def get_abs_corr(result):
    # Merge on index
    merged_data = pd.merge(res, result, left_index=True, right_index=True)
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

# Get absolute correlations for open target plots
abs_correlation_abagen = get_abs_corr(result_abagen)
abs_correlation_inr = get_abs_corr(result_inr)

# Merge the absolute correlation data with Open Target data
abagen_with_global = pd.merge(abs_correlation_abagen, gene_to_score, on='Gene', how='inner')
inr_with_global = pd.merge(abs_correlation_inr, gene_to_score, on='Gene', how='inner')

# Print the number of genes that were successfully matched
print(f"Genes matched with Open Target data: Abagen={len(abagen_with_global)}, INR={len(inr_with_global)}")

# Create a figure with 4 subplots (3 rows layout as requested)
fig = plt.figure(figsize=(20, 24))

# Define the grid layout for the 4 plots
ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)  # Original bar plot (takes full width of row 1)
ax2 = plt.subplot2grid((3, 2), (1, 0), colspan=2)  # Original scatter plot (takes full width of row 2)
ax3 = plt.subplot2grid((3, 2), (2, 0))  # Open target plot for Abagen
ax4 = plt.subplot2grid((3, 2), (2, 1))  # Open target plot for INR

# Plot 1: Original Bar plot from plot_tau_label.py
# Merge the correlation dataframes on the 'Gene' column
merged_correlations = pd.merge(correlation_abagen, correlation_inr, on='Gene', suffixes=(' Abagen', ' INR'))

# Bar plot
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

# Add bold label 'a' to the left of the first subplot
ax1.text(-0.05, 0.98, 'a', transform=ax1.transAxes, fontsize=24, fontweight='bold', va='top', ha='left')

red_genes = ['NEAT1', 'JAZF1', 'HSPH1']
purple_genes = ['TMEM41A', 'PTK2B', 'ZNF184']

unique_genes = melted_correlations['Gene'].unique()
bar_positions = range(len(unique_genes))

# Add light blue background for special genes and color their tick labels
for i, gene in enumerate(unique_genes):
    if gene in red_genes or gene in purple_genes:
        # Add light grey background spanning the width of both bars for this gene
        rect_start = i - 0.4  # Adjust based on bar width
        rect_width = 0.8     # Adjust based on the spacing between groups
        rect = plt.Rectangle((rect_start, ax1.get_ylim()[0]), rect_width, ax1.get_ylim()[1] - ax1.get_ylim()[0],
                         facecolor='grey', alpha=0.3, zorder=-1)
        ax1.add_patch(rect)

for tick in ax1.get_xticklabels():
    gene_name = tick.get_text()
    if gene_name in red_genes:
        tick.set_color('red')
        tick.set_fontweight('bold')
    elif gene_name in purple_genes:
        tick.set_color('purple')
        tick.set_fontweight('bold')

# Plot 2: Original Scatter plot from plot_tau_label.py
# Calculate R and p values
r, p = pearsonr(merged_correlations['Correlation Abagen'], merged_correlations['Correlation INR'])

# Create scatter plot
scatter = ax2.scatter(
    merged_correlations['Correlation Abagen'], 
    merged_correlations['Correlation INR'], 
    s=100, color=palette[0]
)

# Add bold label 'b' to the left of the second subplot
ax2.text(-0.05, 0.98, 'b', transform=ax2.transAxes, fontsize=24, fontweight='bold', va='top', ha='left')

# Add gene labels to each point
for i, row in merged_correlations.iterrows():
    x, y = row['Correlation Abagen'], row['Correlation INR']
    if row['Gene'] in red_genes:
        point = ax2.scatter(x, y, s=100, color='red', zorder=10)
        ax2.annotate(row['Gene'],
                     (x, y),
                     xytext=(7, 0), textcoords='offset points', 
                    color='red', fontweight='bold', fontsize=10)
    elif row['Gene'] in purple_genes:
        point = ax2.scatter(x, y, s=100, color='purple', zorder=11)
        ax2.annotate(row['Gene'],
                     (x, y),
                     xytext=(7, 0), textcoords='offset points', 
                    color='purple', fontweight='bold', fontsize=11)

ax2.set_xlabel('Correlation with Tau (Abagen)')
ax2.set_ylabel('Correlation with Tau (INR)')
ax2.set_title('Scatterplot of Gene Correlations with Tau between Abagen and INR Methods')
ax2.grid(axis='both', linestyle='--', alpha=0.7)

# Add correlation statistics
ax2.annotate(f'R = {r:.2f}, p = {p:.2e}', xy=(0.5, 0.9), xycoords='axes fraction', ha='center', fontsize=14)

# Add regression line
z = np.polyfit(merged_correlations['Correlation Abagen'], merged_correlations['Correlation INR'], 1)
p = np.poly1d(z)
xfit = np.linspace(merged_correlations['Correlation Abagen'].min(), merged_correlations['Correlation Abagen'].max(), 50)
yfit = p(xfit)
ax2.plot(xfit, yfit, "r--")

# Plot 3: Open Target correlation plot for Abagen
corr_abagen, p_abagen = pearsonr(abagen_with_global['Correlation'], abagen_with_global['globalScore'])
sns.scatterplot(x='Correlation', y='globalScore', data=abagen_with_global, 
                s=100, color='#4c72b0', ax=ax3)

# Add bold label 'c' to the left of the third subplot
ax3.text(-0.10, 0.98, 'c', transform=ax3.transAxes, fontsize=24, fontweight='bold', va='top', ha='left')

ax3.set_xlabel('Gene Correlation with Tau (Abagen)')
ax3.set_ylabel('Open Target Global Score')
ax3.set_title('Abagen Gene-Tau Correlation vs Open Target Score')
ax3.grid(axis='both', linestyle='--', alpha=0.7)
ax3.annotate(f'R = {corr_abagen:.2f}, p = {p_abagen:.2e}', 
             xy=(0.5, 0.9), xycoords='axes fraction', ha='center', fontsize=14)

# Add trend line for the Abagen open target plot
if len(abagen_with_global) > 1:
    z = np.polyfit(abagen_with_global['Correlation'], abagen_with_global['globalScore'], 1)
    p = np.poly1d(z)
    xfit = np.linspace(abagen_with_global['Correlation'].min(), abagen_with_global['Correlation'].max(), 50)
    yfit = p(xfit)
    ax3.plot(xfit, yfit, "r--")

# Plot 4: Open Target correlation plot for INR
corr_inr, p_inr = pearsonr(inr_with_global['Correlation'], inr_with_global['globalScore'])
sns.scatterplot(x='Correlation', y='globalScore', data=inr_with_global, 
                s=100, color='#55a868', ax=ax4)

# Add bold label 'd' to the left of the fourth subplot
ax4.text(-0.10, 0.98, 'd', transform=ax4.transAxes, fontsize=24, fontweight='bold', va='top', ha='left')

ax4.set_xlabel('Gene Correlation with Tau (INR)')
ax4.set_ylabel('Open Target Global Score')
ax4.set_title('INR Gene-Tau Correlation vs Open Target Score')
ax4.grid(axis='both', linestyle='--', alpha=0.7)
ax4.annotate(f'R = {corr_inr:.2f}, p = {p_inr:.2e}', 
             xy=(0.5, 0.9), xycoords='axes fraction', ha='center', fontsize=14)

# Add trend line for the INR open target plot
if len(inr_with_global) > 1:
    z = np.polyfit(inr_with_global['Correlation'], inr_with_global['globalScore'], 1)
    p = np.poly1d(z)
    xfit = np.linspace(inr_with_global['Correlation'].min(), inr_with_global['Correlation'].max(), 50)
    yfit = p(xfit)
    ax4.plot(xfit, yfit, "r--")

# Adjust the spacing between plots
plt.subplots_adjust(hspace=0.3)  # Reduce the height space between subplots

plt.savefig("./manuscript_imgs/combined_tau_and_opentarget_plots.pdf", bbox_inches='tight', format='pdf')
plt.show()