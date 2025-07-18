import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import SpectralEmbedding
from scipy.stats import pearsonr
import numpy as np

plt.style.use('seaborn-v0_8-paper')

# Update plotting parameters
plt.rcParams.update({
    # 'font.family': 'science',
    'font.size': 10,
    'axes.labelsize': 20,
    'axes.titlesize': 20,
    'xtick.labelsize': 10,
    'ytick.labelsize': 12,
    'legend.fontsize': 8,
    'figure.titlesize': 20,
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.5
})


# Load data
matter = "83_new"
df_abagen = pd.read_csv(f"./data/{matter}_interpolation_abagen.csv", index_col='label')
df_inravg = pd.read_csv(f"./data/{matter}_interpolation_inrs.csv", index_col='label')

# option 1, get the embedding from result data
gene_df_embedding = df_inravg.T
embedding = SpectralEmbedding(n_components=1)
gene_embedding = embedding.fit_transform(gene_df_embedding)
gene_df_embedding["se"] = gene_embedding[:, 0].flatten()
gene_df_embedding = gene_df_embedding.sort_values(by="se", ascending=True)
print(gene_df_embedding.head())
df_inravg = gene_df_embedding.drop('se', axis=1).T

# Check dimensions of the dataframes
print(df_abagen.shape, df_inravg.shape)

# Ensure df1 and df2 have the same columns
df_abagen = df_abagen[df_inravg.columns]

# Compute correlations
correlation_df = pd.DataFrame(index=df_abagen.columns, columns=['Correlation'])
for gene in df_abagen.columns:
    correlation_df.loc[gene, 'Correlation'] = df_abagen[gene].corr(df_inravg[gene])

correlation_df['Correlation'] = correlation_df['Correlation'].astype(float)

# Calculate average correlation
avg_correlation = correlation_df['Correlation'].mean()
print(f"Average Correlation: {avg_correlation}")

# Plotting
fig, axs = plt.subplots(2, 1, figsize=(14, 16))

# Bar plot
correlation_df['Correlation'].plot(kind='bar', ax=axs[0], color='#4c72b0')
axs[0].set_title(f'Gene Correlation between Abagen and INR Bar Plot (Average Correlation: {avg_correlation:.2f})')
axs[0].set_xlabel('Genes (Ordered by Spectral Embedding)')
axs[0].set_ylabel('Correlation')
axs[0].grid(axis='y', linestyle='--', alpha=0.7)
axs[0].grid(axis='x', linestyle='--', alpha=0.7)

# Add bold label 'a' to the left of the first subplot
axs[0].text(-0.05, 0.98, 'a', transform=axs[0].transAxes, fontsize=24, fontweight='bold', va='top', ha='left')

# Scatter plot
for gene in df_abagen.columns:
    axs[1].scatter(df_abagen[gene], df_inravg[gene], alpha=0.5, color='#4c72b0')

axs[1].set_xlabel('Abagen Expression Values')
axs[1].set_ylabel('Inravg Expression Values')
axs[1].set_title('Scatter Plot of Gene Expression Values')

# Add bold label 'b' to the left of the second subplot
axs[1].text(-0.05, 0.98, 'b', transform=axs[1].transAxes, fontsize=24, fontweight='bold', va='top', ha='left')

# Calculate R and p values
r, p = pearsonr(df_abagen.mean(axis=1), df_inravg.mean(axis=1))

# Annotate R and p values
axs[1].annotate(f'R = {r:.2f}, p = {p:.2e}', xy=(0.5, 0.9), xycoords='axes fraction', ha='center', fontsize=14)

# Add a red fitted line
z = np.polyfit(df_abagen.mean(axis=1), df_inravg.mean(axis=1), 1)
p = np.poly1d(z)
xfit = np.linspace(df_abagen.min().min(), df_abagen.max().max(), 50)
yfit = p(xfit)
axs[1].plot(xfit, yfit, "r--")
axs[1].grid(axis='both', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig("./manuscript_imgs/combined_inr_vs_abg_plots.pdf", bbox_inches='tight', format='pdf')

plt.close()

gene_order = gene_df_embedding.index.to_list()
print(gene_order)
print(len(gene_order))

# Round SE values to 4 decimal places
gene_df_embedding_formatted = gene_df_embedding[['se']].round(4)

# Combine gene_symbol and se with "&"
gene_df_embedding_formatted["combined"] = gene_df_embedding_formatted.index + " & " + gene_df_embedding_formatted["se"].astype(str)

df = gene_df_embedding_formatted.drop(columns=["se"])
# drop index column
df = df.reset_index(drop=True)

columns = {}
for i in range(4):
    col_data = df.iloc[i*25:(i+1)*25].reset_index(drop=True)
    columns[f'col{i+1}'] = col_data['combined'].to_list()


reshaped_df = pd.DataFrame(columns)
reshaped_df['col4'] = reshaped_df['col4'].astype(str) + ' \\\\'

# save
reshaped_df.to_csv('./data/83_new_gene_order_by_se.csv', index=False)