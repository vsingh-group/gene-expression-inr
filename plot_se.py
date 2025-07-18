# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding
from sklearn.preprocessing import StandardScaler, MinMaxScaler

donor = "9861"
matter = "83"

with open("./data/gene_names.csv") as f:
    gene_names = f.readlines()
    gene_names = [x.strip() for x in gene_names]
    
gene_names = set(gene_names)

microarray_df = pd.read_csv(f"./data/abagendata/train_{matter}/{matter}_microarray_{donor}.csv")
microarray_df = microarray_df.iloc[:, 1:].T
microarray_df.columns = microarray_df.iloc[0].astype(int).astype(str)
microarray_df = microarray_df.iloc[1:]
microarray_df.index.name = 'gene_symbol'

# find gene names not in microarry_df
missing_genes = gene_names - set(microarray_df.index)
print("Missing genes: ", missing_genes)
gene_names = gene_names - missing_genes
# use gene_names to get a gene dataframe
gene_names = list(gene_names)
gene_df = microarray_df.loc[gene_names]

# Spectrum Embedding
gene_df_embedding = gene_df.T
embedding = SpectralEmbedding(n_components=6)
gene_embedding = embedding.fit_transform(gene_df_embedding)

gene_df_embedding['se1'] = gene_embedding[:, 0]
gene_df_embedding['se2'] = gene_embedding[:, 1]
gene_df_embedding = gene_df_embedding[['se1', 'se2']]  # Only keep the embedding columns

# Optionally, you can scale the embedding values
scaler = MinMaxScaler()
gene_df_embedding[['se1', 'se2']] = scaler.fit_transform(gene_df_embedding[['se1', 'se2']])

plt.figure(figsize=(8, 6))
plt.scatter(gene_df_embedding['se1'], gene_df_embedding['se2'], alpha=0.6, edgecolors='w', linewidths=0.5)

# Remove x and y labels and ticks
plt.xticks([])
plt.yticks([])
plt.xlabel('')
plt.ylabel('')

# Thicken the plot frame
for spine in plt.gca().spines.values():
    spine.set_linewidth(3)

plt.grid(True)
plt.savefig("se.png", dpi=300, bbox_inches='tight')

# %%
