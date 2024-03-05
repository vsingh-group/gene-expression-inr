import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigsh


with open("./data/gene_names.csv") as f:
    gene_names = f.readlines()
    gene_names = [x.strip() for x in gene_names]
    
gene_names = set(gene_names)

microarray_df = pd.read_csv("./data/donor9861/MicroarrayExpression_mean.csv",
                            header=0, index_col=0)

# find gene names not in microarry_df
missing_genes = gene_names - set(microarray_df.index)
print("Missing genes: ", missing_genes)
gene_names = gene_names - missing_genes
# use gene_names to get a gene dataframe
gene_names = list(gene_names)
gene_df = microarray_df.loc[gene_names]

# PCA
gene_df = gene_df.T
pca = PCA(n_components=1)
pca.fit(gene_df)
pc1_loadings = pca.components_[0]

gene_df_pca = gene_df.T
# add pc1_loadings to gene_df as a new column
gene_df_pca["pc1"] = pc1_loadings
# sort gene_df by pc1
gene_df_pca = gene_df_pca.sort_values(by="pc1", ascending=True)
# gene_df_pca = gene_df_pca[['pc1']]
# save gene_df to a new csv file
gene_df_pca.to_csv("./data/pc1.csv")


# Spectrum Embedding
gene_df_embedding = gene_df.T
embedding = SpectralEmbedding(n_components=1)
gene_embedding = embedding.fit_transform(gene_df_embedding)

gene_df_embedding["se"] = gene_embedding[:, 0].flatten()
gene_df_embedding = gene_df_embedding.sort_values(by="se", ascending=True)
# gene_df_embedding = gene_df_embedding[['embedding']]
gene_df_embedding.to_csv("./data/se.csv")
