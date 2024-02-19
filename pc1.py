import pandas as pd
from sklearn.decomposition import PCA

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

gene_df = gene_df.T
pca = PCA(n_components=1)
pca.fit(gene_df)
pc1_loadings = pca.components_[0]

gene_df = gene_df.T
# add pc1_loadings to gene_df as a new column
gene_df["pc1"] = pc1_loadings
# sort gene_df by pc1
gene_df = gene_df.sort_values(by="pc1", ascending=True)
# save gene_df to a new csv file
gene_df.to_csv("./data/pc1.csv")