import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def get_pc1_se_results(donor, matter="83"):
    with open("./data/gene_names.csv") as f:
        gene_names = f.readlines()
        gene_names = [x.strip() for x in gene_names]
        
    gene_names = set(gene_names)

    microarray_df = pd.read_csv(f"./data/abagendata/train_{matter}_new/{matter}_microarray_{donor}.csv")
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
    gene_df_pca.to_csv(f"./data/abagendata/train_{matter}_new/pc1_{donor}.csv")


    # Spectrum Embedding
    gene_df_embedding = gene_df.T
    embedding = SpectralEmbedding(n_components=1)
    gene_embedding = embedding.fit_transform(gene_df_embedding)

    gene_df_embedding["se"] = gene_embedding[:, 0].flatten()
    gene_df_embedding = gene_df_embedding.sort_values(by="se", ascending=True)
    # gene_df_embedding = gene_df_embedding[['embedding']]
    scaler = MinMaxScaler()
    gene_df_embedding['se'] = scaler.fit_transform(gene_df_embedding[['se']])
    gene_df_embedding.to_csv(f"./data/abagendata/train_{matter}_new/se_{donor}.csv")
    

get_pc1_se_results("9861")
get_pc1_se_results("10021")
# get_pc1_se_results("12876")
# get_pc1_se_results("14380")
# get_pc1_se_results("15496")
# get_pc1_se_results("15697")

# exec(open('./src/data/pc1_se.py').read())
