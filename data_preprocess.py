# %%
import pandas as pd
import numpy as np

# %% 
# For microarray and donor take mean
####################################
microarrays = pd.read_csv("./data/donor9861/MicroarrayExpression.csv",
                          header=None)
probe = pd.read_csv("./data/donor9861/Probes.csv")
microarrays = microarrays.drop(0, axis=1)

print(microarrays.shape)
print(microarrays.head())

microarrays['gene_symbol'] = probe['gene_symbol']
microarrays = microarrays.groupby('gene_symbol').mean()

print(microarrays.shape)
print(microarrays.head())

# save mean expression values
microarrays.to_csv("./data/donor9861/MicroarrayExpression_mean.csv")

# %%
# Merge pc1 result to sample annot with x y z
#############################################
df = pd.read_csv("data/pc1.csv")
annot = pd.read_csv("data/donor9861/SampleAnnot4d.csv")
df_long = df.melt(id_vars=['pc1', 'gene_symbol'], var_name='id', value_name='value')
df_long['id'] = pd.to_numeric(df_long['id']) - 1
annot = annot[["mni_x", "mni_y", "mni_z", "classification"]]
annot = annot.reset_index().rename(columns={'index': 'id'})

merged_df = pd.merge(df_long, annot, on='id')
merged_df = merged_df[['gene_symbol', 'id', 'mni_x', 'mni_y', 'mni_z', 'classification', 'value', 'pc1']]
merged_df = merged_df.sort_values(by=['gene_symbol', 'id'])
merged_df.to_csv("data/pc1_merged.csv", index=False)
merged_df
# %%
