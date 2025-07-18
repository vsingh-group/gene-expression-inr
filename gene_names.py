import pandas as pd
import os

# Load processed gene names from the PC1 files (or SE files)
se_9861 = pd.read_csv("./data/abagendata/train_83_new/se_9861.csv", index_col=0)
se_10021 = pd.read_csv("./data/abagendata/train_83_new/se_10021.csv", index_col=0)

# Get gene names (index)
genes_9861 = set(se_9861.index)
genes_10021 = set(se_10021.index)

# Intersection
common_genes = genes_9861 & genes_10021
print(f"Number of common genes: {len(common_genes)}")

# Optional: save to file
pd.Series(sorted(list(common_genes))).to_csv("common_genes_9861_10021.csv", index=False, header=False)
