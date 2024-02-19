import pandas as pd
import numpy as np

# Load the data
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