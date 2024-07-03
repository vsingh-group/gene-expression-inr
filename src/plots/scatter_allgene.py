import pandas as pd
import matplotlib.pyplot as plt

# Load data
matter = "83_new"
df_abagen = pd.read_csv(f"./data/{matter}_interpolation_abagen.csv", index_col='label')
df_inravg = pd.read_csv(f"./data/{matter}_interpolation_inrs.csv", index_col='label')

# Check dimensions of the dataframes
print(df_abagen.shape, df_inravg.shape)

# Ensure df1 and df2 have the same columns
df_abagen = df_abagen[df_inravg.columns]

# Scatter plot
plt.figure(figsize=(15, 10))
for gene in df_abagen.columns:
    plt.scatter(df_abagen[gene], df_inravg[gene], label=gene, alpha=0.5)

plt.xlabel('Abagen Expression Values')
plt.ylabel('Inravg Expression Values')
plt.title('Scatter Plot of Gene Expression Values')

# Adjusting the legend to be outside the plot and in two columns
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', ncol=2)
plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust the plot area to make room for the legend

plt.grid(axis='both', linestyle='--', alpha=0.7)
plt.savefig("./results/scatter_genes.png", bbox_inches='tight', dpi=300)
plt.show()

# remove legend
plt.legend().remove()
plt.savefig("./results/scatter_genes_nolegend.png", bbox_inches='tight', dpi=300)


