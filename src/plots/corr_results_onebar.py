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

# Compute correlations
correlation_df = pd.DataFrame(index=df_abagen.columns, columns=['Correlation'])
for gene in df_abagen.columns:
    correlation_df.loc[gene, 'Correlation'] = df_abagen[gene].corr(df_inravg[gene])

correlation_df['Correlation'] = correlation_df['Correlation'].astype(float)

# Calculate average correlation
avg_correlation = correlation_df['Correlation'].mean()
print(f"Average Correlation: {avg_correlation}")

# Plotting
plt.figure(figsize=(14, 8))
correlation_df['Correlation'].plot(kind='bar', color='#4c72b0')
plt.title(f'Gene Correlation between Abagen and INR Bar Plot (Average Correlation: {avg_correlation:.2f})')
plt.xlabel('Genes')
plt.ylabel('Correlation')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("./results/corr_barplot.png", bbox_inches='tight', dpi=300)
plt.show()

# add a histogram of correlation
plt.figure(figsize=(14, 8))
plt.hist(correlation_df['Correlation'], bins=30, color='#4c72b0', edgecolor='black')
plt.title(f'Gene Correlation between Abagen and INR Histogram (Average Correlation: {avg_correlation:.2f})')
plt.xlabel('Genes')
plt.ylabel('Correlation')
plt.grid(axis='both', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("./results/corr_hist.png", bbox_inches='tight', dpi=300)