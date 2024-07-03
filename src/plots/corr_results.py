import pandas as pd
import matplotlib.pyplot as plt

donors = ['9861', '10021']
matter = "83_new"
atlas = 'atlas-desikankilliany'
all_results = True

# Create a figure and a set of subplots
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 10))
axes = axes.flatten()

for i, donor in enumerate(donors):
    df_abagen = pd.read_csv(f"./data/{matter}_interpolation_abagen.csv", index_col='label')
    df_inravg = pd.read_csv(f"./data/{matter}_interpolation_inrsn.csv", index_col='label')
    # df_inravg = pd.read_csv(f"./result_ibf_2full+mirror_TT/result_{matter}_{donor}_inrs_avg.csv", index_col='label')

    # Assuming these DataFrames are structured correctly
    df1 = df_abagen
    df2 = df_inravg
    
    print(df1.shape, df2.shape)
    df1 = df1[df2.columns]

    correlation_df = pd.DataFrame(index=df1.columns, columns=['Correlation'])
    for gene in df1.columns:
        correlation_df.loc[gene, 'Correlation'] = df1[gene].corr(df2[gene])
    
    correlation_df['Correlation'] = correlation_df['Correlation'].astype(float)
    
    # Calculate average correlation
    avg_correlation = correlation_df['Correlation'].mean()

    # Creating the bar chart for the subplot
    ax = axes[i]
    bars = ax.bar(correlation_df.index, correlation_df['Correlation'], color='skyblue')
    ax.set_xlabel('Gene Names')
    ax.set_ylabel('Correlation Coefficient')
    ax.set_title(f'Donor {donor} (Avg Corr: {avg_correlation:.2f})')

    ax.tick_params(axis='x', rotation=90, labelsize=6)
    ax.tick_params(axis='y', labelsize=8)

    # Adding gene names on top of each bar, rotated by 90 degrees
    for bar, gene in zip(bars, correlation_df.index):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval, gene, ha='center', va='bottom', fontsize=6, rotation=90)

plt.tight_layout()

# Saving the plot to a PNG file
plt.savefig(f'./results/corr_barplot.png', format='png', dpi=300)

plt.show()
