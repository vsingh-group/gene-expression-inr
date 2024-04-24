import pandas as pd
import matplotlib.pyplot as plt

donor = "9861"
df_abagen = pd.read_csv(f"./data/result_246_{donor}_abagen.csv", index_col='label')
df_inravg = pd.read_csv(f"./data/result_246_{donor}_inr_avg.csv", index_col='label')

# Assuming these DataFrames are structured correctly
df1 = df_abagen
df2 = df_inravg

correlation_df = pd.DataFrame(index=df1.columns, columns=['Correlation'])
for gene in df1.columns:
    correlation_df.loc[gene, 'Correlation'] = df1[gene].corr(df2[gene])
    
correlation_df['Correlation'] = correlation_df['Correlation'].astype(float)

# Creating the bar chart
plt.figure(figsize=(10, 8))
bars = plt.bar(correlation_df.index, correlation_df['Correlation'], color='skyblue')

plt.xlabel('Gene Names')
plt.ylabel('Correlation Coefficient')
plt.title('Correlation between INR Saperately to Abagen by Gene')
plt.xticks(rotation=90, fontsize=6)
plt.tight_layout()

# Adding gene names on top of each bar, rotated by 45 degrees
for bar, gene in zip(bars, correlation_df.index):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval, gene, ha='center', va='bottom', fontsize=6, rotation=90)

# Saving the plot to a PNG file
plt.savefig('./results/correlation_inr_plot.png', format='png', dpi=300)

plt.show()
