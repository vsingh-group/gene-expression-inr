import pandas as pd
import matplotlib.pyplot as plt


donor = "9861"
df_abagen = pd.read_csv(f"./data/result_246_{donor}_abagen.csv", index_col='label')
df_inravg = pd.read_csv(f"./data/result_246_{donor}_inravg.csv", index_col='label')
df1 = df_abagen
df2 = df_inravg

correlation_df = pd.DataFrame(index=df1.columns, columns=['Correlation'])
for gene in df1.columns:
    correlation_df.loc[gene, 'Correlation'] = df1[gene].corr(df2[gene])
    
correlation_df['Correlation'] = correlation_df['Correlation'].astype(float)
# print(correlation_df)

plt.figure(figsize=(10, 8))  # You can adjust the size as per your need
plt.bar(correlation_df.index, correlation_df['Correlation'], color='skyblue')
plt.xlabel('Gene Names')
plt.ylabel('Correlation Coefficient')
plt.title('Correlation between Two DataFrames by Gene')
plt.xticks(rotation=90, fontsize=6)  # Rotate and set font size smaller
plt.tight_layout()  # Adjusts plot parameters to give some padding

# Saving the plot to a PNG file
plt.savefig('./results/correlation_plot.png', format='png', dpi=300)  # Adjust dpi for resolution quality
