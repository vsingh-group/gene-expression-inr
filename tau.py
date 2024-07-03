import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

raw_tau = pd.read_csv('./ADNI_tau.csv')
region_meta = pd.read_csv('./atlas-desikankilliany1.csv')

result_inr = pd.read_csv('./result_83_9861_inrs_avg.csv', index_col=0)

result_abagen1 = pd.read_csv('outputs/abagen_sneha_output.csv')
result_abagen = pd.read_csv('./data/DK_Atlas_86_abagen_mapped_gene_exp_RcLcRsLs.csv')
# take same columns as result_inr from result_abagen
result_abagen = result_abagen[result_inr.columns]
result_abagen1 = result_abagen1[result_inr.columns]
result_inr = result_abagen1

# take only AD in raw_tau
raw_tau = raw_tau[raw_tau['merge_DX'].isin(['Dementia', 'MCI'])]
raw_tau = raw_tau[raw_tau['best_DX'].isin(['LMCI', 'AD'])]
raw_tau = raw_tau.iloc[:, 3:] # drop non value columns

raw_tau['D'] = (raw_tau['Left-Cerebellum-Cortex'] + raw_tau['Right-Cerebellum-Cortex']) / 2
new_tau = raw_tau.div(raw_tau['D'], axis=0)
new_tau = new_tau.sub(1, axis=0)
print(new_tau)

tau1 = region_meta['tau_id'].dropna().to_list()
tau2 = new_tau.columns.to_list()
print(set(tau2) - set(tau1))

# pick a gene that is known to have similar expression to a disease pattern.
# e.g. APOE, BIN1, MAPT and APP in Alzheimer disease.
# Then show the regional correlation between each such gene and the regional
# pattern of atrophy and/or tau of AD subjects.
# We have the latter and can share easily.

new_tau = new_tau.mean(axis=0).to_frame() # take mean for regions
new_tau.reset_index(level=0, inplace=True) # set index to column
new_tau.columns = ['tau_id', 'values'] # rename columns
res = pd.merge(region_meta, new_tau, on='tau_id', how='inner')
res = res[['id', 'values']]

def get_corr(result):
    merged_data = pd.merge(res, result, left_on='id', right_index=True)
    gene_columns = result.columns
    tau_values = merged_data['values']

    # Calculate correlation between each gene and tau values.
    correlations = {}
    for gene in gene_columns:
        correlations[gene] = merged_data[gene].corr(tau_values)
        
    correlation_df = pd.DataFrame(list(correlations.items()),
                                  columns=['Gene', 'Correlation'])
    return correlation_df

correlation_abagen = get_corr(result_abagen)
correlation_inr = get_corr(result_inr)




def plot_combined_correlation(df1, df2, title, filename):
    # Merge the two DataFrames on 'Gene'
    combined_df = pd.merge(df1, df2, on='Gene', suffixes=('_abagen_sneha', '_abagen_new_with_sneha_options'))
    
    # Melt the DataFrame to long format for seaborn plotting
    combined_df_long = combined_df.melt(id_vars='Gene', var_name='Dataset', value_name='Correlation')
    
    # Sort values for better visualization
    combined_df_long.sort_values('Correlation', ascending=False, inplace=True)

    # Set up the matplotlib figure
    plt.figure(figsize=(12, len(df1) * 0.4))

    # Create a seaborn barplot
    ax = sns.barplot(x='Correlation', y='Gene', hue='Dataset', data=combined_df_long, dodge=True)
    
    # Add informative title and labels
    plt.title(title)
    plt.xlabel('Correlation with Tau Pathology')
    plt.ylabel('Gene')
    
    # Add legend
    plt.legend()

    # Configure the x-ticks to appear on both top and bottom
    ax.xaxis.set_ticks_position('both')  # Set x-ticks to appear on both top and bottom
    ax.xaxis.set_label_position('bottom')  # Set x-label to appear on bottom only
    ax.tick_params(axis='x', which='both', labeltop=True)  # Enable labels on top

    # Adding light grey dashed lines to separate each row
    for i in range(len(combined_df_long['Gene'].unique())):
        ax.axhline(i - 0.5, color='grey', linestyle='--', linewidth=0.5)

    # Save the plot
    plt.savefig(filename, bbox_inches='tight')
    
    # Close the plot to free up memory
    plt.close()

# Example usage
plot_combined_correlation(correlation_abagen, correlation_inr, 'Comparison of Gene-Tau Correlations', 'comparison_tau_correlation.png')
