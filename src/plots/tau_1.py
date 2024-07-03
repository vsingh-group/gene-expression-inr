import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

donor_list = ['9861', '10021', '12876', '14380', '15496', '15697']
donor_list = ['9861', '10021']

raw_tau = pd.read_csv('./ADNI_tau.csv')
region_meta = pd.read_csv('./atlas-desikankilliany-meta.csv')

# dfs = []
# for donor in donor_list:
#     dfs.append(pd.read_csv(f'./data/result_83_new_{donor}_inrs_avg.csv', index_col=0))
# # take average of all donors
# result_inr = pd.concat(dfs).groupby(level=0).mean()
# result_inr.to_csv('./data/83_new_interpolation_inrs.csv')
result_inr = pd.read_csv('./data/83_new_interpolation_inrs.csv', index_col=0)
# make abagen have same label as inr
result_abagen = pd.read_csv('./data/83_new_interpolation_abagen.csv')
result_abagen.index = result_inr.index
result_abagen = result_abagen[result_inr.columns.to_list()]
# take only columns that are in both datasets

# take only AD in raw_tau
raw_tau = raw_tau[raw_tau['merge_DX'].isin(['Dementia', 'MCI'])]
raw_tau = raw_tau[raw_tau['best_DX'].isin(['LMCI', 'AD'])]
raw_tau = raw_tau.iloc[:, 3:] # drop non value columns

raw_tau['D'] = (raw_tau['Left-Cerebellum-Cortex'] + raw_tau['Right-Cerebellum-Cortex']) / 2
new_tau = raw_tau.div(raw_tau['D'], axis=0)
new_tau = new_tau.sub(1, axis=0)
# remove D
new_tau = new_tau.drop(columns=['D'])

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
new_tau.columns = ['tau_id', 'tau'] # rename columns
res = pd.merge(region_meta, new_tau, on='tau_id', how='inner')
res = res[['id', 'tau']]
res.index = res['id']
res = res.drop(columns=['id'])

print(res.head())
print(res.shape, result_abagen.shape, result_inr.shape)

def get_corr(result):
    #merge on index
    merged_data = pd.merge(res, result, left_index=True, right_index=True)
    print(merged_data.shape)
    gene_columns = result.columns
    tau_values = merged_data['tau']

    # Calculate correlation between each gene and tau values.
    correlations = {}
    for gene in gene_columns:
        correlations[gene] = merged_data[gene].corr(tau_values)
        
    correlation_df = pd.DataFrame(list(correlations.items()),
                                  columns=['Gene', 'Correlation'])
    return correlation_df

correlation_abagen = get_corr(result_abagen)
correlation_inr = get_corr(result_inr)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming correlation_abagen and correlation_inr are the correlation dataframes obtained from the get_corr function

# Merge the correlation dataframes on the 'Gene' column
merged_correlations = pd.merge(correlation_abagen, correlation_inr, on='Gene', suffixes=(' Abagen', ' INR'))

# Melt the dataframe for easier plotting with seaborn
melted_correlations = pd.melt(merged_correlations, id_vars=['Gene'], 
                              value_vars=['Correlation Abagen', 'Correlation INR'],
                              var_name='Method', value_name='Correlation')

# Define a professional color palette
palette = sns.color_palette(['#4c72b0', '#55a868'])  # Blue and green shades

# Plot the bar plot
plt.figure(figsize=(14, 8))
sns.barplot(x='Gene', y='Correlation', hue='Method', data=melted_correlations, palette=palette)

# Add labels and title
plt.xlabel('Gene')
plt.ylabel('Correlation with Tau')
plt.title('Comparison of Gene Correlations with Tau between Abagen and INR Methods')
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability

# Customize legend
plt.legend(title='Method', loc='upper right')

plt.tight_layout()  # Adjust layout to make room for rotated x-axis labels
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.grid(axis='x', linestyle='--', alpha=0.7)

plt.savefig("tau_plot_1.png", bbox_inches='tight', dpi=300)
