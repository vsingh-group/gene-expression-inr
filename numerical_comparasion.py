import nibabel as nib
import numpy as np
import pandas as pd
import os
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from tqdm import tqdm

# Function to calculate R and MSE between two NIfTI files
def calculate_r_mse(file1, file2):
    # Load NIfTI files
    img1 = nib.load(file1).get_fdata().ravel()
    img2 = nib.load(file2).get_fdata().ravel()
    
    # Calculate Pearson correlation (R)
    R, _ = pearsonr(img1, img2)
    
    # Calculate Mean Squared Error (MSE)
    MSE = mean_squared_error(img1, img2)
    
    return R, MSE

# Function to generate simplified LaTeX formatted gene data
def generate_simplified_gene_table(df_r, df_mse, output_file='gene_data.tex'):
    """
    Generate simplified LaTeX formatted gene data with R and MSE values.
    Format: genename & R1 & MSE1 & R2 & MSE2 & R3 & MSE3 \\
    
    Parameters:
    - df_r: DataFrame with R values (genes as index, methods as columns)
    - df_mse: DataFrame with MSE values (genes as index, methods as columns)
    - output_file: File to save the formatted data
    """
    # Ensure both dataframes have the same indices and columns
    assert all(df_r.index == df_mse.index), "DataFrames must have the same indices"
    assert all(df_r.columns == df_mse.columns), "DataFrames must have the same columns"
    
    # Get method names and genes
    methods = df_r.columns
    genes = df_r.index
    
    # Open file for writing
    with open(output_file, 'w') as f:
        # For each gene, write a line in the requested format
        for gene in genes:
            line = gene
            
            # Add R and MSE values for each method
            for method in methods:
                r_value = df_r.loc[gene, method]
                mse_value = df_mse.loc[gene, method]
                line += f" & {r_value:.6f} & {mse_value:.6f}"
            
            # Add the line ending
            line += " \\\\"
            
            # Write to file
            f.write(line + "\n")
    
    print(f"Simplified gene data saved to {output_file}")
    return output_file

# Function to calculate and print mean ± std summary
def print_mean_std_summary(df_r, df_mse):
    """
    Calculate and print the mean ± std summary for R and MSE values.
    
    Parameters:
    - df_r: DataFrame with R values
    - df_mse: DataFrame with MSE values
    """
    # Calculate mean and std
    avg_r = df_r.mean()
    std_r = df_r.std()
    avg_mse = df_mse.mean()
    std_mse = df_mse.std()
    
    # Print summary
    print("\nSummary Table with Mean ± Standard Deviation:")
    print("Method        R                    MSE")
    print("--------------------------------------------")
    
    for method in avg_r.index:
        r_mean = avg_r[method]
        r_std = std_r[method]
        mse_mean = avg_mse[method]
        mse_std = std_mse[method]
        print(f"{method:12} {r_mean:.4f} ± {r_std:.4f}    {mse_mean:.4f} ± {mse_std:.4f}")

# Main script execution
if __name__ == "__main__":
    # Path templates
    abagen_dir = "/home/xizheng/brain-inr/result_ibf_2full+mirror_TT/nii_abagen/"
    path_template_abagen = os.path.join(abagen_dir, "{}_83_new_abagen.nii.gz")
    path_template_inr = "/home/xizheng/brain-inr/result_ibf_2full+mirror_TT/nii_9861_83_new_sep/{}_83_new_inr.nii.gz"
    path_template_inrs = "/home/xizheng/brain-inr/result_ibf_2full+mirror_TT/nii_9861_83_new/{}_83_new_inrs.nii.gz"
    path_template_inrs_avg = "/home/xizheng/brain-inr/result_ibf_2full+mirror_TT/nii_9861_83_new/{}_83_new_inrs_avg.nii.gz"

    # Get the list of available abagen files first
    abagen_files = os.listdir(abagen_dir)
    print(f"Number of files in the abagen directory: {len(abagen_files)}")

    # Extract gene names from available abagen files
    available_genes = []
    for file in abagen_files:
        if file.endswith("_83_new_abagen.nii.gz"):
            gene_name = file.replace("_83_new_abagen.nii.gz", "")
            available_genes.append(gene_name)

    print(f"Found {len(available_genes)} genes with abagen files")

    # Initialize results lists and valid genes list
    results_r = []
    results_mse = []
    valid_genes = []

    # Process each available gene with tqdm progress bar
    for gene in tqdm(available_genes, desc="Processing genes"):
        # Get file paths
        file_abagen = path_template_abagen.format(gene)
        file_inr = path_template_inr.format(gene)
        file_inrs = path_template_inrs.format(gene)
        file_inrs_avg = path_template_inrs_avg.format(gene)
        
        # Check if all other required files exist
        additional_files = [file_inr, file_inrs, file_inrs_avg]
        all_files_exist = all(os.path.isfile(file) for file in additional_files)
        
        if all_files_exist:
            r_values = []  # To store R values for the current gene
            mse_values = [] # To store MSE values for the current gene
            
            # Calculate R and MSE for each comparison
            try:
                for file_compare in additional_files:
                    R, MSE = calculate_r_mse(file_abagen, file_compare)
                    r_values.append(R)
                    mse_values.append(MSE)
                
                # Append results only if all calculations were successful
                results_r.append(r_values)
                results_mse.append(mse_values)
                valid_genes.append(gene)
            except Exception as e:
                print(f"Error processing gene {gene}: {e}")
        else:
            # Report which files are missing
            missing_files = [file for file in additional_files if not os.path.isfile(file)]
            print(f"Skipping gene {gene} - missing files: {', '.join([os.path.basename(f) for f in missing_files])}")

    # Create pandas DataFrame with only valid genes
    columns = ['INR', 'INR_Spectral', 'INR_Spectral_Avg']
    df_r = pd.DataFrame(results_r, index=valid_genes, columns=columns)
    df_mse = pd.DataFrame(results_mse, index=valid_genes, columns=columns)

    # Output the number of valid genes processed
    print(f"\nSuccessfully processed {len(valid_genes)} out of {len(available_genes)} genes with abagen files")

    # Output the results to verify
    print("\nR Values Table:")
    print(df_r)
    print("\nMSE Values Table:")
    print(df_mse)

    # Print the mean ± std summary
    print_mean_std_summary(df_r, df_mse)

    # Save to CSV files
    df_r.to_csv('R_values_comparison.csv')
    df_mse.to_csv('MSE_values_comparison.csv')

    # Save the list of valid genes
    with open('valid_genes.txt', 'w') as f:
        for gene in valid_genes:
            f.write(f"{gene}\n")
    
    # Generate simplified LaTeX formatted gene data
    simplified_table = generate_simplified_gene_table(df_r, df_mse, 'gene_data.tex')
    print("\nSimplified gene data LaTeX format saved successfully!")