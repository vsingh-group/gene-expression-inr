# %%
import wandb
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


STEPS = 500



def process_noisy_runs(project_name):
    api = wandb.Api()
    runs = api.runs(project_name)
    
    noisy_data_9861 = []
    noisy_data_10021 = []
    
    for run in runs:
        config = run.config
        if 'noise_percentage' not in config:
            continue
            
        noisy_loss = []
        clean_loss = []
        val_loss = []
        
        # Use enumerate to limit the number of rows processed
        for i, row in enumerate(run.scan_history(keys=['noisy_loss', 'clean_loss', 'val_loss'])):
            if i >= STEPS:  # Break after STEPS rows
                break
            noisy_loss.append(row['noisy_loss'])
            clean_loss.append(row['clean_loss'])
            val_loss.append(row['val_loss'])
        
        noisy_loss = np.array(noisy_loss)
        clean_loss = np.array(clean_loss)
        val_loss = np.array(val_loss)
        
        # Pad if less than STEPS
        if len(noisy_loss) < STEPS:
            noisy_loss = np.pad(noisy_loss, (0, STEPS - len(noisy_loss)), 'edge')
            clean_loss = np.pad(clean_loss, (0, STEPS - len(clean_loss)), 'edge')
            val_loss = np.pad(val_loss, (0, STEPS - len(val_loss)), 'edge')
        
        record = {
            'noise_percentage': config.get('noise_percentage'),
            'noisy_loss': noisy_loss,
            'clean_loss': clean_loss,
            'val_loss': val_loss,
            'test_loss': run.summary.get('test_loss')
        }
        
        if str(config.get('donor')) == '9861':
            noisy_data_9861.append(record)
        elif str(config.get('donor')) == '10021':
            noisy_data_10021.append(record)
    
    return pd.DataFrame(noisy_data_9861), pd.DataFrame(noisy_data_10021)


def process_wandb_runs(project_name):
    api = wandb.Api()
    runs = api.runs(project_name)
    
    data_9861 = []
    data_10021 = []
    
    for run in runs:
        config = run.config
        loss_values = []
        
        # Use enumerate to limit the number of rows processed
        for i, row in enumerate(run.scan_history(keys=['loss'])):
            if i >= STEPS:  # Break after STEPS rows
                break
            loss_values.append(row['loss'])
        
        loss_values = np.array(loss_values)
        print(len(loss_values))
        
        # Pad if less than STEPS
        if len(loss_values) < STEPS:
            loss_values = np.pad(loss_values, (0, STEPS - len(loss_values)), 'edge')
        
        record = {
            'lr': config.get('lr'),
            'encoding_dim': config.get('encoding_dim'),
            'hidden_features': config.get('hidden_features'),
            'hidden_layers': config.get('hidden_layers'),
            'loss_values': loss_values,
            'final_loss': loss_values[-1] if len(loss_values) > 0 else None
        }
        
        if str(config.get('donor')) == '9861':
            data_9861.append(record)
        elif str(config.get('donor')) == '10021':
            data_10021.append(record)
    
    return pd.DataFrame(data_9861), pd.DataFrame(data_10021)


def get_param_comparison(df, param_name):
    """Group data by parameter and return loss values for each unique value"""
    grouped = df.groupby(param_name)['loss_values'].apply(list)
    return {value: losses for value, losses in grouped.items()}


def get_noise_comparisons(df):
    grouped = df.groupby('noise_percentage')
    return {
        'noisy_loss': {noise: list(group['noisy_loss']) for noise, group in grouped},
        'clean_loss': {noise: list(group['clean_loss']) for noise, group in grouped},
        'val_loss': {noise: list(group['val_loss']) for noise, group in grouped},
        'test_loss': {noise: list(group['test_loss']) for noise, group in grouped}
    }


def get_all_comparisons(df):
    comparisons = {
        'lr': get_param_comparison(df, 'lr'),
        'encoding_dim': get_param_comparison(df, 'encoding_dim'),
        'hidden_features': get_param_comparison(df, 'hidden_features'),
        'hidden_layers': get_param_comparison(df, 'hidden_layers')
    }
    return comparisons


df_9861, df_10021 = process_wandb_runs("yuxizheng/donor_9861_siren")
df_9861_noisy, df_10021_noisy = process_noisy_runs("yuxizheng/donor_9861_siren_noisy")

# %%
plt.style.use('seaborn-v0_8-paper') # ggplot, classic, seaborn-v0_8-paper, seaborn-v0_8-poster, bmh

plt.rcParams.update({
    # 'font.family': 'serif',
    'font.size': 14,
    'axes.labelsize': 20,
    'axes.titlesize': 20,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 13,
    'figure.titlesize': 20,
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.5
})

def create_parameter_plot(ax, data_dict, param_name, title):
    colors = sns.color_palette('deep')
    for idx, (value, losses) in enumerate(data_dict.items()):
        steps = np.arange(1, len(losses[0]) + 1)
        mean_loss = np.mean(losses, axis=0)
        ax.plot(steps, mean_loss, label=f'{value}', color=colors[idx])
    
    ax.set_title(title, pad=10)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Loss')
    ax.legend(title=param_name, frameon=True, edgecolor='black')
    ax.grid(True, linestyle='--', alpha=0.5)

def create_noise_line_plot(ax, data_dict, title):
    colors = sns.color_palette('deep')
    for idx, (noise_level, losses) in enumerate(data_dict.items()):
        steps = np.arange(1, len(losses[0]) + 1)
        mean_loss = np.mean(losses, axis=0)
        ax.plot(steps, mean_loss, label=f'{noise_level}', color=colors[idx])
    
    ax.set_title(title, pad=10)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Loss')
    ax.legend(title="Noise Percentage", frameon=True, edgecolor='black')
    ax.grid(True, linestyle='--', alpha=0.5)

def create_noise_scatter_plot(ax, data_dict, title):
    noise_levels = []
    test_losses = []
    
    # Collect and clean data
    for noise_level, losses in data_dict.items():
        # Filter out None values and ensure numeric data
        valid_losses = [loss for loss in losses if loss is not None and np.isfinite(loss)]
        if valid_losses:  # Only add if we have valid losses
            noise_levels.extend([noise_level] * len(valid_losses))
            test_losses.extend(valid_losses)
    
    # Convert to numpy arrays for regression
    noise_levels = np.array(noise_levels)
    test_losses = np.array(test_losses)
    
    if len(noise_levels) > 0 and len(test_losses) > 0:
        # Create scatter plot
        ax.scatter(noise_levels, test_losses, alpha=0.6, label='Data points')
        
        try:
            # Add regression line
            z = np.polyfit(noise_levels, test_losses, 1)
            p = np.poly1d(z)
            x_reg = np.linspace(min(noise_levels), max(noise_levels), 100)
            y_reg = p(x_reg)
            ax.plot(x_reg, y_reg, 'r--', alpha=0.8)
            
        except Exception as e:
            print(f"Error fitting regression line for {title}: {e}")
            print(f"Number of data points: {len(noise_levels)}")
            print(f"Noise levels range: {min(noise_levels)} to {max(noise_levels)}")
    else:
        print(f"No valid data points for {title}")
    
    ax.set_title(title, pad=10)
    ax.set_xlabel('Noise Level')
    ax.set_ylabel('Test Loss')
    ax.grid(True, linestyle='--', alpha=0.5)

    
def plot_all_comparisons(df_9861, df_10021, df_9861_noisy, df_10021_noisy, save_path=None):
    fig, axs = plt.subplots(4, 4, figsize=(20, 20), dpi=300)
    
    # First two rows remain the same
    comparisons_9861 = get_all_comparisons(df_9861)
    comparisons_10021 = get_all_comparisons(df_10021)
    
    param_names = ['lr', 'encoding_dim', 'hidden_features', 'hidden_layers']
    titles_9861 = [
        'Learning Rate (Donor 9861)',
        'Encoding Dimension (Donor 9861)',
        'Feature Size (Donor 9861)',
        'Hidden Layer Depth (Donor 9861)'
    ]
    
    # Add row labels (a, b, c, d)
    labels = ['a', 'b', 'c', 'd']
    for idx, label in enumerate(labels):
        # Position the label on the left side of the first plot in each row
        axs[idx, 0].text(-0.2, 1.1, label, transform=axs[idx, 0].transAxes,
                        fontsize=20, fontweight='bold')
    
    for i, (param, title) in enumerate(zip(param_names, titles_9861)):
        create_parameter_plot(axs[0, i], comparisons_9861[param], param, title)
    
    titles_10021 = [
        'Learning Rate (Donor 10021)',
        'Encoding Dimension (Donor 10021)',
        'Feature Size (Donor 10021)',
        'Hidden Layer Depth (Donor 10021)'
    ]
    
    for i, (param, title) in enumerate(zip(param_names, titles_10021)):
        create_parameter_plot(axs[1, i], comparisons_10021[param], param, title)
    
    # Add noisy data plots
    noise_comparisons_9861 = get_noise_comparisons(df_9861_noisy)
    noise_comparisons_10021 = get_noise_comparisons(df_10021_noisy)
    
    # Third row: Donor 9861 noisy
    noise_titles_9861 = [
        'Training Loss Clean (9861)',
        'Training Loss Noisy (9861)',
        'Validation Loss (9861)',
        'Final Loss Comparison (9861)'
    ]
    
    create_noise_line_plot(axs[2, 0], noise_comparisons_9861['clean_loss'], noise_titles_9861[0])
    create_noise_line_plot(axs[2, 1], noise_comparisons_9861['noisy_loss'], noise_titles_9861[1])
    create_noise_line_plot(axs[2, 2], noise_comparisons_9861['val_loss'], noise_titles_9861[2])
    create_noise_scatter_plot(axs[2, 3], noise_comparisons_9861['test_loss'], noise_titles_9861[3])
    
    # Fourth row: Donor 10021 noisy
    noise_titles_10021 = [
        'Training Loss Clean (10021)',
        'Training Loss Noisy (10021)',
        'Validation Loss (10021)',
        'Final Loss Comparison (10021)'
    ]
    
    create_noise_line_plot(axs[3, 0], noise_comparisons_10021['clean_loss'], noise_titles_10021[0])
    create_noise_line_plot(axs[3, 1], noise_comparisons_10021['noisy_loss'], noise_titles_10021[1])
    create_noise_line_plot(axs[3, 2], noise_comparisons_10021['val_loss'], noise_titles_10021[2])
    create_noise_scatter_plot(axs[3, 3], noise_comparisons_10021['test_loss'], noise_titles_10021[3])
    # axs[2, 3].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    plt.tight_layout()
    
    if save_path:
        # Save both PNG and SVG versions
        # png_path = f"{base_name}_{STEPS}.png"
        pdf_path = f"{save_path}_{STEPS}.pdf"
        
        # plt.savefig(png_path, bbox_inches='tight', dpi=300)
        plt.savefig(pdf_path, bbox_inches='tight', format='pdf')
    
    return fig, axs

# Usage
fig, axs = plot_all_comparisons(df_9861, df_10021, df_9861_noisy, df_10021_noisy, save_path='./manuscript_imgs/parameter_comparison')
plt.show()



# %%
