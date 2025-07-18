# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Embedder:

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.**np.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = np.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        # Assuming inputs is a numpy array, not a tensor
        return np.concatenate([fn(inputs) for fn in self.embed_fns], axis=-1)

def get_embedder(multires, i=0):
    if i == -1:
        return lambda x: x, 3

    embed_kwargs = {
        'include_input': False,
        'input_dims': 1,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [np.sin, np.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): 
        return eo.embed(x)
    return embed, embedder_obj.out_dim

def encode_df(df, multires):
    # position embedding
    embed, _ = get_embedder(multires=multires)
    embedded_data = embed(df[["se"]])
    # add embedded_data to gene_df as new columns
    for i in range(embedded_data.shape[1]):
        p_f = 'sin' if i % 2 == 0 else 'cos'
        df[f"se_{p_f}{i}"] = embedded_data[:, i]
    
    return df

# create a dataframe data range from 0, 1
df = pd.DataFrame({'se': np.linspace(0, 1.0, num=100)})
df = encode_df(df, 4)

fig, ax = plt.subplots(figsize=(8, 6))

# Plot each column except the 'se' column
for column in df.columns[1:]:  # skipping the 'se' column
    ax.plot(df['se'], df[column], linewidth=3)
    
for spine in ax.spines.values():
    spine.set_linewidth(3)
    
# remove ticks
ax.set_xticks([])
ax.set_yticks([])

ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

# Show the plot
plt.savefig('positional_encode.png', dpi=300, bbox_inches='tight')
# %%
