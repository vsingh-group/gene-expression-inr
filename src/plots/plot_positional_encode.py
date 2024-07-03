# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from modules import encode_df

# create a dataframe data range from 0, 1
df = pd.DataFrame({'se': np.linspace(0, 1.0, num=100)})
df = encode_df(df, 4)

df

fig, ax = plt.subplots(figsize=(10, 8))

# Plot each column except the 'se' column
for column in df.columns[1:]:  # skipping the 'se' column
    ax.plot(df['se'], df[column])

# Set the title and labels
# Add a legend to the plot
ax.legend()

# Show the plot
plt.show()
# %%
