# %%
# Merge pc1 result to sample annot with x y z
#############################################
import pandas as pd
import numpy as np

def merge_5d(mode, donor, matter='83'):
    df = pd.read_csv(f"data/abagendata/train_{matter}/{mode}_{donor}.csv")
    annot = pd.read_csv(f"data/abagendata/train_{matter}/{matter}_annotation_{donor}_4d.csv")
    df_long = df.melt(id_vars=[mode, 'gene_symbol'], var_name='well_id', value_name='value')
    annot = annot[["mni_x", "mni_y", "mni_z", "classification", "well_id"]]
    
    df_long['well_id'] = df_long['well_id'].astype(str)
    annot['well_id'] = annot['well_id'].astype(str)
    
    print(df_long.shape, annot.shape)

    merged_df = pd.merge(df_long, annot, on='well_id', how='inner')
    print(merged_df)
    merged_df = merged_df[['gene_symbol', 'well_id', 'mni_x', 'mni_y', 'mni_z', 'classification', 'value', mode]]
    merged_df = merged_df.sort_values(by=['gene_symbol', 'well_id'])
    merged_df.to_csv(f"data/abagendata/train_{matter}/{mode}_{donor}_merged.csv", index=False)

merge_5d('se', '9861')
merge_5d('se', "10021")
merge_5d('se', "12876")
merge_5d('se', "14380")
merge_5d('se', "15496")
merge_5d('se', "15697")

# exec(open('./src/data/data_merge.py').read())